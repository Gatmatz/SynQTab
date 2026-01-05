import logging
import warnings
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D  # Added for custom legend handle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# --- Constants ---
N_GEN_SAMPLES = 1000
ERROR_RATE = 0.10
N_COLS_TO_CORRUPT = 3  # Balance impact by corrupting fixed number of cols per error
RANDOM_SEED = 42

class SynthcityModelOption(Enum):
    # MARGINAL_DISTRIBUTIONS = 'marginal_distributions'
    CTGAN = 'ctgan'
    NFLOW = 'nflow'
    ARF = 'arf'
    ADSGAN = 'adsgan'
    RTVAE = 'rtvae'
    TVAE = 'tvae'
    DDPM = 'ddpm' 
    PATEGAN = 'pategan'
    DPGAN = 'dpgan'

class SynthcitySettings:
    def __init__(
        self, 
        model_name: SynthcityModelOption,
        n_samples: int = 1000,
    ):
        self.model_name = model_name.value
        self.n_samples = n_samples

    def to_dict(self) -> dict:
        return vars(self)

class DataCorruptor:
    """
    Handles reproducible injection of realistic data errors.
    """
    def __init__(self, df, target_col='target', seed=RANDOM_SEED):
        self.df_original = df.copy()
        self.target_col = target_col
        self.n_rows = len(df)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Identify candidate columns (Excluding Target)
        features = df.drop(columns=[target_col])
        self.num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()
        self.all_cols = self.num_cols + self.cat_cols
        
        # Pre-select columns to corrupt for each error type
        self.target_cols_map = {
            'scale': self._pick_cols(self.num_cols),
            'shift': self._pick_cols(self.cat_cols),
            'missing': self._pick_cols(self.all_cols),
            'noise': self._pick_cols(self.num_cols)
        }
        
        # Pre-calculate row indices for each error type
        n_errors = int(self.n_rows * ERROR_RATE)
        shuffled_indices = self.rng.permutation(self.df_original.index)
        
        self.indices = {
            'scale': shuffled_indices[0:n_errors],
            'shift': shuffled_indices[n_errors:2*n_errors],
            'missing': shuffled_indices[2*n_errors:3*n_errors],
            'noise': shuffled_indices[3*n_errors:4*n_errors],
            'label_flip': shuffled_indices[4*n_errors:5*n_errors]
        }

    def _pick_cols(self, candidates):
        if not candidates: return []
        n = min(len(candidates), N_COLS_TO_CORRUPT)
        # Sort first to ensure reproducibility before choice
        candidates = sorted(candidates) 
        return list(self.rng.choice(candidates, size=n, replace=False))

    def _inject_scaling(self, df, indices):
        """Multiplies selected numerical columns by 10."""
        cols = self.target_cols_map['scale']
        if not cols: return df
        df.loc[indices, cols] *= 100
        return df

    def _inject_shifting(self, df, indices):
        """
        Shifts categorical values ensuring the new value is NEVER the same as the original.
        Uses a modular shift to guarantee corruption.
        """
        cols = self.target_cols_map['shift']
        if not cols: return df
        
        for col in cols:
            unique_vals = df[col].unique()
            n_unique = len(unique_vals)
            if n_unique < 2: continue
            
            # Map values to indices and back
            val_to_idx = {val: i for i, val in enumerate(unique_vals)}
            idx_to_val = {i: val for i, val in enumerate(unique_vals)}
            
            current_indices = df.loc[indices, col].map(val_to_idx).values
            
            # Generate random offset k such that 1 <= k < n_unique
            offset = self.rng.integers(1, n_unique)
            new_indices = (current_indices + offset) % n_unique
            
            df.loc[indices, col] = [idx_to_val[idx] for idx in new_indices]
        return df

    def _inject_missing(self, df, indices):
        cols = self.target_cols_map['missing']
        if not cols: return df
        for col in cols:
            if col in self.num_cols:
                df.loc[indices, col] = -1
            else:
                df.loc[indices, col] = "?"
        return df

    def _inject_noise(self, df, indices):
        cols = self.target_cols_map['noise']
        if not cols: return df
        for col in cols:
            std_dev = df[col].std()
            scale_factor = self.rng.uniform(0.1, 2.0)
            noise = self.rng.normal(loc=0, scale=scale_factor * std_dev, size=len(indices))
            df.loc[indices, col] += noise
        return df
        
    def _inject_label_flipping(self, df, indices):
        target = self.target_col
        unique_labels = df[target].unique()
        if len(unique_labels) < 2: return df
        
        current_vals = df.loc[indices, target]
        def get_other_label(val):
            options = [x for x in unique_labels if x != val]
            return self.rng.choice(options)
        df.loc[indices, target] = current_vals.apply(get_other_label)
        return df

    def get_corrupted_data(self, active_errors: list):
        df_curr = self.df_original.copy()
        if 'scale' in active_errors: df_curr = self._inject_scaling(df_curr, self.indices['scale'])
        if 'shift' in active_errors: df_curr = self._inject_shifting(df_curr, self.indices['shift'])
        if 'noise' in active_errors: df_curr = self._inject_noise(df_curr, self.indices['noise'])
        if 'missing' in active_errors: df_curr = self._inject_missing(df_curr, self.indices['missing'])
        if 'label_flip' in active_errors: df_curr = self._inject_label_flipping(df_curr, self.indices['label_flip'])
        return df_curr

def clean_placeholder_rows(df: pd.DataFrame, placeholders: list):
    mask = df.isin(placeholders)
    rows_with_placeholder_mask = mask.any(axis=1)
    filtered_df = df[~rows_with_placeholder_mask].copy()
    return filtered_df

def read_adult_df(file_path: str = "adult.data"):
    try:
        df = pd.read_csv(
            file_path, header=0, index_col=False, skipinitialspace=True, 
            names=[
                'age', 'workclass', 'fnlwat', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'target'
            ]
        )
        return df
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        raise

class SynthcityGenerator:
    def __init__(self, settings: SynthcitySettings):
        self.settings = settings.to_dict()
        self.generator = None
    
    def generate(self, X_initial, y_initial):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target_name = y_initial.columns[0] if isinstance(y_initial, pd.DataFrame) else y_initial.name
            loader = GenericDataLoader(
                pd.concat([X_initial, y_initial], axis=1),
                target_column=target_name
            )
            self.generator = Plugins().get(self.settings["model_name"])
            self.generator.fit(loader)
            return self.generator.generate(count=self.settings["n_samples"]).dataframe()

def evaluate_utility(synthetic_df, real_test_df, target_col='target'):
    syn_df_enc = synthetic_df.copy()
    real_test_enc = real_test_df.copy()
    syn_df_enc.replace(['?', -1], np.nan, inplace=True)
    real_test_enc.replace(['?', -1], np.nan, inplace=True)

    for col in syn_df_enc.columns:
        is_cat = syn_df_enc[col].dtype == 'object' or real_test_enc[col].dtype == 'object'
        if is_cat:
            fill_val = 'missing_value'
            syn_df_enc[col] = syn_df_enc[col].fillna(fill_val).astype(str)
            real_test_enc[col] = real_test_enc[col].fillna(fill_val).astype(str)
            vocab = set(syn_df_enc[col].unique()) | set(real_test_enc[col].unique())
            le = LabelEncoder()
            le.fit(list(vocab))
            syn_df_enc[col] = le.transform(syn_df_enc[col])
            real_test_enc[col] = le.transform(real_test_enc[col])
        else:
            median_val = syn_df_enc[col].median()
            syn_df_enc[col] = syn_df_enc[col].fillna(median_val)
            real_test_enc[col] = real_test_enc[col].fillna(median_val)

    X_train = syn_df_enc.drop(columns=[target_col])
    y_train = syn_df_enc[target_col]
    X_test = real_test_enc.drop(columns=[target_col])
    y_test = real_test_enc[target_col]
    
    clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return f1_score(y_test, preds, average='weighted')

# --- NEW COMPREHENSIVE PLOTTING FUNCTION ---
def plot_comprehensive_results(all_results):
    """
    Generates a grouped bar chart with patterns, colors, and 'winner' markers.
    Mimics the style of the user-provided reference image.
    all_results structure: { 'ModelName': { 'ScenarioName': score, ... }, ... }
    """
    # 1. Setup Data for Plotting
    models = list(all_results.keys())
    
    # Extract scenarios from the first model (assuming all have same scenarios)
    # Filter out "Clean (Baseline)" for the bar plot
    sample_scenarios = list(all_results[models[0]].keys())
    scenarios_to_plot = [s for s in sample_scenarios if "Clean (Baseline)" not in s]
    
    n_models = len(models)
    n_scenarios = len(scenarios_to_plot)
    
    # 2. Define Styles (Colors and Hatches)
    styles = {
        "Clean Scaling":  {'color': '#3498db', 'hatch': '...'}, # Blue, dotted
        "Clean Shifting": {'color': '#f1c40f', 'hatch': 'xx'},  # Yellow, crossed
        "Clean Missing":  {'color': '#2ecc71', 'hatch': '\\\\'},# Green, back stripy
        "Clean Noise":    {'color': '#9b59b6', 'hatch': '++'},  # Purple, plus
        "Clean Labels":   {'color': '#34495e', 'hatch': '**'},  # Dark Blue, stars
        "No cleaning":    {'color': '#e74c3c', 'hatch': '///'}, # Red-ish, stripy
    }

    # 3. Create Figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Width settings
    bar_width = 0.12
    # Calculate group width to center labels
    total_group_width = n_scenarios * bar_width
    indices = np.arange(n_models)
    
    # 4. Plot Bars
    for i, scenario in enumerate(scenarios_to_plot):
        # Gather scores for this scenario across all models
        scores = [all_results[m].get(scenario, 0.0) for m in models]
        
        # Calculate position offset
        offset = (i - n_scenarios / 2) * bar_width + (bar_width / 2)
        positions = indices + offset
        
        style = styles.get(scenario, {'color': 'gray', 'hatch': ''})
        
        rects = ax.bar(
            positions, 
            scores, 
            bar_width, 
            label=scenario,
            color=style['color'],
            edgecolor='black',
            hatch=style['hatch'],
            alpha=0.85 
        )

    # 5. Add "Winner" Triangles AND Baseline Lines
    for idx, model in enumerate(models):
        # --- A. Winner Triangle Logic (Best of the bars) ---
        model_scores = {s: all_results[model][s] for s in scenarios_to_plot}
        best_scenario = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_scenario]
        
        # Find the x-position of this specific bar
        scenario_index = scenarios_to_plot.index(best_scenario)
        offset = (scenario_index - n_scenarios / 2) * bar_width + (bar_width / 2)
        x_pos = idx + offset
        
        # Draw red inverted triangle marker
        ax.plot(
            x_pos, 
            best_score + 0.02, # Slightly above bar
            marker='v', 
            markersize=16, 
            color='red', 
            markeredgecolor='black',
            linestyle='None',
        )
        
        # --- B. Baseline Line Logic (Horizontal line for each group) ---
        baseline_score = all_results[model].get("Clean (Baseline)", 0.0)
        
        # Determine start and end X for the horizontal line (covering the whole group)
        group_left_edge = idx - (total_group_width / 2)
        group_right_edge = idx + (total_group_width / 2)
        baseline_offset = 0.2
        
        ax.hlines(
            y=baseline_score, 
            xmin=group_left_edge - baseline_offset, 
            xmax=group_right_edge + baseline_offset, 
            colors='green', 
            linestyles='dashdot', 
            linewidth=2,
            zorder=5 # Ensure it sits above bars if they overlap
        )

    # 6. Formatting
    ax.set_ylabel('F1 Score', fontsize=14)
    ax.set_title('Impact of Data Cleaning Steps on Downstream Model Utility', fontsize=16, fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(models, fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 1.05) # Leave room for triangles
    
    # 7. Custom Legend
    # Get existing handles (bars)
    handles, labels = ax.get_legend_handles_labels()
    
    # Create a proxy artist for the "Perfect Data" line
    line_handle = Line2D([0], [0], color='green', linewidth=2, linestyle='dashdot')
    handles.append(line_handle)
    labels.append("Perfect Data")
    
    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.0), 
        ncol=len(scenarios_to_plot) // 2 + 2, 
        frameon=True,
        fontsize=14
    )
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = "impact_plot.png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Comprehensive plot saved to {filename}")

def main():
    logger.info("Loading Adult dataset...")
    raw_df = read_adult_df()
    raw_df = clean_placeholder_rows(raw_df, ["?", -1])
    
    X = raw_df.drop('target', axis=1)
    y = raw_df['target']
    
    X_source, X_heldout, y_source, y_heldout = train_test_split(
        X, y, train_size=5000, test_size=15000, stratify=y, random_state=RANDOM_SEED
    )
    
    df_source = pd.concat([X_source, y_source], axis=1)
    df_heldout = pd.concat([X_heldout, y_heldout], axis=1)
    
    corruptor = DataCorruptor(df_source, target_col='target')
    
    all_errors = ['scale', 'shift', 'missing', 'noise', 'label_flip']
    
    # Define Scenarios
    scenarios_config = {
        "Clean (Baseline)": [],
        "Clean Scaling":   [e for e in all_errors if e != 'scale'],
        "Clean Shifting":  [e for e in all_errors if e != 'shift'],
        "Clean Missing":   [e for e in all_errors if e != 'missing'],
        "Clean Noise":     [e for e in all_errors if e != 'noise'],
        "Clean Labels":    [e for e in all_errors if e != 'label_flip'],
        "No cleaning": all_errors,
    }
    
    # Master dictionary to store results: { 'ModelName': { 'Scenario': Score } }
    global_results = {}
    
    logger.info("Starting model evaluation loop...")
    
    for model_opt in SynthcityModelOption:
        logger.info(f"Processing Model: {model_opt.name}")
        model_results = {}
        
        for scenario_name, active_errors in scenarios_config.items():
            
            # 1. Corrupt
            dirty_df = corruptor.get_corrupted_data(active_errors=active_errors)
            
            # 2. Generate
            settings = SynthcitySettings(model_name=model_opt, n_samples=N_GEN_SAMPLES)
            generator = SynthcityGenerator(settings=settings)
            
            try:
                X_dirty = dirty_df.drop('target', axis=1)
                y_dirty = dirty_df['target']
                
                synthetic_data = generator.generate(X_dirty, y_dirty)
                
                # 3. Evaluate
                score = evaluate_utility(synthetic_data, df_heldout)
                model_results[scenario_name] = score
                logger.info(f"  > {scenario_name}: {score:.4f}")
                
            except Exception as e:
                logger.error(f"  > {scenario_name} FAILED: {e}")
                model_results[scenario_name] = 0.0

        global_results[model_opt.name] = model_results

    # 4. Generate Comprehensive Plot
    plot_comprehensive_results(global_results)

if __name__ == '__main__':
    main()