import pandas as pd
import matplotlib.pyplot as plt
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.plots.Plot import Plot
from synqtab.enums.data import DataErrorColor

class Lineplot(Plot):
    def __init__(self, params: dict):
        super().__init__(params)
    
    def read_data(self):
        df_lines = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#RH#SH')
        df_baseline = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#R#S')
        return df_lines, df_baseline

    def run(self):
        df_lines, df_baseline = self.read_data()
        metric = self.params.get("metric", "QLT")
        
        # 1. Metric Specific Data Cleaning
        if metric == "EFF":
            df_lines.loc[df_lines['avg_result'] < 0, 'avg_result'] = 0
            df_baseline.loc[df_baseline['avg_result'] < 0, 'avg_result'] = 0

        error_types = sorted(df_lines['data_error'].unique())
        generators = sorted(df_lines['generator'].unique())
        noise_ratios = [10, 20, 40]

        fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharex=True)
        axes = axes.flatten()
        handles_labels = {}

        for idx, generator in enumerate(generators[:9]):
            ax = axes[idx]
            y_all = []

            # 2. Add Gray Reference Line at 0.5 for APR/ARC
            if metric in ["APR", "ARC"]:
                ref_line = ax.axhline(0.5, color='#d3d3d3', linestyle='-', linewidth=1.5, zorder=1)
                if "Random Guess (0.5)" not in handles_labels:
                    handles_labels["Random Guess (0.5)"] = ref_line

            # 3. Plot Error Lines (Colors from Enum - No Red/No Green)
            for error in error_types:
                color = DataErrorColor[error].value if error in DataErrorColor.__members__ else '#7f7f7f'
                
                subset = df_lines[(df_lines['generator'] == generator) & (df_lines['data_error'] == error)]
                if not subset.empty:
                    subset = subset.sort_values('noise_ratio')
                    x_vals = [int(r) for r in subset['noise_ratio']]
                    y_vals = subset['avg_result'].tolist()
                    y_all.extend(y_vals)
                    
                    line, = ax.plot(x_vals, y_vals, marker='o', label=error, color=color, linewidth=2, markersize=7, zorder=3)
                    if error not in handles_labels: 
                        handles_labels[error] = line

            # 4. Plot Baseline (Always Green)
            baseline_subset = df_baseline[df_baseline['generator'] == generator]
            if not baseline_subset.empty:
                median_val = baseline_subset['avg_result'].median()
                y_base = [median_val] * len(noise_ratios)
                y_all.extend(y_base)
                
                line_base, = ax.plot(noise_ratios, y_base, marker='x', linestyle='--', 
                                     color='#2ca02c', label='Baseline', linewidth=2.5, markersize=10, zorder=4)
                if 'Baseline' not in handles_labels:
                    handles_labels['Baseline'] = line_base
            
            self._apply_axis_styling(ax, generator, metric, y_all, noise_ratios)

        # 5. Global Legend
        fig.legend(list(handles_labels.values()), list(handles_labels.keys()), 
                   loc='upper center', ncol=5, fontsize='large', bbox_to_anchor=(0.5, 1.05), frameon=False)
        
        plt.tight_layout()
        save_path = f"result_plots/{self.params.get('title', 'default')}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=200)

    def _apply_axis_styling(self, ax, generator, metric, y_all, noise_ratios):
        ax.set_title(generator, fontweight='bold', pad=10)
        ax.set_xticks(noise_ratios)
        ax.grid(True, linestyle=':', alpha=0.4, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if y_all:
            y_min, y_max = min(y_all), max(y_all)
            # Ensure 0.5 is visible in APR/ARC plots
            if metric in ["APR", "ARC"]:
                y_min, y_max = min(y_min, 0.45), max(y_max, 0.55)
                
            padding = 0.15 * (y_max - y_min) if y_max != y_min else 0.1
            ax.set_ylim(y_min - padding, y_max + padding)

# Example Usage for APR
params = {'title': "3. ARC_lineplot_grid", 'metric': 'ARC'}
Lineplot(params).run()