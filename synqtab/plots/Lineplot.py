import pandas as pd
import matplotlib.pyplot as plt
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.plots.Plot import Plot
from synqtab.enums.data import DataErrorColor

class Lineplot(Plot):
    def __init__(self, params: dict):
        super().__init__(params)
    
    def read_data(self):
        if self.params.get("metric") == "MLAUG":
            df_baseline_precision = PostgresClient.lineplot_query(f'{self.params.get("APR", "APR")}#R#S')
            df_baseline_recall = PostgresClient.lineplot_query(f'{self.params.get("ARC", "ARC")}#R#S')

            df_lines_precision = PostgresClient.lineplot_query(f'{self.params.get("APR", "APR")}#RH#SH')
            df_lines_recall = PostgresClient.lineplot_query(f'{self.params.get("ARC", "ARC")}#RH#SH')
            
            # Concatenate and average precision and recall per group
            df_lines = (
                pd.concat([df_lines_precision, df_lines_recall])
                .groupby(['generator', 'data_error', 'noise_ratio'], as_index=False)['avg_result']
                .mean()
            )
            df_lines['avg_result'] = df_lines['avg_result'].round(4)

            df_baseline = (
                pd.concat([df_baseline_precision, df_baseline_recall])
                .groupby(['generator', 'data_error', 'noise_ratio'], as_index=False)['avg_result']
                .mean()
            )
            df_baseline['avg_result'] = df_baseline['avg_result'].round(4)
            return df_lines, df_baseline

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
        y_global = []

        # First pass: collect all y values to determine global y range
        for generator in generators[:9]:
            for error in error_types:
                subset = df_lines[(df_lines['generator'] == generator) & (df_lines['data_error'] == error)]
                if not subset.empty:
                    y_global.extend(subset['avg_result'].tolist())
            baseline_subset = df_baseline[df_baseline['generator'] == generator]
            if not baseline_subset.empty:
                y_global.append(baseline_subset['avg_result'].median())

        for idx, generator in enumerate(generators[:9]):
            ax = axes[idx]

            # 2. Add Gray Reference Line at 0.5 for APR/ARC
            if metric in ["MLAUG"]:
                ref_line = ax.axhline(0.5, color='#d3d3d3', linestyle='-', linewidth=1.5, zorder=1)
                if "Baseline (0.5)" not in handles_labels:
                    handles_labels["Baseline (0.5)"] = ref_line

            # 3. Plot Error Lines
            for error in error_types:
                color = DataErrorColor[error].value if error in DataErrorColor.__members__ else '#7f7f7f'
                
                subset = df_lines[(df_lines['generator'] == generator) & (df_lines['data_error'] == error)]
                if not subset.empty:
                    subset = subset.sort_values('noise_ratio')
                    x_vals = [int(r) for r in subset['noise_ratio']]
                    y_vals = subset['avg_result'].tolist()
                    
                    line, = ax.plot(x_vals, y_vals, marker='o', label=error, color=color, linewidth=2, markersize=7, zorder=3)
                    if error not in handles_labels: 
                        handles_labels[error] = line

            # 4. Plot Baseline (Green)
            baseline_subset = df_baseline[df_baseline['generator'] == generator]
            if not baseline_subset.empty:
                median_val = baseline_subset['avg_result'].median()
                y_base = [median_val] * len(noise_ratios)
                
                line_base, = ax.plot(noise_ratios, y_base, marker='', linestyle='--', 
                                     color='#2ca02c', label='Baseline', linewidth=2.5, zorder=4)
                if 'Baseline' not in handles_labels:
                    handles_labels['Clean Baseline'] = line_base
            
            self._apply_axis_styling(ax, generator, metric, y_global, noise_ratios)

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
            if metric in ["MLAUG"]:
                y_min, y_max = min(y_min, 0.45), max(y_max, 0.55)
                
            padding = 0.15 * (y_max - y_min) if y_max != y_min else 0.1
            ax.set_ylim(y_min - padding, y_max + padding)

params = {'title': "2. MLAUG_lineplot_grid", 'metric': 'MLAUG'}
Lineplot(params).run()