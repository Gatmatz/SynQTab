import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.plots.Plot import Plot

class Lineplot(Plot):
    def __init__(self, params:dict):
        super().__init__(params)
    
    def read_data(self):
        # 1. Load the datasets
        df_lines = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#RH#SH')
        df_baseline = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#R#S')  # Baseline
        return df_lines, df_baseline

    def run(self):
        df_lines, df_baseline = self.read_data()

        error_types = sorted(df_lines['data_error'].unique())
        generators = sorted(df_lines['generator'].unique())
        noise_ratios = [10, 20, 40]

        fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharex=True, sharey=False)
        axes = axes.flatten()

        color_map = dict(zip(error_types, plt.cm.tab10.colors[:len(error_types)]))
        handles_labels = {}

        for idx, generator in enumerate(generators[:9]):
            ax = axes[idx]
            y_all = []
            
            for error in error_types:
                subset = df_lines[(df_lines['generator'] == generator) & (df_lines['data_error'] == error)]
                color = color_map[error]
                lw = 1.5
                
                if not subset.empty:
                    x_vals = [int(r) for r in subset['noise_ratio']]
                    y_vals = subset['avg_result'].tolist()
                    y_all.extend(y_vals)
                    line, = ax.plot(x_vals, y_vals, marker='o', label=error, color=color, linewidth=lw)
                    if error not in handles_labels: handles_labels[error] = line
                
                baseline_subset = df_baseline[(df_baseline['generator'] == generator) & (df_baseline['data_error'] == error)]
                if not baseline_subset.empty:
                    y_base_val = baseline_subset['avg_result'].iloc[0]
                    y_base = [y_base_val] * 3
                    y_all.extend(y_base)
                    
                    line_base, = ax.plot(noise_ratios, y_base, marker='x', linestyle=':', color=color, label=f'{error} baseline', linewidth=lw)
                    label_base = f'{error} baseline'
                    if label_base not in handles_labels: handles_labels[label_base] = line_base

            if y_all:
                y_min, y_max = min(y_all), max(y_all)
                if y_min == y_max:
                    ax.set_ylim(y_min - 0.1, y_max + 0.1)
                else:
                    padding = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - padding, y_max + padding)
            else:
                ax.set_ylim(0, 1)

            ax.set_title(generator)
            ax.set_xlabel('Noise Ratio')
            ax.set_ylabel(f'{self.params.get("metric", "QLT")} Score')
            ax.set_xticks(noise_ratios)
            ax.set_xticklabels([str(n) for n in noise_ratios])
            ax.grid(True, linestyle='--', alpha=0.7)

        # Legend and layout
        fig.legend(list(handles_labels.values()), list(handles_labels.keys()), loc='upper center', ncol=4, fontsize='large', bbox_to_anchor=(0.5, 1.03))
        plt.tight_layout(rect=(0,0,1,0.95))
        plt.savefig(f"result_plots/{self.params.get('title', 'default')}.png", bbox_inches='tight')


params = {
    'title': "QLT_lineplot_grid",
    'metric': 'QLT'
}

Lineplot(params).run()