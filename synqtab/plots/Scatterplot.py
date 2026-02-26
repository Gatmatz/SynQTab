import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.plots.Plot import Plot

class Scatterplot(Plot):
    def __init__(self, params:dict):
        super().__init__(params)
    
    def read_data(self):
        # 1. Load the datasets
        x_axis = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#R#RH')
        # todo: replace the different baselines for each error type to a single baseline.
        y_axis = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#S#SH')  # Baseline
        return x_axis, y_axis

    def run(self):
        x_axis, y_axis = self.read_data()
        merge_cols = ['data_error', 'noise_ratio', 'generator']
        merged = pd.merge(x_axis, y_axis, on=merge_cols, suffixes=('_x', '_y'))

        generators = merged['generator'].unique()
        data_errors = merged['data_error'].unique()
        noise_ratios = [10, 20, 40]
        markers = {10: 'o', 20: 's', 40: '^'}
        colors = plt.cm.get_cmap('tab10', len(data_errors))
        color_map = {err: colors(i) for i, err in enumerate(data_errors)}

        fig, axes = plt.subplots(3, 3, figsize=(18, 20)) 
        axes = axes.flatten()

        for idx, generator in enumerate(generators):
            ax = axes[idx]
            gen_df = merged[merged['generator'] == generator]
            
            for err in data_errors:
                for nr in noise_ratios:
                    sub = gen_df[(gen_df['data_error'] == err) & (gen_df['noise_ratio'].astype(int) == nr)]
                    if not sub.empty:
                        ax.scatter(
                            sub['avg_result_x'],
                            sub['avg_result_y'],
                            label=f"{err}, {nr}%", # Label every point so fig.legend can find them
                            color=color_map[err],
                            marker=markers[nr],
                            s=100,
                            edgecolor='k',
                            alpha=0.8
                        )
            
            ax.set_title(generator)
            ax.set_xlabel(f"{self.params.get('metric', 'QLT')} RRhat")
            ax.set_ylabel(f"{self.params.get('metric', 'QLT')} SShat")

        for j in range(len(generators), 9):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        
        fig.legend(
            handles, 
            labels, 
            loc='lower center', 
            ncol=len(data_errors),
            bbox_to_anchor=(0.5, 0.02),
            fontsize=10
        )

        # 4. Adjust layout to ensure legend doesn't overlap plots
        plt.tight_layout(rect=[0, 0.05, 1, 1]) 
        plt.savefig(f"result_plots/{self.params.get('metric', 'default')}_scatterplot_grid.png", bbox_inches='tight')


params = {
    'title': "QLT_scatterplot_grid",
    'metric': 'QLT'
}

Scatterplot(params).run()