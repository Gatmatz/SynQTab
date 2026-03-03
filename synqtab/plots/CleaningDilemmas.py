import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from synqtab.plots.Plot import Plot
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.enums.data import DataErrorColor

class CleaningDilemmas(Plot):
    def __init__(self, params: dict):
        super().__init__(params)
    
    def read_data(self):
        df_semi = PostgresClient.statistical_importance(evaluation_id=f'{self.params.get("metric", "QLT")}#RH#SH', split='SEMI')
        df_imp = PostgresClient.statistical_importance(evaluation_id=f'{self.params.get("metric", "QLT")}#RH#SH', split='IMP')
        return df_semi, df_imp

    def get_merged_data(self, df_semi, df_imp):
        # Merge to get SEMI and IMP values side-by-side for each data point
        merged_df = pd.merge(
            df_semi, 
            df_imp, 
            on=['generator', 'data_error', 'dataset_id'], 
            suffixes=('_SEMI', '_IMP')
        )
        return merged_df

    def plot_scatter_grid(self, merged_df, filename):
        generators = merged_df['generator'].unique()
        n_gens = len(generators)
        
        # Setup 3x3 grid (or dynamic based on generator count)
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3)
        axes = axes.flatten()

        metric_name = self.params.get("metric", "QLT")

        for i, gen in enumerate(generators):
            if i >= len(axes): break
            
            ax = axes[i]
            gen_data = merged_df[merged_df['generator'] == gen]

            # 1. Define bounds for shading
            min_val = min(gen_data['avg_result_IMP'].min(), gen_data['avg_result_SEMI'].min()) * 0.9
            max_val = max(gen_data['avg_result_IMP'].max(), gen_data['avg_result_SEMI'].max()) * 1.1
            lims = [min_val, max_val]

            # 2. Shading regions (Semi Better vs Imp Better)
            ax.fill_between(lims, lims, max_val, color='lightblue', alpha=0.3, label='SEMI better')
            ax.fill_between(lims, min_val, lims, color='lightcoral', alpha=0.2, label='IMP better')
            
            # 3. Reference Diagonal Line
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=1)

            # 4. Scatter points colored by data_error
            for error_type, group in gen_data.groupby('data_error'):
                color = DataErrorColor[error_type].value
                ax.scatter(
                    group['avg_result_IMP'],
                    group['avg_result_SEMI'],
                    edgecolors='k', color=color, s=50, alpha=0.8, zorder=2,
                    label=error_type
                )

            # Formatting
            ax.set_title(f"{gen}", fontweight='bold')
            ax.set_xlabel(f"{metric_name} (IMP)")
            ax.set_ylabel(f"{metric_name} (SEMI)")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.grid(True, linestyle=':', alpha=0.6)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Single figure-level legend for data error types
        handles, labels = axes[0].get_legend_handles_labels()
        # Keep only scatter handles (exclude shading patches)
        scatter_handles = [(h, l) for h, l in zip(handles, labels) if l not in ('SEMI better', 'IMP better')]
        if scatter_handles:
            sh, sl = zip(*scatter_handles)
            fig.legend(sh, sl, title='Data Error', loc='upper center',
                       ncol=len(sl), fontsize=10, title_fontsize=11,
                       bbox_to_anchor=(0.5, 0.97))

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        df_semi, df_imp = self.read_data()

        # If metric is 'EFF', set negative avg_result values to 0 in both dataframes
        if self.params.get("metric") == "EFF":
            df_semi.loc[df_semi['avg_result'] < 0, 'avg_result'] = 0
            df_imp.loc[df_imp['avg_result'] < 0, 'avg_result'] = 0

        merged_df = self.get_merged_data(df_semi, df_imp)

        output_path = f'result_plots/{self.params.get("metric", "QLT")}_scatter_grid.png'
        self.plot_scatter_grid(merged_df, filename=output_path)

params = {"metric": "EFF"}
CleaningDilemmas(params).run()