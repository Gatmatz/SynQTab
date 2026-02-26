import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from synqtab.plots.Plot import Plot
from synqtab.data.clients.PostgresClient import PostgresClient

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
        print(merged_df.head())  # Debug: Check merged data structure
        exit(0)
        return merged_df

    def plot_scatter_grid(self, merged_df, filename):
        generators = merged_df['generator'].unique()
        n_gens = len(generators)
        
        # Setup 3x3 grid (or dynamic based on generator count)
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
        axes = axes.flatten()

        metric_name = self.params.get("metric", "QLT")

        for i, gen in enumerate(generators):
            if i >= len(axes): break
            
            ax = axes[i]
            gen_data = merged_df[merged_df['generator'] == gen]
            
            x = gen_data['avg_result_IMP']
            y = gen_data['avg_result_SEMI']

            # 1. Define bounds for shading
            min_val = min(x.min(), y.min()) * 0.9
            max_val = max(x.max(), y.max()) * 1.1
            lims = [min_val, max_val]

            # 2. Shading regions (Semi Better vs Imp Better)
            ax.fill_between(lims, lims, max_val, color='lightblue', alpha=0.3, label='SEMI better')
            ax.fill_between(lims, min_val, lims, color='lightcoral', alpha=0.2, label='IMP better')
            
            # 3. Reference Diagonal Line
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=1)

            # 4. Scatter points
            ax.scatter(x, y, edgecolors='k', color='orange', s=50, alpha=0.8, zorder=2)

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

        plt.suptitle(f'SEMI vs IMP Performance Comparison ({metric_name})', fontsize=16)
        plt.savefig(filename, dpi=300)
        plt.close()

    def run(self):
        df_semi, df_imp = self.read_data()
        merged_df = self.get_merged_data(df_semi, df_imp)
        
        output_path = f'result_plots/{self.params.get("metric", "QLT")}_scatter_grid.png'
        self.plot_scatter_grid(merged_df, filename=output_path)

params = {"metric": "EFF"}
CleaningDilemmas(params).run()