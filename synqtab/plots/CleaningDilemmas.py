import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from synqtab.plots.Plot import Plot
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.enums.plots import DataErrorColor, DataErrorMarker, FontSize, PlotFont, MODEL_ORDER

class CleaningDilemmas(Plot):
    def __init__(self, params: dict):
        super().__init__(params)
    
    def read_data(self):
        df_semi = PostgresClient.statistical_importance(evaluation_id=f'{self.params.get("metric", "QLT")}#RH#SH', split='SEMI')
        df_imp = PostgresClient.statistical_importance(evaluation_id=f'{self.params.get("metric", "QLT")}#RH#SH', split='IMP')
        return df_semi, df_imp

    def get_merged_data(self, df_semi, df_imp):
        agg_semi = (
            df_semi
            .groupby(['generator', 'data_error'], as_index=False)['avg_result']
            .mean()
        )
        agg_imp = (
            df_imp
            .groupby(['generator', 'data_error'], as_index=False)['avg_result']
            .mean()
        )
        merged_df = pd.merge(
            agg_semi,
            agg_imp,
            on=['generator', 'data_error'],
            suffixes=('_SEMI', '_IMP')
        )
        return merged_df

    def _add_diagonal_arrows(self, ax):
        """
        Adds stylized arrows pointing to corners with labels positioned 
        at the top of the plot area, similar to the reference image.
        """
        # Top Left arrow
        ax.annotate(
            'Less perfect better',
            xy=(0.02, 0.98),
            xytext=(0.35, 0.85),
            arrowprops=dict(
                arrowstyle='->',
                color='orange',
                lw=2.5,
                shrinkA=0, shrinkB=0
            ),
            color='black',
            fontsize=FontSize.TICK,
            fontweight='bold',
            ha='center', va='center',
            zorder=5
        )

        # Bottom right arrow
        ax.annotate(
            'More imperfect better',
            xy=(0.98, 0.02),
            xytext=(0.65, 0.15),
            arrowprops=dict(
                arrowstyle='->',
                color='steelblue',
                lw=2.5,
                shrinkA=0, shrinkB=0
            ),
            color='black',
            fontsize=FontSize.TICK,
            fontweight='bold',
            ha='center', va='center',
            zorder=5
        )

    def plot_scatter_grid(self, merged_df, filename):
        available = merged_df['generator'].unique()
        generators = [g.value for g in MODEL_ORDER if g.value in available]
        
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        
        # Adjust layout to leave room for the global legend and titles
        plt.subplots_adjust(top=0.83, hspace=0.45, wspace=0.4)
        axes = axes.flatten()

        metric_name = self.params.get("metric", "QLT")

        for i, gen in enumerate(generators):
            if i >= len(axes): break
            
            ax = axes[i]
            gen_data = merged_df[merged_df['generator'] == gen]

            # Set fixed limits 0 to 1
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Reference Diagonal Line
            ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, alpha=0.6, zorder=1)

            # Add arrows and their respective "memos"
            self._add_diagonal_arrows(ax)

            # Scatter points: Hollow circles
            for error_type, group in gen_data.groupby('data_error'):
                color  = DataErrorColor[error_type].value if error_type in DataErrorColor.__members__ else 'gray'
                marker = DataErrorMarker[error_type].value if error_type in DataErrorMarker.__members__ else 'o'
                
                ax.scatter(
                    group['avg_result_IMP'],
                    group['avg_result_SEMI'],
                    edgecolors=color,
                    facecolors='none', 
                    marker=marker,
                    linewidths=1.5,
                    s=65, 
                    alpha=0.8, 
                    zorder=3,
                    label=error_type,
                    clip_on=False
                )

            # Formatting
            ax.set_title(f"{gen}", fontweight='bold', fontsize=FontSize.SUBTITLE, pad=20)
            ax.set_xlabel(f"{metric_name} (IMP)", fontsize=FontSize.X_LABEL)
            ax.set_ylabel(f"{metric_name} (SEMI)", fontsize=FontSize.Y_LABEL)
            
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis='both', labelsize=FontSize.TICK)
            
            # Hide top/right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Unified Legend at the very top
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title='Data Error Type', loc='upper center',
                       ncol=min(len(labels), 4), fontsize=FontSize.LEGEND, 
                       title_fontsize=FontSize.LEGEND, bbox_to_anchor=(0.5, 0.97),
                       frameon=False)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        plt.rcParams['font.family'] = PlotFont.FAMILY.value
        plt.rcParams[f'font.{PlotFont.FAMILY.value}'] = [PlotFont.NAME.value, PlotFont.FALLBACK.value]
        
        df_semi, df_imp = self.read_data()

        # Constraints for 0-1 range
        if self.params.get("metric") == "EFF":
            df_semi.loc[df_semi['avg_result'] < 0, 'avg_result'] = 0
            df_imp.loc[df_imp['avg_result'] < 0, 'avg_result'] = 0
        
        df_semi.loc[df_semi['avg_result'] > 1, 'avg_result'] = 1
        df_imp.loc[df_imp['avg_result'] > 1, 'avg_result'] = 1

        merged_df = self.get_merged_data(df_semi, df_imp)

        output_path = f'result_plots/{self.params.get("metric", "QLT")}_scatter_grid.png'
        self.plot_scatter_grid(merged_df, filename=output_path)

if __name__ == '__main__':
    params = {'metric': 'EFF'}
    CleaningDilemmas(params).run()