import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.plots.Plot import Plot
from synqtab.enums.plots import DataErrorColor, DataErrorMarker, FontSize, PlotFont, MODEL_ORDER

class Scatterplot(Plot):
    def __init__(self, params: dict):
        super().__init__(params)
    
    def read_data(self):
        x_axis = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#R#RH')
        y_axis = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#S#SH') 
        return x_axis, y_axis

    def run(self):
        plt.rcParams['font.family'] = PlotFont.FAMILY.value
        plt.rcParams[f'font.{PlotFont.FAMILY.value}'] = [PlotFont.NAME.value, PlotFont.FALLBACK.value]
        x_axis, y_axis = self.read_data()
        merge_cols = ['data_error', 'noise_ratio', 'generator']
        merged = pd.merge(x_axis, y_axis, on=merge_cols, suffixes=('_x', '_y'))

        available = merged['generator'].unique()
        generators = [g.value for g in MODEL_ORDER if g.value in available]
        data_errors = merged['data_error'].unique()
        noise_ratios = [10, 20, 40]
        markers = {10: 'o', 20: 's', 40: '^'}
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 20))
        axes = axes.flatten()

        # Compute global axis limits for uniform scale across all subplots
        global_min = min(merged['avg_result_x'].min(), merged['avg_result_y'].min()) * 0.95
        global_max = max(merged['avg_result_x'].max(), merged['avg_result_y'].max()) * 1.05
        global_lims = [global_min, global_max]

        for idx, generator in enumerate(generators):
            ax = axes[idx]
            gen_df = merged[merged['generator'] == generator]
            
            # 1. Add Diagonal Reference Line (y = x)
            ax.plot(global_lims, global_lims, color='black', linestyle='--', alpha=0.3, zorder=1)
            ax.set_xlim(global_lims)
            ax.set_ylim(global_lims)

            for err in data_errors:
                for nr in noise_ratios:
                    sub = gen_df[(gen_df['data_error'] == err) & (gen_df['noise_ratio'].astype(int) == nr)]
                    if not sub.empty:
                        # Use Enum for color mapping
                        point_color = DataErrorColor[err].value if err in DataErrorColor.__members__ else '#808080'
                        marker = DataErrorMarker[err].value if err in DataErrorMarker.__members__ else markers[nr]
                        
                        ax.scatter(
                            sub['avg_result_x'],
                            sub['avg_result_y'],
                            color=point_color,
                            marker=marker,
                            s=250,  # Bigger marker size
                            edgecolor='white',
                            linewidth=1.2,
                            alpha=0.85,
                            zorder=3
                        )
            
            ax.set_title(generator, fontsize=FontSize.SUBTITLE, fontweight='bold', pad=10)
            ax.set_xlabel(f"{self.params.get('metric', 'QLT')} RRhat", fontsize=FontSize.X_LABEL)
            ax.set_ylabel(f"{self.params.get('metric', 'QLT')} SShat", fontsize=FontSize.Y_LABEL)
            ax.tick_params(axis='both', labelsize=FontSize.TICK)
            ax.grid(True, linestyle=':', alpha=0.4)

        # Cleanup unused axes
        for j in range(len(generators), 9):
            fig.delaxes(axes[j])

        # 2. Create Explicit Legends (Proxy Artists)
        # Error Type Legend (Colors)
        error_handles = [
            Line2D([0], [0], marker=DataErrorMarker[err].value if err in DataErrorMarker.__members__ else 'o',
                   color='w', markerfacecolor=DataErrorColor[err].value,
                   markersize=12, label=err)
            for err in data_errors if err in DataErrorColor.__members__
        ]
        
        # Noise Ratio Legend (Shapes)
        noise_handles = [
            Line2D([0], [0], marker=markers[nr], color='w', 
                   markerfacecolor='gray', markeredgecolor='k', 
                   markersize=12, label=f"{nr}% Noise") 
            for nr in noise_ratios
        ]

        # Add legends to the bottom of the figure: error types on top row, noise below
        first_legend = fig.legend(handles=error_handles, title="Error Types",
                                  loc='lower center', bbox_to_anchor=(0.5, 0.055),
                                  ncol=len(error_handles), frameon=True,
                                  fontsize=FontSize.LEGEND, title_fontsize=FontSize.LEGEND)

        fig.add_artist(first_legend)
        fig.legend(handles=noise_handles, title="Noise Intensity",
                   loc='lower center', bbox_to_anchor=(0.5, 0.01),
                   ncol=len(noise_handles), frameon=True,
                   fontsize=FontSize.LEGEND, title_fontsize=FontSize.LEGEND)

        plt.tight_layout(rect=[0, 0.13, 1, 1])
        plt.savefig(f"result_plots/{self.params.get('title', 'default')}.png", bbox_inches='tight')

if __name__ == '__main__':
    params = {'title': "QLT_scatterplot_grid", 'metric': 'QLT'}
    Scatterplot(params).run()