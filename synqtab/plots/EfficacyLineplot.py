import pandas as pd
import matplotlib.pyplot as plt
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.plots.Plot import Plot
from synqtab.enums.plots import DataErrorColor, DataErrorMarker, FontSize, PlotFont, MODEL_ORDER

class EfficacyLineplot(Plot):
    def __init__(self, params: dict):
        super().__init__(params)
    
    def read_data(self):
        df_lines = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#RH#SH')
        df_baseline = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#R#S')
        return df_lines, df_baseline

    def run(self):
        plt.rcParams['font.family'] = PlotFont.FAMILY.value
        plt.rcParams[f'font.{PlotFont.FAMILY.value}'] = [PlotFont.NAME.value, PlotFont.FALLBACK.value]
        df_lines, df_baseline = self.read_data()
        metric = self.params.get("metric", "QLT")
        df_lines.loc[df_lines['avg_result'] < 0, 'avg_result'] = 0
        df_baseline.loc[df_baseline['avg_result'] < 0, 'avg_result'] = 0
        
        error_types = sorted(df_lines['data_error'].unique())
        available = df_lines['generator'].unique()
        generators = [g.value for g in MODEL_ORDER if g.value in available]
        noise_ratios = [10, 20, 40]

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
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

            for error in error_types:
                color = DataErrorColor[error].value if error in DataErrorColor.__members__ else '#7f7f7f'
                
                subset = df_lines[(df_lines['generator'] == generator) & (df_lines['data_error'] == error)]
                if not subset.empty:
                    subset = subset.sort_values('noise_ratio')
                    x_vals = [int(r) for r in subset['noise_ratio']]
                    y_vals = subset['avg_result'].tolist()
                    
                    marker = DataErrorMarker[error].value if error in DataErrorMarker.__members__ else 'o'
                    line, = ax.plot(x_vals, y_vals, marker=marker, label=error, color=color, linewidth=2,
                                    markersize=12, markerfacecolor='white', markeredgecolor=color,
                                    markeredgewidth=2.5, zorder=3)
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
                   loc='upper center', ncol=5, fontsize=FontSize.LEGEND, bbox_to_anchor=(0.5, 1.05), frameon=False)
        
        plt.tight_layout(h_pad=4.0)
        save_path = f"result_plots/{self.params.get('title', 'default')}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=200)

    def _apply_axis_styling(self, ax, generator, metric, y_all, noise_ratios):
        ax.set_title(generator, fontweight='bold', pad=10, fontsize=FontSize.SUBTITLE)
        ax.set_xlabel('Noise Ratio (%)', fontsize=FontSize.X_LABEL, labelpad=12)
        ax.set_ylabel(metric, fontsize=FontSize.Y_LABEL)
        ax.set_xticks(noise_ratios)
        ax.tick_params(axis='both', labelsize=FontSize.TICK)
        ax.grid(True, linestyle=':', alpha=0.4, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if y_all:
            y_min, y_max = min(y_all), max(y_all)
                
            padding = 0.15 * (y_max - y_min) if y_max != y_min else 0.1
            ax.set_ylim(y_min - padding, y_max + padding)

if __name__ == '__main__':
    params = {'title': "1. EFF_lineplot_grid", 'metric': 'EFF'}
    EfficacyLineplot(params).run()