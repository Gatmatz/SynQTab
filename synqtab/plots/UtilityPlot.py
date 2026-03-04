import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.plots.Plot import Plot
from synqtab.enums.plots import DataErrorColor


class UtilityPlot(Plot):
    def __init__(self, params):
        super().__init__(params)

    def read_data(self):
        df_bars = PostgresClient.util_query(
            evaluation_id=self.params.get("evaluation_id", "QLT"),
            split=self.params.get("split", "IMP")
        )

        df_baseline = PostgresClient.util_query(
            evaluation_id=self.params.get("evaluation_id", "QLT").replace("#RH#SH", "#R#S"),
            split=self.params.get("split", "IMP")
        )
        return df_bars, df_baseline
    
    def run(self):
        df_bars, df_baseline = self.read_data()

        generators = sorted(df_bars['generator'].unique())
        data_errors = sorted(df_bars['data_error'].unique())
        noise_ratios = sorted(df_bars['noise_ratio'].unique(), key=lambda x: int(float(x)))

        # First pass: compute global y range
        all_y = list(df_bars['total_experiments'].astype(int))
        if not df_baseline.empty:
            all_y += list(df_baseline['total_experiments'].astype(int))
        y_min = min(all_y)
        y_max = max(all_y)
        padding = 0.1 * (y_max - y_min) if y_max != y_min else 1
        y_lim = (max(0, y_min - padding), y_max + padding)

        # Precompute global fallback baseline in case a generator has no baseline data
        global_baseline_val = df_baseline['total_experiments'].astype(int).sum() if not df_baseline.empty else None

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

        legend_handles = []

        for idx, generator in enumerate(generators[:9]):
            ax = axes[idx]
            gen_df = df_bars[df_bars['generator'] == generator]

            x = np.arange(len(noise_ratios))
            width = 0.8 / len(data_errors) if data_errors else 0.8
            offsets = np.linspace(-(len(data_errors) - 1) / 2, (len(data_errors) - 1) / 2, len(data_errors)) * width

            for i, data_error in enumerate(data_errors):
                color = DataErrorColor[data_error].value if data_error in DataErrorColor.__members__ else '#7f7f7f'
                y_vals = []
                for nr in noise_ratios:
                    row_df = gen_df[
                        (gen_df['data_error'] == data_error) &
                        (gen_df['noise_ratio'].astype(str) == str(nr))
                    ]
                    y_vals.append(int(row_df['total_experiments'].values[0]) if not row_df.empty else 0)

                bars = ax.bar(x + offsets[i], y_vals, width, label=data_error, color=color)
                if idx == 0:
                    legend_handles.append(bars)

            # Baseline green horizontal line
            gen_baseline = df_baseline[df_baseline['generator'] == generator]
            baseline_val = (
                gen_baseline['total_experiments'].astype(int).sum()
                if not gen_baseline.empty
                else global_baseline_val
            )
            if baseline_val is not None:
                baseline_line = ax.axhline(
                    baseline_val, color='#2ca02c', linestyle='--',
                    linewidth=2.5, zorder=4, label='Baseline'
                )
                if idx == 0:
                    legend_handles.append(baseline_line)

            ax.set_title(generator, fontweight='bold', pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels([str(nr) for nr in noise_ratios], fontsize=9)
            ax.set_xlabel("Noise Ratio", fontsize=10)
            ax.set_ylabel("Total Experiments", fontsize=10)
            ax.set_ylim(y_lim)
            ax.grid(True, linestyle=':', alpha=0.4, zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for idx in range(len(generators), 9):
            axes[idx].set_visible(False)

        fig.suptitle(self.params.get('title', 'UtilPlot'), fontsize=18, y=1.01)
        legend_labels = data_errors + (['Baseline'] if len(generators) > 0 else [])
        fig.legend(
            legend_handles, legend_labels,
            title="Data Error", loc='upper center',
            bbox_to_anchor=(0.5, 0.97), ncol=len(legend_labels),
            fontsize=11, frameon=False
        )

        plt.tight_layout()
        save_path = f"result_plots/{self.params.get('title', 'default')}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=200)


if __name__ == '__main__':
    params = {
        'title': "Utility Plot",
        'evaluation_id': 'EFF#RH#SH',
        'split': 'IMP'
    }
    UtilityPlot(params).run()