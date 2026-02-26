import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from synqtab.data.clients.PostgresClient import PostgresClient
from synqtab.plots.Plot import Plot

class Barplot(Plot):
    def __init__(self, params:dict):
        super().__init__(params)
    
    def read_data(self):
        # 1. Load the datasets
        df_one = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#R#RH')
        df_two = PostgresClient.lineplot_query(f'{self.params.get("metric", "QLT")}#S#SH')
        return df_one, df_two

    def run(self):
        df_one, df_two = self.read_data()
        merged = pd.merge(
            df_one,
            df_two,
            on=["data_error", "noise_ratio", "generator"],
            suffixes=("_one", "_two")
        )
        merged["avg_result_diff"] = merged["avg_result_two"] - merged["avg_result_one"]

        generators = sorted(merged["generator"].unique())
        data_errors = sorted(merged["data_error"].unique())
        noise_ratios_int = sorted([int(float(nr)) for nr in merged["noise_ratio"].unique()])

        fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharey=False)
        
        fig.suptitle(f"{self.params.get('metric', 'Barplot')} S-SH minus R-RH", fontsize=18, y=0.97)

        legend_handles = []

        for idx, generator in enumerate(generators[:9]):
            row, col = divmod(idx, 3)
            ax = axes[row, col]
            gen_df = merged[merged["generator"] == generator]
            
            if not gen_df.empty:
                max_val = gen_df["avg_result_diff"].abs().max()
                limit = max_val * 1.15 if max_val > 0 else 1.0
                ax.set_ylim(-limit, limit)

            x = np.arange(len(noise_ratios_int))
            width = 0.8 / len(data_errors) if len(data_errors) > 0 else 0.8
            
            for i, data_error in enumerate(data_errors):
                y_vals = []
                for nr in noise_ratios_int:
                    row_df = gen_df[(gen_df["data_error"] == data_error) & 
                                    (gen_df["noise_ratio"].astype(float) == nr)]
                    y_vals.append(row_df["avg_result_diff"].values[0] if not row_df.empty else 0)
                
                bar = ax.bar(x + i * width, y_vals, width, label=data_error)
                
                if idx == 0:
                    legend_handles.append(bar)

            group_center_offset = (width * (len(data_errors) - 1)) / 2
            for xc in [0.5, 1.5]:
                ax.axvline(x=xc + group_center_offset, color='gray', linestyle=':', linewidth=1, alpha=0.4)

            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title(generator, fontsize=12, pad=8)
            
            ax.set_xticks(x + group_center_offset)
            ax.set_xticklabels([str(nr) for nr in noise_ratios_int], fontsize=9)
            
            ax.tick_params(axis='y', labelsize=9, labelleft=True)
            if col == 0:
                ax.set_ylabel("Diff", fontsize=10)

        fig.legend(legend_handles, data_errors, title="Data Error", loc='upper center', 
                   bbox_to_anchor=(0.5, 0.93), ncol=len(data_errors), 
                   fontsize=11, frameon=False)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.90])
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        plt.savefig(f"result_plots/{self.params.get('title', 'default')}.png", bbox_inches='tight', dpi=200)

params = {
    'title': "QLT_barplot_grid",
    'metric': 'QLT'
}

Barplot(params).run()