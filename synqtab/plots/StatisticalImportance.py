from synqtab.plots.Plot import Plot
from synqtab.data.clients.PostgresClient import PostgresClient
from scipy import stats
import numpy as np
import pandas as pd

class StatisticalImportance(Plot):
    def __init__(self, params:dict):
        super().__init__(params)
    
    def read_data(self):
        # 1. Load the datasets
        df_semi = PostgresClient.statistical_importance(evaluation_id=f'{self.params.get("metric", "QLT")}#RH#SH', split='SEMI')
        df_imp = PostgresClient.statistical_importance(evaluation_id=f'{self.params.get("metric", "QLT")}#RH#SH', split='IMP')
        return df_semi, df_imp

    def perform_statistical_analysis(self, df_semi, df_imp):
        # 1. Load the datasets
        semi_df = df_semi.copy()
        imp_df = df_imp.copy()

        semi_df['split'] = 'SEMI'
        imp_df['split'] = 'IMP'

        merged_df = pd.merge(
            semi_df, 
            imp_df, 
            on=['generator', 'data_error', 'dataset_id'], 
            suffixes=('_SEMI', '_IMP')
        )

        groups = merged_df.groupby(['generator', 'data_error'])
        results = []

        for (gen, error), group in groups:
            semi_vals = group['avg_result_SEMI']
            imp_vals = group['avg_result_IMP']
            diffs = semi_vals - imp_vals
            non_zero_diffs = diffs[diffs != 0]
            
            if len(non_zero_diffs) == 0:
                p_val_t = 1.0
                p_val_w = 1.0
                mean_diff = 0.0
                cohen_d = 0.0
            else:

                if len(group) >= 2:
                    t_stat, p_val_t = stats.ttest_rel(semi_vals, imp_vals)
                else:
                    p_val_t = np.nan
                
                if len(non_zero_diffs) >= 5:
                    w_stat, p_val_w = stats.wilcoxon(semi_vals, imp_vals)
                else:
                    p_val_w = np.nan
                    
                mean_diff = diffs.mean()
                std_diff = diffs.std()
                cohen_d = mean_diff / std_diff if std_diff != 0 else 0

            results.append({
                'generator': gen,
                'data_error': error,
                'p_value_t': p_val_t,
                'p_value_wilcoxon': p_val_w,
                'mean_diff': mean_diff,
                'cohen_d': cohen_d,
                'n_samples': len(group),
                'n_diff': len(non_zero_diffs)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='p_value_t')
        return results_df
    
    def plot_pvalue_heatmap(self, results_df, filename='result_plots/pvalue_heatmap.png'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        pivot = results_df.pivot(index='generator', columns='data_error', values='p_value_t')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'p-value (t-test)'})
        plt.title('Paired t-test p-values by Generator and Data Error')
        plt.ylabel('Generator')
        plt.xlabel('Data Error')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def run(self):
        df_semi, df_imp = self.read_data()
        results_df = self.perform_statistical_analysis(df_semi, df_imp)
        self.plot_pvalue_heatmap(results_df, filename=f'result_plots/{self.params.get("metric", "QLT")}_pvalue_heatmap.png')
    
    

params = {
    "metric": "EFF"
}
StatisticalImportance(params).run()