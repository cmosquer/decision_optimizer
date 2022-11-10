from decision_optimizer.Calibrator import *
from decision_optimizer.Evaluator import Evaluator
from decision_optimizer.TestSetting import TestSetting
from decision_optimizer.evaluation_utils import plot_costs_vs_theta
from scipy.special import logit, expit
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    CALCULATE_METRICS = True
    AGGREGATE_EXISTING_CSV = True
    ground_truth_column = "GT"
    tests_csvs = "/data/Runs_Test/"
    run_csvs = sorted(os.listdir(tests_csvs))
    output_dir = f"/data/RESAMPLE_ALL/"

    # Configure scorers
    evaluated_score_column = 'Malignancy'

    sampling_proportions = [25, 50, 75, 100]



    if CALCULATE_METRICS:
        # --- Configure test settings & expected scenarios ----#
        scenarios_cost_false_positive = [1]
        scenarios_cost_false_negative = [1]
        scenarios_expected_positive_prior = [0.5]

        os.makedirs(output_dir, exist_ok=True)
        parameters = {}
        for runcsv in run_csvs:
            run = runcsv.replace('.csv','')
            parameters[run] = {}
            print(runcsv)
            joined_df_test = pd.read_csv(tests_csvs + runcsv)
            joined_df_test.columns = ['id', 'GT', 'Malignancy', 'subset']
            joined_df_test.GT = joined_df_test.GT.astype(int)

            # -----------Prepare predefined expected scenarios----------------------#
            predefined_scenarios = []
            for cost_false_positive, cost_false_negative, expected_positive_prior in zip(scenarios_cost_false_positive,
                                                                                         scenarios_cost_false_negative,
                                                                                         scenarios_expected_positive_prior):
                predefined_theta = np.log(
                    (cost_false_positive / cost_false_negative) * ((1 - expected_positive_prior) / expected_positive_prior))
                effective_positive_prior = expit(-predefined_theta)

                predefined_scenarios.append({
                    'expected_positive_prior': expected_positive_prior,
                    'cost_false_positive': cost_false_positive,
                    'cost_false_negative': cost_false_negative,
                    'effective_positive_prior': effective_positive_prior,
                    'predefined_theta': predefined_theta
                })

            # ---------------Evaluate multiple test settings & expected scenarios ---------------------------#
            all_test_results = pd.DataFrame()
            k_scenario = 0
            current_scenario = predefined_scenarios[0]
            for proportion in sampling_proportions:

                joined_df_test_subgroup = joined_df_test.sample(frac=proportion/100)

                print('Subgroup samples in test ', len(joined_df_test_subgroup))
                os.makedirs(f'/{output_dir}/test_plots/{proportion}/', exist_ok=True)
                test_output_dir = f'/{output_dir}/test_plots/{proportion}/{runcsv.replace(".csv", "/")}'
                os.makedirs(test_output_dir, exist_ok=True)
                test_setting = TestSetting(joined_df_test_subgroup,
                                           evaluated_score_column,
                                           ground_truth_column,
                                           test_output_dir)
                test_setting.set_current_expected_scenario(current_scenario)
                test_setting.initial_evaluation()
                test_setting.results['sampling_proportion'] = [proportion]*len(test_setting.results)

                test_setting.results['N'] = [len(joined_df_test_subgroup)]*len(test_setting.results)
                test_setting.results['N_positive'] = [len(joined_df_test_subgroup[joined_df_test_subgroup[ground_truth_column] == 1])]*len(test_setting.results)
                test_setting.results['N_dark'] = [len(
                    joined_df_test_subgroup[joined_df_test_subgroup['subset'] == 'dark'])] * len(
                    test_setting.results)
                test_setting.results['N_light'] = [len(
                    joined_df_test_subgroup[joined_df_test_subgroup['subset'] == 'light'])] * len(
                    test_setting.results)
                all_test_results = all_test_results.append(test_setting.results, ignore_index=True)

            all_test_results['run'] = [runcsv.replace('.csv', '')]*len(all_test_results)
            if os.path.exists(f'/{output_dir}/test_metrics.csv'):
                existing_results = pd.read_csv(f'/{output_dir}/test_metrics.csv')
                existing_results = existing_results.append(all_test_results)
                existing_results.to_csv(f'/{output_dir}/test_metrics.csv', index=False)
            else:
                all_test_results.to_csv(f'/{output_dir}/test_metrics.csv', index=False)


    if AGGREGATE_EXISTING_CSV:
        if not os.path.exists(f'/{output_dir}/raw_test_metrics.csv'):
            #Calculate deltas with perfect calibrated version
            all_test_results = pd.read_csv(f'/{output_dir}/test_metrics.csv')
            for run in all_test_results.run.unique():
                print(run)
                for proportion in all_test_results.sampling_proportion.unique():
                    perfect_run = all_test_results[(all_test_results.run == run) &
                                                   (all_test_results.calibrator == 'perfect_PAV') &
                                                   (all_test_results.sampling_proportion == proportion)]

                    for metric in ['CE', 'Balanced_CE', 'Brier', 'Balanced_Brier', 'predefined_cost']:
                        if metric not in all_test_results.columns:
                            metric = metric.replace('predefined_','')
                        perfect_metric = perfect_run[metric].values[0]
                        for method in ['no_calibration']:
                            method_metric = all_test_results[(all_test_results.run == run) &
                                                             (all_test_results.calibrator == method) &
                                                             (all_test_results.sampling_proportion == proportion)][metric].values[0]
                            all_test_results.at[(all_test_results['run'] == run) &
                                                (all_test_results.calibrator == method) &
                                                (all_test_results.sampling_proportion == proportion),
                                                'delta_' + metric] = method_metric - perfect_metric
            all_test_results.to_csv(f'/{output_dir}/raw_test_metrics.csv', index=False)

        #Sort columns
        set = 'test'
        all_results = pd.read_csv(f'/{output_dir}/raw_test_metrics.csv')
        print(all_results.columns)
        metrics = [m for m in ['AUCROC', 'AUCPR',
                     'adjusted_AUCPR', 'EER',
                     'ECE_Naeini', 'MCE_Naeini', 'ACE_Naeini', 'ECE_Guo', 'MCE_Guo', 'ACE_Guo',
                     'Brier', 'delta_Brier', 'Balanced_Brier', 'delta_Balanced_Brier',
                     'CE', 'delta_CE',  'Balanced_CE', 'delta_Balanced_CE',
                     'predefined_precision', 'predefined_recall', 'predefined_specificity',
                     'predefined_FPR', 'predefined_FNR', 'predefined_balanced_acc',
                    'predefined_cost', 'discrim_cost', 'delta_predefined_cost'
                     ] if m in all_results.columns]
        print(metrics)
        info = [c for c in ['run', 'calibrator', 'N', 'sampling_proportion',
                                    'N_positive', 'N_dark' , 'N_light', 'test_real_prior'] if c in all_results.columns]
        all_results = all_results[info + metrics]
        all_results.columns = [m.replace('predefined_', '') for m in all_results.columns]
        all_results.to_csv(f'/{output_dir}/{set}_metrics.csv', index=False)
        all_results.to_excel(f'/{output_dir}/{set}_metrics.xls', index=False)

        metrics = [m.replace('predefined_','') for m in metrics]
        agg_dict = dict(zip(metrics, [['mean', 'std', 'median', 'min', 'max']] * len(metrics)))
        aggregated_results = pd.DataFrame()
        all_test_results = pd.read_csv(f'/{output_dir}/test_metrics.csv')
        method = 'no_calibration'
        for proportion in sampling_proportions:
            subdf = all_test_results[
                (all_test_results.calibrator == method) & (all_test_results.sampling_proportion == proportion)]
            assert len(subdf) == len(run_csvs)
            agg_subdf = subdf.agg(agg_dict)
            agg_subdf['method'] = [method] * len(agg_subdf)
            agg_subdf['sampling_proportion'] = [proportion] * len(agg_subdf)
            agg_subdf['statistic'] = agg_subdf.index
            agg_subdf = agg_subdf[['method', 'sampling_proportion', 'statistic'] + metrics]
            agg_subdf.columns = [m.replace('predefined_', '') for m in agg_subdf.columns]
            aggregated_results = aggregated_results.append(agg_subdf, ignore_index=True)
        aggregated_results.to_csv(f'/{output_dir}/aggregated_test_metrics.csv', index=False)
        aggregated_results.to_excel(f'/{output_dir}/aggregated_test_metrics.xls', index=False)
        plot_metrics = [m for m in ['AUCROC', 'AUCPR', 'adjusted_AUCPR',
                   'ECE_Naeini', 'MCE_Naeini', 'ACE_Naeini',
                   'ECE_Guo', 'MCE_Guo', 'ACE_Guo',
                   'precision', 'recall', 'specificity',
                   'cost', 'discrim_cost', 'delta_cost',
                   'CE', 'delta_CE', 'balanced_acc',
                   'Brier', 'delta_Brier'] if m in all_test_results.columns]
        fig, axs = plt.subplots(7, 3, figsize=(12, 24))
        fig.tight_layout()

        for j, metric in enumerate(plot_metrics):
            ax = axs[j // 3][j % 3]
            sns.boxplot(x="sampling_proportion", y=metric,
                        palette="Blues",
                        data=all_test_results[all_test_results.calibrator.isin(['no_calibration'])],
                        ax=ax)
            ax.set_title(metric)
            ax.set_xlabel('')
            ax.set_ylabel('')
            sns.despine(offset=10, trim=True)

            if j < len(plot_metrics)-1:
                ax.set_xticklabels([])
        fig.savefig(f'{output_dir}/metrics_plot.png', dpi=500)

if __name__ == '__main__':
    main()
