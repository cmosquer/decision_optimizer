from decision_optimizer.Calibrator import *
from decision_optimizer.Evaluator import Evaluator
from decision_optimizer.TestSetting import TestSetting
from decision_optimizer.evaluation_utils import plot_costs_vs_theta
from scipy.special import logit, expit
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    CALCULATE_METRICS = True
    AGGREGATE_EXISTING_CSV = True
    ground_truth_column = "GT"
    training_csvs ="/data/Runs_Validation/"
    tests_csvs = "/data/Runs_Test/"
    run_csvs = sorted(os.listdir(training_csvs))
    output_dir = f"/data/CALIBRATION/"
    os.makedirs(f'/{output_dir}/Runs_Test_with_calibration/', exist_ok=True)
    os.makedirs(f'/{output_dir}/Runs_Validation_with_calibration/', exist_ok=True)

    # Configure scorers
    evaluated_score_columns = ['Malignancy']
    evaluated_subgroups = ['all', 'light', 'dark']

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
            joined_df_train = pd.read_csv(training_csvs + runcsv)
            joined_df_train.columns = ['GT', 'Malignancy', 'subset']
            joined_df_train.GT = joined_df_train.GT.astype(int)

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
            all_train_results = pd.DataFrame()
            for k_score, evaluated_score_column in enumerate(evaluated_score_columns):
                for k_scenario, current_scenario in enumerate(predefined_scenarios):
                    for k_subgroup, subgroup in enumerate(evaluated_subgroups):
                        print(subgroup.upper())
                        calibrators_dict = {
                            'LOG-REG': LogisticRegressionWCECalibrator()
                        }

                        train_test_output_dir = f'/{output_dir}/validation_plots/{runcsv.replace(".csv", "/")}'
                        os.makedirs(train_test_output_dir, exist_ok=True)
                        train_test_setting = TestSetting(joined_df_train,
                                                         evaluated_score_column,
                                                         ground_truth_column,
                                                         train_test_output_dir)
                        train_test_setting.set_current_expected_scenario(current_scenario)
                        train_test_setting.initial_evaluation()

                        if subgroup == 'all':
                            joined_df_test_subgroup = joined_df_test
                        else:
                            joined_df_test_subgroup = joined_df_test[joined_df_test.subset == subgroup]
                        print('Subgroup samples in test ', len(joined_df_test_subgroup))
                        os.makedirs(f'/{output_dir}/test_plots/{subgroup}/', exist_ok=True)
                        test_output_dir = f'/{output_dir}/test_plots/{subgroup}/{runcsv.replace(".csv", "/")}'
                        os.makedirs(test_output_dir, exist_ok=True)
                        test_setting = TestSetting(joined_df_test_subgroup,
                                                   evaluated_score_column,
                                                   ground_truth_column,
                                                   test_output_dir)
                        test_setting.set_current_expected_scenario(current_scenario)
                        test_setting.initial_evaluation()


                        # Fit calibrators with train subset
                        for calibrator_name, calibrator in calibrators_dict.items():
                            # Fit calibrators. For logReg, use effective_positive_priors to compute LLRs and to fit algorithm
                            calibrator.train(scores=joined_df_train[evaluated_score_column].values,
                                             labels=joined_df_train[ground_truth_column].values,
                                             effective_positive_prior=0.5)

                            # Evaluate effect on the same training set
                            train_test_setting.set_current_expected_scenario(current_scenario)
                            train_test_setting.calibration_evaluation(calibrator_name, calibrator)

                            # Evaluate effect on the test set
                            test_setting.set_current_expected_scenario(current_scenario)
                            test_setting.calibration_evaluation(calibrator_name, calibrator, get_posteriors_thresholds=True)

                        if subgroup == 'all':
                            test_csv_dir = f'/{output_dir}/Runs_Test_with_calibration/{runcsv}'
                            test_setting.df.to_csv(test_csv_dir)
                            train_test_setting.results['N'] = [len(joined_df_train)] * len(train_test_setting.results)
                            train_test_setting.results['N_positive'] = [len(
                                joined_df_train[joined_df_train[ground_truth_column] == 1])] * len(
                                train_test_setting.results)
                            train_csv_dir = f'/{output_dir}/Runs_Validation_with_calibration/{runcsv}'
                            train_test_setting.df.to_csv(train_csv_dir)
                            all_train_results = all_train_results.append(train_test_setting.results, ignore_index=True)

                            parameters[run].update(test_setting.calibrators_parameters)
                        test_setting.results['subset'] = [subgroup]*len(test_setting.results)
                        test_setting.results['N'] = [len(joined_df_test_subgroup)]*len(test_setting.results)
                        test_setting.results['N_positive'] = [len(joined_df_test_subgroup[joined_df_test_subgroup[ground_truth_column] == 1])]*len(test_setting.results)
                        all_test_results = all_test_results.append(test_setting.results, ignore_index=True)

            all_test_results['run'] = [runcsv.replace('.csv', '')]*len(all_test_results)
            if os.path.exists(f'/{output_dir}/test_metrics.csv'):
                existing_results = pd.read_csv(f'/{output_dir}/test_metrics.csv')
                existing_results = existing_results.append(all_test_results)
                existing_results.to_csv(f'/{output_dir}/test_metrics.csv', index=False)
            else:
                all_test_results.to_csv(f'/{output_dir}/test_metrics.csv', index=False)

            all_train_results['run'] = [runcsv.replace('.csv', '')]*len(all_train_results)
            if os.path.exists(f'/{output_dir}/validation_metrics.csv'):
                existing_results_train = pd.read_csv(f'/{output_dir}/validation_metrics.csv')
                existing_results_train = existing_results_train.append(all_train_results)
                existing_results_train.to_csv(f'/{output_dir}/validation_metrics.csv', index=False)
            else:
                all_train_results.to_csv(f'/{output_dir}/validation_metrics.csv', index=False)

        try:
            with open(f'/{output_dir}/parameters_results.json', "w") as fp:
                json.dump(parameters, fp)
        except Exception as e:
            print('Error saving parameter json', e)

        #Calculate deltas with perfect calibrated version
        all_test_results = pd.read_csv(f'/{output_dir}/test_metrics.csv')
        for run in all_test_results.run.unique():
            print(run)
            for subgroup in all_test_results.subset.unique():
                perfect_run = all_test_results[(all_test_results.run == run) &
                                               (all_test_results.calibrator == 'perfect_PAV') &
                                               (all_test_results.subset == subgroup)]

                for metric in ['CE', 'Balanced_CE', 'Brier', 'Balanced_Brier', 'predefined_cost']:
                    perfect_metric = perfect_run[metric].values[0]
                    for method in ['no_calibration', 'LOG-REG']:
                        method_metric = all_test_results[(all_test_results.run == run) &
                                                         (all_test_results.calibrator == method) &
                                                         (all_test_results.subset == subgroup)][metric].values[0]
                        all_test_results.at[(all_test_results['run'] == run) &
                                            (all_test_results.calibrator == method) &
                                            (all_test_results.subset == subgroup),
                                            'delta_' + metric] = method_metric - perfect_metric
        all_test_results.to_csv(f'/{output_dir}/test_metrics.csv', index=False)

    if AGGREGATE_EXISTING_CSV:
        #Sort columns
        for set in ['validation', 'test']:
            all_results = pd.read_csv(f'/{output_dir}/{set}_metrics.csv')
            metrics = ['AUCROC', 'AUCPR',
                         'adjusted_AUCPR', 'EER',
                         'ECE_Naeini', 'MCE_Naeini', 'ACE_Naeini', 'ECE_Guo', 'MCE_Guo', 'ACE_Guo',
                         'Brier', 'delta_Brier', 'Balanced_Brier', 'delta_Balanced_Brier',
                         'CE', 'delta_CE',  'Balanced_CE', 'delta_Balanced_CE',
                         'predefined_precision', 'predefined_recall', 'predefined_specificity',
                         'predefined_FPR', 'predefined_FNR',
                        'predefined_cost', 'discrim_cost', 'delta_predefined_cost'
                         ]
            all_results = all_results[['run', 'subset', 'calibrator', 'N',
                                        'N_positive', 'test_real_prior'] + metrics]
            all_results.columns = [m.replace('predefined_', '') for m in all_results.columns]
            all_results.to_csv(f'/{output_dir}/{set}_metrics.csv', index=False)
            all_results.to_excel(f'/{output_dir}/{set}_metrics.xls', index=False)

        agg_dict = dict(zip(metrics, [['mean', 'std', 'median', 'min', 'max']] * len(metrics)))
        aggregated_results = pd.DataFrame()
        all_test_results = pd.read_csv(f'/{output_dir}/test_metrics.csv')
        for method in ['no_calibration', 'LOG-REG']:
            for subgroup in evaluated_subgroups:
                subdf = all_test_results[
                    (all_test_results.calibrator == method) & (all_test_results.subset == subgroup)]
                assert len(subdf) == len(run_csvs)
                agg_subdf = subdf.agg(agg_dict)
                agg_subdf['method'] = [method] * len(agg_subdf)
                agg_subdf['subset'] = [subgroup] * len(agg_subdf)
                agg_subdf['statistic'] = agg_subdf.index
                agg_subdf = agg_subdf[['method', 'subset', 'statistic'] + metrics]
                agg_subdf.columns = [m.replace('predefined_', '') for m in agg_subdf.columns]
                aggregated_results = aggregated_results.append(agg_subdf, ignore_index=True)
        aggregated_results.to_csv(f'/{output_dir}/aggregated_test_metrics.csv', index=False)

        plot_metrics = ['AUCROC', 'AUCPR', 'adjusted_AUCPR',
                   'ECE_Naeini', 'MCE_Naeini', 'ACE_Naeini',
                   'ECE_Guo', 'MCE_Guo', 'ACE_Guo',
                   'precision', 'recall', 'specificity',
                   'cost', 'discrim_cost', 'delta_cost',
                   'CE', 'delta_CE', 'Brier', 'delta_Brier']
        fig, axs = plt.subplots(7, 3, figsize=(12, 24))
        fig.tight_layout()

        for j, metric in enumerate(plot_metrics):
            ax = axs[j // 3][j % 3]
            sns.boxplot(x="calibrator", y=metric,
                        hue="subset", palette=["m", "g", "b"],
                        data=test_metrics[test_metrics.calibrator.isin(['no_calibration', 'LOG-REG'])],
                        ax=ax)
            ax.set_title(metric)
            ax.set_xlabel('')
            ax.set_ylabel('')
            sns.despine(offset=10, trim=True)
            if j > 0:
                ax.get_legend().remove()
            if j < len(plot_metrics)-1:
                ax.set_xticklabels([])
        fig.savefig(f'{output_dir}/metrics_plot.png', dpi=500)

if __name__ == '__main__':
    main()
