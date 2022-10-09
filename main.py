from decision_optimizer.Segmentator import TorchSegmentator, KerasSegmentator
from decision_optimizer.Scorer import *
from decision_optimizer.Calibrator import *
from decision_optimizer.Splitter import Splitter
from decision_optimizer.Evaluator import Evaluator
from decision_optimizer.TestSetting import TestSetting
from decision_optimizer.evaluation_utils import plot_costs_vs_theta
from scipy.special import logit, expit
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    debug = False
    output_dir = f"/data/decision_optimizer_results/"

    cxr_dir = "/data/images/"
    pneumothorax_mask_dir = "/data/pneumothorax_masks/"
    binarize_mask_th = 0.005
    ground_truth_csv = "/data/csv_procesados/Pneumothorax_without_reports_with_gt.csv"
    ground_truth_column = "GT"
    output_scores_csv = f"/data/decision_optimizer_results/scores_and_gt.csv"
    # Configure scorers
    evaluated_score_columns = ['lung_normalized_score',
                               'range_normalized_score',
                               ]

    lung_dir = "/data/lung_masks_hybridgnet/"
    lung_mask_type = "binary_image"   #or pickle

    # --- Configure test settings & expected scenarios ----#
    real_positive_priors = [0.5, 0.25]
    scenarios_cost_false_positive = [1]
    scenarios_cost_false_negative = [1]
    scenarios_expected_positive_prior = [0.5]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("/data/decision_optimizer_results/", exist_ok=True)
    if debug:
        output_scores_csv = output_scores_csv.replace('.csv','_debug.csv')
        pneumothorax_mask_dir = pneumothorax_mask_dir[:-1] + '_debug/'
        ground_truth_csv = ground_truth_csv.replace('.csv', '_debug.csv')

    assert os.path.exists(ground_truth_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv).drop_duplicates('SOPInstanceUID').set_index('SOPInstanceUID')
    if not os.path.exists(output_scores_csv) or debug:
        output_scores_df = ground_truth_df.copy()
    else:
        output_scores_df = pd.read_csv(output_scores_csv).set_index('SOPInstanceUID')
    parameters = {}

    if bool(int(os.environ['CREATE_PNEUMOTHORAX_MASKS'])):
        model_config = {'CHECKPOINT': '/pneumothorax_model/NT-RX-tr-20191028-00_fold0.pth',
                        'PY': 'Albunet',
                        'CLASS': 'AlbuNet',
                        'ARGS': {'pretrained': False},
                        'ALBU_TRANSFORM': '/pneumothorax_model/pneumothorax_transforms.json'}
        if bool(int(os.environ['CREATE_JPGS'])):
            jpgs_dir = cxr_dir + "jpgs/"
        else:
            jpgs_dir = None
        pneumothorax_inferer = TorchSegmentator(model_config_dict=model_config, save_jpg_dir=jpgs_dir)
        pneumothorax_inferer.load_model()
        pneumothorax_inferer.generate_all_masks(cxr_dir=cxr_dir, output_masks_dir=pneumothorax_mask_dir)
            
    if bool(int(os.environ['CREATE_LUNG_MASKS_UNET'])):
        model_config = {'CHECKPOINT': '/lung_model/Left-Lung-Weights.h5,/lung_model/Right-Lung-Weights.h5',
                        'JSON': '/lung_model/model_config.json',
                        'DIM': 128}
        lung_inferer = KerasSegmentator(model_config_dict=model_config)
        lung_inferer.load_model()
        lung_inferer.generate_all_masks(cxr_dir=cxr_dir, output_masks_dir=lung_dir)

    #---------------------Check lung scorer--------------------------------------#
    if 'lung_normalized_score' in evaluated_score_columns:
        if 'lung_normalized_score' not in output_scores_df.columns:
            lung_scorer = LungMaskScorer(
                                         pneumothorax_masks_dir=pneumothorax_mask_dir,
                                         imgs_dir=cxr_dir+"jpgs",
                                         binarize_mask_th=binarize_mask_th,
                                         lung_masks_dir=lung_dir,
                                         lung_mask_type=lung_mask_type,
                                        )

            if bool(int(os.environ['CREATE_MASKS_PLOTS'])):
                plot_dir = output_dir+f"masks_on_image/th{binarize_mask_th}/"
                if debug:
                    plot_dir = plot_dir+'debug/'
                os.makedirs(plot_dir, exist_ok=True)
            else:
                plot_dir = None
            csv_path = f"{output_dir}/scores_hybridgnet.csv"
            if debug:
                csv_path = csv_path.replace('.csv','_debug.csv')
            lung_scorer.obtain_scores(csv_path, plot_dir=plot_dir,
                                      #calculate_dice_df=output_scores_df
                                      )
            temp_df = pd.read_csv(csv_path).set_index('SOPInstanceUID')
            output_scores_df = output_scores_df.join(temp_df, rsuffix='_lung')
            output_scores_df.to_csv(output_scores_csv)
            os.remove(csv_path)
        parameters['lung_scorer'] = {
                                          'max_whole_dataset_ptx_area': str(output_scores_df['pneumothorax_area'].max()),
                                          'min_whole_dataset_ptx_area': str(output_scores_df['pneumothorax_area'].min()),
                                          'max_whole_dataset_lung_area': str(output_scores_df['lung_area'].max()),
                                          'min_whole_dataset_lung_area': str(output_scores_df['lung_area'].min()),
                                          'max_whole_dataset_normalized': output_scores_df['lung_normalized_score'].max(),
                                          'min_whole_dataset_normalized': output_scores_df['lung_normalized_score'].min(),
                                          }

    # ---------------------Get training set ----------------------------------#
    print("Splitting original dataset of {} samples".format(len(ground_truth_df)))

    splitter = Splitter(split_level='patient',
                        stratify_criteria=ground_truth_column,
                        random_state=42)
    splitter.split_df(ground_truth_df, train_size=0.5)
    ground_truth_train = ground_truth_df.loc[splitter.current_trainset_filenames]#.reset_index(drop=False)

    print("\n--------------\nTrain dataset of {} samples\n--------------\n".format(len(ground_truth_train)))

    parameters.update({
                        'binarize_mask_th': binarize_mask_th,
                        'train_set':
                            {
                                'N_images': len(ground_truth_train),
                                'N_images_pnemuthorax': len(ground_truth_train[ground_truth_train[ground_truth_column]==1])
                            }
                      })

    #---------------------Check range scorer--------------------------------------#
    parameters['range_scorer'] = {}
    if 'range_normalized_score' in evaluated_score_columns:
        if 'range_normalized_score' not in output_scores_df.columns:
            range_scorer = NormalizationScorer(baseline_scores_file=output_scores_csv,
                                                pneumothorax_masks_dir=pneumothorax_mask_dir,
                                                imgs_dir=cxr_dir+"jpgs",
                                                baseline_score_name='pneumothorax_area',
                                                binarize_mask_th=binarize_mask_th
                                               )
            range_scorer.obtain_range(train_sops=ground_truth_train.index.values)
            csv_path = f"{output_dir}/scores_range_normalization.csv"
            if debug:
                csv_path = csv_path.replace('.csv', '_debug.csv')
            range_scorer.obtain_scores(csv_path,
                                      calculate_dice_df=output_scores_df
                                       )
            temp_df = pd.read_csv(csv_path).set_index('SOPInstanceUID')
            output_scores_df = output_scores_df.join(temp_df, rsuffix='_range')
            output_scores_df.to_csv(output_scores_csv)
            os.remove(csv_path)
            parameters['range_scorer'].update(
                                                {'max_train': str(range_scorer.range_max),
                                               'min_train': str(range_scorer.range_min)})
        parameters['range_scorer'] = {
                                      'max_whole_dataset_normalized': output_scores_df['range_normalized_score'].max(),
                                      'min_whole_dataset_normalized': output_scores_df['range_normalized_score'].min(),
                                      }
    with open(f'/{output_dir}/parameters_results.json', "w") as fp:
        json.dump(parameters, fp)

    joined_df_train = output_scores_df[output_scores_df.index.isin(ground_truth_train.index)]#SOPInstanceUID].reset_index(drop=False)
    print('len df train', len(joined_df_train), 'columns ',joined_df_train.columns)
    # -----------Prepare predefined expected scenarios----------------------#
    predefined_scenarios = []
    for cost_false_positive, cost_false_negative, expected_positive_prior in zip(scenarios_cost_false_positive,
                                                                                 scenarios_cost_false_negative,
                                                                                 scenarios_expected_positive_prior):
        predefined_theta = np.log((cost_false_positive/cost_false_negative)*((1-expected_positive_prior)/expected_positive_prior))
        effective_positive_prior = expit(-predefined_theta)

        predefined_scenarios.append({
                                    'expected_positive_prior': expected_positive_prior,
                                    'cost_false_positive': cost_false_positive,
                                    'cost_false_negative': cost_false_negative,
                                    'effective_positive_prior': effective_positive_prior,
                                    'predefined_theta': predefined_theta
                                    })

    #---------------Evaluate multiple test settings & expected scenarios ---------------------------#
    all_test_results = pd.DataFrame()
    all_train_results = pd.DataFrame()
    for k_score, evaluated_score_column in enumerate(evaluated_score_columns):
        os.makedirs(f'/{output_dir}/{evaluated_score_column}/', exist_ok=True)
        calibrators_dict = {
                    'LOG-REG': LogisticRegressionWCECalibrator()
                            }

        train_test_output_dir = f'/{output_dir}/{evaluated_score_column}/train/'
        os.makedirs(train_test_output_dir, exist_ok=True)
        train_test_setting = TestSetting(joined_df_train,
                                         evaluated_score_column,
                                         ground_truth_column,
                                         train_test_output_dir)
        train_test_setting.initial_evaluation()

        for k_test, test_setting_prior in enumerate(real_positive_priors):
            #Create test setting with this positive prior
            #LLR_histograms_fig, LLR_histograms_axs = plt.subplots(len(calibrators_dict) + 2)#, sharex=True)
            os.makedirs(f'/{output_dir}/{evaluated_score_column}/test_prior{test_setting_prior}/', exist_ok=True)
            ground_truth_test = splitter.get_test_subset(
                pos_prior=test_setting_prior, final_size=1620*2)  # Mantener constante el N total de test
            joined_df_test = output_scores_df.loc[ground_truth_test.index]#.reset_index(drop=False)

            test_output_dir = f'/{output_dir}/{evaluated_score_column}/test_prior{test_setting_prior}/'
            os.makedirs(test_output_dir, exist_ok=True)
            test_setting = TestSetting(joined_df_test,
                                       evaluated_score_column,
                                       ground_truth_column,
                                       test_output_dir)
            test_setting_prior = test_setting.setting_prior
            #print("\n--------------\nTest scenario with {} samples and "
            #      "positive prior of {}\n--------------\n".format(len(ground_truth_test), test_setting_prior))
            test_setting.initial_evaluation()
            for k_scenario, current_scenario in enumerate(predefined_scenarios):

                # Fit calibrators with train subset
                for calibrator_name, calibrator in calibrators_dict.items():
                    #Fit calibrators. For logReg, use effective_positive_priors to compute LLRs and to fit algorithm
                    calibrator.train(scores=joined_df_train[evaluated_score_column].values,
                                     labels=joined_df_train[ground_truth_column].values,
                                     effective_positive_prior=current_scenario['effective_positive_prior'])

                    # Evaluate effect on the same training set
                    train_test_setting.set_current_expected_scenario(current_scenario)
                    train_test_setting.calibration_evaluation(calibrator_name, calibrator)

                    # Evaluate effect on the test set
                    test_setting.set_current_expected_scenario(current_scenario)
                    test_setting.calibration_evaluation(calibrator_name, calibrator, get_posteriors_thresholds=True)

            test_csv_dir = f'/{test_output_dir}/test.csv'
            test_setting.df.to_csv(test_csv_dir)
            all_test_results = all_test_results.append(test_setting.results, ignore_index=True)
            all_train_results = all_train_results.append(train_test_setting.results, ignore_index=True)
            parameters.update(test_setting.calibrators_parameters)
        train_test_setting.df.to_csv(f'/{train_test_output_dir}/train.csv')

    try:
        with open(f'/{output_dir}/parameters_results.json', "w") as fp:
            json.dump(parameters, fp)
    except Exception as e:
        print('Error saving parameter json', e)

    all_test_results.to_csv(f'/{output_dir}/test_settings_results.csv', index=False)
    all_train_results.to_csv(f'/{output_dir}/train_settings_results.csv', index=False)
    #output_scores_df['SOPInstanceUID'] = output_scores_df.index
    output_scores_df.to_csv(output_scores_csv)

if __name__ == '__main__':
    main()
