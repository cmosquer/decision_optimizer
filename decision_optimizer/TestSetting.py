import pandas as pd
import numpy as np
import os
from decision_optimizer.Plotter import Plotter
from decision_optimizer.Evaluator import Evaluator
from decision_optimizer.Calibrator import *

class TestSetting():
    def __init__(self, df, evaluated_score_column, ground_truth_column, output_dir):
        self.labels = df[ground_truth_column].values.astype(int)
        self.scores = df[evaluated_score_column].values
        self.df = df
        self.plotter = Plotter()
        self.calibrators_parameters = {}
        self.setting_prior = np.sum(self.labels) / len(self.labels)
        self.results = pd.DataFrame()
        self.output_dir = output_dir
        self.params = {'test_real_prior': self.setting_prior,
                       'scorer': evaluated_score_column
                       }
        self.scenario_params = {}
        self.df['no_calibration'] = self.scores

    def fit_perfect_PAV(self):
        # Fit perfect PAV to test (independent of predefined expected parameters)
        perfect_test_calibrator = PAVCalibrator()
        perfect_test_calibrator.train(scores=self.scores,
                                      labels=self.labels)
        self.df['perfect_PAV'] = perfect_test_calibrator.trainset_calibrated_scores
        self.calibrators_parameters['perfect_PAV'] = str(perfect_test_calibrator.parameters)

    def initial_evaluation(self):
        self.fit_perfect_PAV()
        print("\nEvaluating metrics for perfect PAV (PAV fitted on test set) as reference...")
        self.evaluate_pipeline(pipeline_name='perfect_PAV',
                               color='blue')
        self.plotter.save_fig('/{}/perfect_PAV.png'.format(self.output_dir))

        print("\nEvaluating metrics for uncalibrated scores...")
        self.evaluate_pipeline(pipeline_name='no_calibration',
                               color='green', restart_plotter=True)
        self.plotter.save_fig('/{}/no_calibration.png'.format(self.output_dir))

    def calibration_evaluation(self, calibrator_name, calibrator,
                               color='orange', get_posteriors_thresholds=False):

        # Apply calibrator to this test subset
        calibrated = calibrator.apply_calibration(self.scores)
        self.df[calibrator_name] = calibrated

        print(f"\nEvaluating metrics for {calibrator_name} calibrated scores...")
        self.calibrators_parameters[calibrator_name] = str(calibrator.parameters)
        if get_posteriors_thresholds:
            self.get_posteriors_thresholds(calibrator)
        self.evaluate_pipeline(calibrator_name, color, restart_plotter=True)
        self.plotter.save_fig('/{}/effective{:.2f}/{}.png'.format(self.output_dir,
                                                            self.scenario_params['effective_positive_prior'],
                                                            calibrator_name))

    def set_current_expected_scenario(self, current_scenario):
        self.scenario_params = current_scenario.copy()
        self.scenario_params.update(self.params)
        os.makedirs('/{}/effective{:.2f}/'.format(self.output_dir,
                                                  self.scenario_params['effective_positive_prior']), exist_ok=True)

    def evaluate_pipeline(self, pipeline_name, color, restart_plotter=False):
        evaluator = Evaluator(self.labels,
                              self.df[pipeline_name].values,
                              positive_prior=self.setting_prior)
        if restart_plotter:
            self.plotter = Plotter()
        metrics = evaluator.apply_evaluator(evaluation_name=pipeline_name,
                                  color=color,
                                  plotter=self.plotter
                                  )
        if len(self.scenario_params) > 0:
            metrics = evaluator.apply_threshold(self.scenario_params['predefined_theta'],
                                                threshold_name='predefined',
                                                plotter=self.plotter
                                                )
            metrics.update(self.scenario_params)
        metrics.update({'calibrator': pipeline_name})
        metrics.update({'setting_prior': self.setting_prior})
        self.results = self.results.append(metrics, ignore_index=True)



    def get_posteriors_thresholds(self, calibrator):
        # Log value of thresholds for posteriors before and after calibration
        calibrated_posteriors_threshold = expit(self.scenario_params['predefined_theta']
                                                + logit(calibrator.effective_positive_prior))
        if hasattr(calibrator, 'apply_inverse'):
            raw_posteriors_threshold = calibrator.apply_inverse(self.scenario_params['predefined_theta'],
                                                                is_LLR=True)
        else:
            raw_posteriors_threshold = None

        print("The threshold of LLR={} on calibrated scores corresponds to a score of {} on calibrated"
              "posteriors and {} on uncalibrated posteriors".format(self.scenario_params['predefined_theta'],
                                                                    calibrated_posteriors_threshold,
                                                                    raw_posteriors_threshold))
        self.scenario_params.update({'raw_posteriors_threshold': raw_posteriors_threshold,
                                     'calibrated_posteriors_threshold': calibrated_posteriors_threshold})