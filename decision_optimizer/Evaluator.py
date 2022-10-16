import numpy as np
import pandas as pd
from sklearn.metrics import *
from scipy.special import logit, expit

from decision_optimizer.Calibrator import *
from decision_optimizer.MetricsCalculator import MetricsCalculator

class Evaluator():

    def __init__(self, test_labels, test_posteriors,
                 positive_prior=0.5, cost_FP=1, cost_FN=1):
        self.positive_posteriors = test_posteriors
        self.correct_posteriors()

        self.labels = test_labels.astype(int)
        self.LLR = logit(self.positive_posteriors) - logit(positive_prior)

        self.real_positive_prior = np.sum(np.squeeze(self.labels)) / len(self.labels)

        self.set_error_rates()
        self.set_detection_cost_function(positive_prior, cost_FP, cost_FN)

        self.metrics = {}
        self.metrics_calculator = MetricsCalculator(self)

    def correct_posteriors(self, epsilon=1e-15):
        # For stability, posteriors cannot be exactly 1 or exactly 0
        posteriors_1 = np.argwhere(self.positive_posteriors == 1.)[:, 0]
        if len(posteriors_1) > 0:
            #print("corrected {} posteriors that where exactly 1 with epsilon {}".format(len(posteriors_1), epsilon))
            self.positive_posteriors[posteriors_1] = self.positive_posteriors[posteriors_1] - epsilon

        posteriors_0 = np.argwhere(self.positive_posteriors == 0.)[:, 0]
        if len(posteriors_0) > 0:
            #print("corrected {} posteriors that where exactly 0 with epsilon {}".format(len(posteriors_0), epsilon))
            self.positive_posteriors[posteriors_0] = self.positive_posteriors[posteriors_0] + epsilon

    def set_error_rates(self):
        fpr, tpr, threshold = roc_curve(self.labels, self.LLR)
        self.tpr = tpr
        self.fpr = fpr
        self.thresholds_ROC = threshold

        precision, recall, threshold = precision_recall_curve(self.labels, self.LLR)
        self.recalls = recall
        self.precisions = precision
        self.thresholds_PR = threshold

    def set_detection_cost_function(self, expected_test_prior, cost_false_negative, cost_false_positive):
        """
        detection_cost_function = CMiss x PTarget x PMiss|Target + CFalseAlarm x (1-PTarget) x PFalseAlarm|NonTarget
        """
        self.detection_cost_function = cost_false_negative * (1 - self.tpr) * expected_test_prior + \
                                       cost_false_positive * self.fpr * (1 - expected_test_prior)

    def get_unthresholded_metrics(self):
        brier, CE = self.metrics_calculator.get_proper_scores()
        balanced_brier, balanced_CE = self.metrics_calculator.get_proper_scores(balanced=True)
        ECE_Naeini, MCE_Naeini, _, _, _ = self.metrics_calculator.get_calibration_errors_Naeini()
        ECE_Guo, MCE_Guo, _, _, _ = self.metrics_calculator.get_calibration_errors_Guo()

        ACE_Naeini = self.metrics_calculator.get_adaptive_calibration_error_Naeini()
        ACE_Guo = self.metrics_calculator.get_adaptive_calibration_error_Guo()
        metrics_dict = {'AUCROC': self.metrics_calculator.get_AUCROC(),
                        'AUCPR': self.metrics_calculator.get_AUCPR(),
                        'adjusted_AUCPR': self.metrics_calculator.get_adjusted_AUCPR(),
                        'EER': self.metrics_calculator.get_EER()[0],
                        'ECE_Naeini': ECE_Naeini,
                        'MCE_Naeini': MCE_Naeini,
                        'ACE_Naeini': ACE_Naeini,
                        'ECE_Guo': ECE_Guo,
                        'MCE_Guo': MCE_Guo,
                        'ACE_Guo': ACE_Guo,
                        'Brier': brier,
                        'Balanced_Brier': balanced_brier,
                        'CE': CE,
                        'Balanced_CE': balanced_CE,
                        'discrim_cost': self.metrics_calculator.get_DCFmin()[0]
                        }
        self.metrics.update(metrics_dict)
        return metrics_dict

    def set_threshold_scenario(self, current_theta, threshold_name=""):
        """
        :param current_theta: LLR threshold
        :param threshold_name:
        :return:
        """
        self.current_threshold_name = threshold_name
        self.current_theta = current_theta

        self.current_theta_idx_ROC = 0
        for j, th in enumerate(self.thresholds_ROC):
            if th <= current_theta:
                self.current_theta_idx_ROC = j - 1
                break

        # Verify the available thresholds means the same binary predictions than the requested LLR_threshold:
        self.bin_preds_current_theta = np.array(self.LLR > self.current_theta,
                                                        dtype=np.int)
        bin_preds_closest_ROC_th = np.array(self.LLR > self.thresholds_ROC[self.current_theta_idx_ROC],
                                                             dtype=np.int)
        sum = np.sum(np.squeeze(self.bin_preds_current_theta) - np.squeeze(bin_preds_closest_ROC_th))

        if sum != 0.0:
            print("{} binary predictions were different with LLR={} threshold and the closest available "
                  "threshold from ROC curve array, which is LLR={}".format(int(abs(sum)),
                                                                           current_theta,
                                                                           self.thresholds_ROC[self.current_theta_idx_ROC]))
        self.current_theta_idx_PR = 0
        for j, th in enumerate(self.thresholds_PR):
            if th >= current_theta:
                self.current_theta_idx_PR = j - 1
                break
        bin_preds_closest_PR_th = np.array(self.LLR > self.thresholds_PR[self.current_theta_idx_PR], dtype=np.int)
        sum = np.sum(np.squeeze(self.bin_preds_current_theta) - np.squeeze(bin_preds_closest_PR_th))
        if sum != 0.0:
            print("{} binary predictions were different with LLR={} threshold and the closest available "
                  "threshold from PR curve array, which is LLR={}".format(int(abs(sum)),
                                                                          current_theta,
                                                                          self.thresholds_PR[self.current_theta_idx_PR]))

    def get_threshold_metrics(self):
        # Get binary precision and recall to calculate binary AUCPR
        current_cost = self.detection_cost_function[self.current_theta_idx_ROC]
        metrics_dict = {
                        f'{self.current_threshold_name}_cost': current_cost,
                        f'{self.current_threshold_name}_precision': self.precisions[self.current_theta_idx_PR],
                        f'{self.current_threshold_name}_recall': self.recalls[self.current_theta_idx_PR],
                        f'{self.current_threshold_name}_TPR': self.tpr[self.current_theta_idx_ROC],
                        f'{self.current_threshold_name}_specificity': 1 - self.fpr[self.current_theta_idx_ROC],
                        f'{self.current_threshold_name}_FPR': self.fpr[self.current_theta_idx_ROC], #Pfa
                        f'{self.current_threshold_name}_FNR': 1-self.tpr[self.current_theta_idx_ROC], #Pmiss
                        f'{self.current_threshold_name}_calib_cost': current_cost - self.metrics['discrim_cost'],
                        f'{self.current_threshold_name}_balanced_acc': self.metrics_calculator.get_accuracy(self.bin_preds_current_theta,
                                                                                                            balanced=True)
}

        self.metrics.update(metrics_dict)
        return metrics_dict

    def apply_evaluator(self, evaluation_name, color, plotter):
        self.get_unthresholded_metrics()
        plotter.add_evaluator(self, color, evaluation_name)
        return self.metrics

    def apply_threshold(self, current_theta, threshold_name, plotter):
        self.set_threshold_scenario(current_theta, threshold_name)
        self.get_threshold_metrics()
        plotter.add_threshold_point()
        return self.metrics


