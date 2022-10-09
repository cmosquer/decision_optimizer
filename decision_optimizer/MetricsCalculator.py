from sklearn.metrics import *
from sklearn.metrics import auc as sklearnAUC
from math import log
import numpy as np

class MetricsCalculator():
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def get_proper_scores(self, stratify=True):
        brier = brier_score_loss(self.evaluator.labels, self.evaluator.posteriors)
        CE = log_loss(self.evaluator.labels, self.evaluator.posteriors)
        #TODO stratify
        return brier, CE
    def get_AUCROC(self):
        """
        Calculate area under ROC curve
        """
        auc_llr = roc_auc_score(self.evaluator.labels, self.evaluator.LLR)
        auc_posteriors = roc_auc_score(self.evaluator.labels, self.evaluator.posteriors)
        assert '{:.4f}'.format(auc_llr) == '{:.4f}'.format(auc_posteriors)
        return auc_llr

    def get_EER(self):
        """ Get equal error rate """
        idx = np.nanargmin(np.absolute((1 - self.evaluator.tpr - self.evaluator.fpr)))
        return self.evaluator.fpr[idx], idx

    def get_AUCPR(self):
        return sklearnAUC(self.evaluator.recalls, self.evaluator.precisions)

    def get_adjusted_AUCPR(self):
        return 1 - (log(self.get_AUCPR()) / log(self.evaluator.real_positive_prior))

    def get_DCFmin(self):
        # Minimum cost on the detection cost function (discrmination-only cost)
        idx = np.argmin(self.evaluator.detection_cost_function)
        return self.evaluator.detection_cost_function[idx], idx

    def get_ECE(self):
        # Assign each prediction to a bin
        bin_bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        bin_indices = np.digitize(self.evaluator.posteriors, bin_bounds)
        # Save the accuracy, confidence and size of each bin
        positives_proportion_per_bin = np.zeros(len(bin_bounds))
        mean_predicted_score_per_bin = np.zeros(len(bin_bounds))
        counts = np.zeros(len(bin_bounds))

        for bin in range(len(bin_bounds)):
            counts[bin] = len(self.evaluator.posteriors[bin_indices == bin])
            if counts[bin] > 0:
                positives_proportion_per_bin[bin] = (self.evaluator.labels[bin_indices == bin]).sum() / counts[bin]
                mean_predicted_score_per_bin[bin] = (self.evaluator.posteriors[bin_indices == bin]).sum() / counts[bin]

        calibration_errors = positives_proportion_per_bin - mean_predicted_score_per_bin
        #self.counts_per_bin = counts
        assert np.sum(counts) == len(self.evaluator.labels)
        #self.positives_proportion_per_bin = positives_proportion_per_bin
        total_proportion_per_bin = counts / len(self.evaluator.posteriors)
        weighted_calibration_errors = np.abs(calibration_errors) * total_proportion_per_bin
        return np.sum(weighted_calibration_errors), counts, positives_proportion_per_bin

