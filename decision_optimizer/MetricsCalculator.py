from sklearn.metrics import *
from sklearn.metrics import auc as sklearnAUC
from math import log
import numpy as np

class MetricsCalculator():
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def get_proper_scores(self, balanced=False):
        brier = brier_score_loss(self.evaluator.labels, self.evaluator.positive_posteriors)
        CE = log_loss(self.evaluator.labels, self.evaluator.positive_posteriors)
        if balanced:

            brier = brier_score_loss(self.evaluator.labels[self.evaluator.labels == 1],
                                     self.evaluator.positive_posteriors[self.evaluator.labels == 1], pos_label=1) + \
                    brier_score_loss(self.evaluator.labels[self.evaluator.labels == 0],
                                     self.evaluator.positive_posteriors[self.evaluator.labels == 0], pos_label=1)

            CE = log_loss(self.evaluator.labels[self.evaluator.labels == 1],
                                     self.evaluator.positive_posteriors[self.evaluator.labels == 1], labels=[0, 1]) + \
                    log_loss(self.evaluator.labels[self.evaluator.labels == 0],
                                     self.evaluator.positive_posteriors[self.evaluator.labels == 0], labels=[0, 1])

        return brier, CE
    def get_AUCROC(self):
        """
        Calculate area under ROC curve
        """
        auc_llr = roc_auc_score(self.evaluator.labels, self.evaluator.LLR)
        auc_posteriors = roc_auc_score(self.evaluator.labels, self.evaluator.positive_posteriors)
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

    def get_calibration_errors_Naeini(self):  # (y_pred, y_true)
        # Assign each prediction to a bin
        bin_bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        bin_indices = np.digitize(self.evaluator.positive_posteriors, bin_bounds)

        positives_proportion_per_bin = np.zeros(len(bin_bounds))  # o(i)
        mean_predicted_score_per_bin = np.zeros(len(bin_bounds))  # e(i)
        counts = np.zeros(len(bin_bounds))

        for bin in range(len(bin_bounds)):
            counts[bin] = len(self.evaluator.positive_posteriors[bin_indices == bin])
            if counts[bin] > 0:
                positives_proportion_per_bin[bin] = (self.evaluator.labels[bin_indices == bin]).sum() / counts[bin]
                mean_predicted_score_per_bin[bin] = (self.evaluator.positive_posteriors[bin_indices == bin]).sum() / counts[bin]
        assert np.sum(counts) == len(self.evaluator.labels)

        naeini_calibration_errors = np.abs(positives_proportion_per_bin - mean_predicted_score_per_bin)
        total_proportion_per_bin = counts / len(self.evaluator.positive_posteriors)  # P(i)
        ECE = np.sum(naeini_calibration_errors * total_proportion_per_bin)
        MCE = np.max(np.abs(naeini_calibration_errors))

        return ECE, MCE, counts, positives_proportion_per_bin, mean_predicted_score_per_bin

    def get_calibration_errors_Guo(self, th=0.5):
        # Assign each prediction to a bin
        bin_bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        confidences = np.array([1 - p if p <= th else p for p in self.evaluator.positive_posteriors])  # confidences

        bin_indices = np.digitize(confidences, bin_bounds)
        # Save the accuracy, confidence and size of each bin
        accuracy_per_bin = np.zeros(len(bin_bounds))  # acc(Bm)
        mean_predicted_confidence_per_bin = np.zeros(len(bin_bounds))  # conf(Bm)
        counts = np.zeros(len(bin_bounds))

        for bin in range(len(bin_bounds)):
            bin_confidences = confidences[bin_indices == bin]
            bin_labels = self.evaluator.labels[bin_indices == bin]
            bin_binary_predictions = np.array(self.evaluator.positive_posteriors >= th)[bin_indices == bin]
            counts[bin] = len(bin_confidences)
            if counts[bin] > 0:
                accuracy_per_bin[bin] = np.array(bin_labels == bin_binary_predictions).sum() / counts[bin]
                mean_predicted_confidence_per_bin[bin] = bin_confidences.sum() / len(bin_confidences)
        assert np.sum(counts) == len(self.evaluator.labels)
        guo_calibration_errors = np.abs(accuracy_per_bin - mean_predicted_confidence_per_bin)
        total_proportion_per_bin = counts / len(self.evaluator.positive_posteriors)  # |Bm|/n
        ECE = np.sum(guo_calibration_errors * total_proportion_per_bin)
        MCE = np.max(np.abs(guo_calibration_errors))

        return ECE, MCE, counts, accuracy_per_bin, mean_predicted_confidence_per_bin

    def get_adaptive_calibration_error_Naeini(self, n_bins=10):
        """
        Gets the Adaptive Expected Calibration Error using traditional binary definition of calibration error
        """
        npt = len(self.evaluator.positive_posteriors)
        histedges_equalN = np.interp(np.linspace(0, npt, n_bins + 1), np.arange(npt),
                                     np.sort(self.evaluator.positive_posteriors))
        n, bin_boundaries = np.histogram(self.evaluator.positive_posteriors, histedges_equalN)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        AdaEce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculates |fraction of positives - mean positive posterior| in each bin
            in_bin = [True if (bin_lower <= x < bin_upper) else False for x in self.evaluator.positive_posteriors]  # revisar qué pasa con extremos código original
            proportion_of_bin_samples = in_bin.count(True) / len(in_bin)
            if (proportion_of_bin_samples) > 0:
                positive_fraction_in_bin = self.evaluator.labels[in_bin].sum() / in_bin.count(True)
                mean_predicted_score_in_bin = self.evaluator.positive_posteriors[in_bin].mean()
                AdaEce += abs(positive_fraction_in_bin-mean_predicted_score_in_bin) * proportion_of_bin_samples
        return AdaEce

    def get_adaptive_calibration_error_Guo(self, n_bins=10, th=0.5):
        """
        Gets the Adaptive Expected Calibration Error
        """
        confidences = np.array([1 - p if p <= th else p for p in self.evaluator.positive_posteriors])  # confidences
        predictions = np.array(self.evaluator.positive_posteriors > 0.5).astype(int)
        correct_predictions = np.where(predictions == self.evaluator.labels, 1, 0)
        npt = len(confidences)
        histedges_equalN = np.interp(np.linspace(0, npt, n_bins + 1), np.arange(npt), np.sort(confidences))
        n, bin_boundaries = np.histogram(confidences, histedges_equalN)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        AdaEce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculates |accuracy - confidence| in each bin
            in_bin = [True if (bin_lower <= x < bin_upper) else False for x in confidences]  # revisar qué pasa con extremos código original
            proportion_of_bin_samples = in_bin.count(True) / len(in_bin)
            if (proportion_of_bin_samples) > 0:
                accuracy_in_bin = correct_predictions[in_bin].mean()
                mean_confidence_in_bin = confidences[in_bin].mean()
                AdaEce += abs(accuracy_in_bin - mean_confidence_in_bin) * proportion_of_bin_samples
        return AdaEce

    def get_accuracy(self, binarized_preds, balanced=False):
        if balanced:
            return balanced_accuracy_score(self.evaluator.labels, binarized_preds)
        else:
            return accuracy_score(self.evaluator.labels, binarized_preds)