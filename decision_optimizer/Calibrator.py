import numpy as np
from sklearn._isotonic import _inplace_contiguous_isotonic_regression as fastpav
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression

from decision_optimizer import evaluation_utils as utils

class Calibrator():
    def __init__(self):
        self.trainset_calibrated_scores = None
        self.trainset_positive_prior = None
        self.trainset_negative_prior = None
        self.parameters = None

        self.current_raw_scores = None
        self.current_calibrated_posteriors = None
        self.current_calibrated_LLRs = None

    def train(self, scores, labels, effective_positive_prior=None):

        self.labels = np.array(labels, dtype=np.float64)
        self.scores = np.array(scores, dtype=np.float64)
        self.trainset_positive_prior = labels.sum() / len(labels)
        self.trainset_negative_prior = 1 - self.trainset_positive_prior
        if effective_positive_prior is None:
            effective_positive_prior = self.trainset_positive_prior # que prior usar?
        self.effective_positive_prior = effective_positive_prior
        self.fit_algorithm()
        self.apply_algorithm(self.scores)
        self.trainset_calibrated_scores = self.current_calibrated_posteriors

    def apply_calibration(self, new_scores):
        self.current_raw_scores = new_scores
        self.apply_algorithm(new_scores)
        return self.current_calibrated_posteriors


class PAVCalibrator(Calibrator):
    def fit_algorithm(self):
        ii = np.lexsort((-self.labels, self.scores))  # when scores are equal, those with label 1 are considered inferior
        calibrated_scores = np.ones_like(self.scores)
        y = np.empty_like(self.scores)
        y[:] = self.labels[ii]
        fastpav(y, calibrated_scores)
        positive_prop, ff, counts = np.unique(y, return_index=True, return_counts=True)
        positive_samples = np.rint(counts * positive_prop).astype(int)
        negative_samples = counts - positive_samples
        assert positive_samples.sum() == self.labels.sum()
        assert negative_samples.sum() == len(self.labels) - self.labels.sum()
        low_bin_bounds = self.scores[ii[ff]]
        self.parameters = [positive_prop, low_bin_bounds]

    def apply_algorithm(self, current_raw_scores):
        indices = np.digitize(current_raw_scores, self.parameters[1]) - 1
        self.current_calibrated_posteriors = self.parameters[0][indices]


class HistogramCalibrator(Calibrator):
    """
    Bins the data based on equal size interval (each bin contains approximately
    equal size of samples).
    """
    def fit_algorithm(self, n_bins=10):
        sorted_indices = np.argsort(self.scores)
        binned_y_true = np.array_split(self.labels[sorted_indices], n_bins)
        binned_y_prob = np.array_split(self.scores[sorted_indices], n_bins)
        bins = []
        for i in range(len(binned_y_prob) - 1):
            last_prob = binned_y_prob[i][-1]
            next_first_prob = binned_y_prob[i + 1][0]
            bins.append((last_prob + next_first_prob) / 2.0)

        bins.append(1.0)
        self.parameters = [np.array([np.mean(value) for value in binned_y_true]), np.array(bins)]

    def apply_algorithm(self, current_raw_scores):
        """
        Predicts the calibrated probability.
        """
        indices = np.searchsorted(self.parameters[1], current_raw_scores)
        self.current_calibrated_posteriors = self.parameters[0][indices]


class LogisticRegressionWCECalibrator(Calibrator):
    """
    Calibration with a Logistic Regression using weighted cross entropy as loss function
    """
    def __init__(self, sklearn=False, use_LLR=True):
        super(Calibrator, self).__init__()
        self.use_LLR = use_LLR
        self.sklearn = sklearn

    def correct_posteriors(self, epsilon=1e-15):
        # For stability, posteriors cannot be exactly 1 or exactly 0
        posteriors_1 = np.argwhere(self.scores == 1.)[:, 0]
        if len(posteriors_1) > 0:
            #print("corrected {} posteriors that where exactly 1 with epsilon {}".format(len(posteriors_1), epsilon))
            self.scores[posteriors_1] = self.scores[posteriors_1] - epsilon

        posteriors_0 = np.argwhere(self.scores == 0.)[:, 0]
        if len(posteriors_0) > 0:
            #print("corrected {} posteriors that where exactly 0 with epsilon {}".format(len(posteriors_0), epsilon))
            self.scores[posteriors_0] = self.scores[posteriors_0] + epsilon

    def fit_algorithm(self):
        if self.use_LLR:
            self.correct_posteriors()
            x = logit(self.scores) - logit(self.effective_positive_prior)
        else:
            x = self.scores

        if self.sklearn:
            # It was verified that sklearn with class_weight='balanced' is exactly equivalent to
            # the other method IF the priors of self.labels match self.effective_positive_prior
            clf = LogisticRegression(random_state=0, class_weight='balanced').fit(x.reshape(-1, 1), self.labels)
            self.parameters = [clf.coef_[0][0], clf.intercept_[0]]
        else:
            trobj = utils.optobjective(utils.obj, sign=1.0,
                                 pos_scores=x[self.labels == 1.0],
                                 neg_scores=x[self.labels == 0.0],
                                 positive_prior=self.effective_positive_prior)
            result = minimize(trobj, np.array([1.0, 0.0]), method='L-BFGS-B', jac=True)


            self.parameters = [result.x[0], result.x[1]]

    def apply_algorithm(self, current_raw_scores):
        current_LLR = logit(current_raw_scores) - logit(self.effective_positive_prior)
        self.current_calibrated_LLRs = self.parameters[0] * current_LLR + self.parameters[1]
        self.current_calibrated_posteriors = expit(self.current_calibrated_LLRs + logit(self.effective_positive_prior))

    def apply_inverse(self, current_calibrated_scores, is_LLR=True):
        if not is_LLR:
            current_calibrated_LLR = logit(current_calibrated_scores) - logit(self.effective_positive_prior)
        else:
            current_calibrated_LLR = current_calibrated_scores
        current_raw_LLR = (current_calibrated_LLR - self.parameters[1]) / self.parameters[0]
        current_raw_scores = expit(current_raw_LLR + logit(self.effective_positive_prior))
        return current_raw_scores