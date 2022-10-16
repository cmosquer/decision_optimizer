import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve, CalibrationDisplay
from decision_optimizer.Evaluator import Evaluator

class Plotter():
    def __init__(self):
        self.axs_dict = self.prepare_empty_axs_dict()

    def save_fig(self, path, save_editable=False):
        for name, ax in self.axs_dict.items():
            ax.legend()
        self.axs_dict['fig'].savefig(path, dpi=500)
        if save_editable:
            self.axs_dict['fig'].savefig(path[:-4]+'.svg')

    def add_evaluator(self, evaluator, color, evaluation_name):
        assert isinstance(evaluator, Evaluator)
        self.evaluator = evaluator
        self.fig_color = color
        self.fig_name = evaluation_name

        # Plot curves
        self.plot_PR()
        self.plot_DCF()
        self.plot_CALIBRATION_CURVE_NAEINI()
        self.plot_CALIBRATION_CURVE_GUO()
        self.plot_ROC()
        self.plot_LLR_histograms()

        self.add_EER_point()

    def plot_PR(self):
        ax = self.axs_dict['PR']
        lbl = 'AUC={:.2f} (adj={:.2f})'.format(
            self.evaluator.metrics['AUCPR'], self.evaluator.metrics['adjusted_AUCPR'])
        ax.plot(self.evaluator.recalls, self.evaluator.precisions,
                color=self.fig_color, label=lbl
                )
        ax.axhline(self.evaluator.real_positive_prior, color='gray', ls='--', label='Positive prior')

    def plot_DCF(self):
        ax = self.axs_dict['DCF']
        ax.plot(self.evaluator.thresholds_ROC,
                self.evaluator.detection_cost_function,
                # label=f"{self.fig_name}",
                color=self.fig_color)
        min_cost, min_cost_idx = self.evaluator.metrics_calculator.get_DCFmin()
        lbl = 'min cost={:.2f} (theta={:.2f})'.format(  # self.fig_name,
            min_cost,
            self.evaluator.thresholds_ROC[min_cost_idx])
        ax.scatter(self.evaluator.thresholds_ROC[min_cost_idx],
                   min_cost, marker='x',
                   label=lbl,
                   color=self.fig_color)

    def plot_CALIBRATION_CURVE_NAEINI(self):
        ax = self.axs_dict['CALIBRATION_CURVE_NAEINI']
        positives_proportion_per_bin, mean_predicted_score_per_bin = calibration_curve(self.evaluator.labels,
                                                                                       self.evaluator.positive_posteriors,
                                                                                       n_bins=10)
        calibration_plot = CalibrationDisplay(positives_proportion_per_bin,
                                              mean_predicted_score_per_bin,
                                              self.evaluator.positive_posteriors)
        ECE_Naeini, MCE_Naeini, counts_per_bin, \
        positives_proportion_per_bin, mean_predicted_score_per_bin = self.evaluator.metrics_calculator.get_calibration_errors_Naeini()

        calibration_plot.plot(ax=ax, color=self.fig_color, label="ECE={:.3f}".format(ECE_Naeini)
                              # label=self.fig_name
                              )
        # bins_centers = np.linspace(0, 1, len(self.positives_proportion_per_bin))
        bins_centers = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) + 0.05

        # Error bars
        ax.bar(bins_centers, bins_centers, width=0.1, alpha=0.2, edgecolor='black', color='r', hatch='\\')
        ax.bar(bins_centers, positives_proportion_per_bin, width=0.1, alpha=0.2, edgecolor='black', color='b')
        ax.set_xlabel('Positive posterior')
        ax.set_ylabel('Fraction of positives')
        for j, c in enumerate(counts_per_bin):
            ax.text(bins_centers[j], positives_proportion_per_bin[j] + 0.01, str(int(c)))

    def plot_CALIBRATION_CURVE_GUO(self):
        ax = self.axs_dict['CALIBRATION_CURVE_GUO']

        ECE_Guo, MCE_Guo, counts_per_bin, \
        accuracy_per_bin, mean_confidence_per_bin = self.evaluator.metrics_calculator.get_calibration_errors_Guo()

        # bins_centers = np.linspace(0, 1, len(self.positives_proportion_per_bin))
        bins_centers = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) + 0.05

        # Error bars
        ax.bar(bins_centers, bins_centers, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')
        # Draw bars and identity line
        ax.bar(bins_centers, accuracy_per_bin, width=0.1, alpha=0.9, edgecolor='black', color='b',
               label="ECE={:.3f}".format(ECE_Guo))
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        for j, c in enumerate(counts_per_bin):
            ax.text(bins_centers[j], accuracy_per_bin[j] + 0.01, str(int(c)))
        ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, linewidth=1, label='Perfectly calibrated')


    def plot_ROC(self):
        ax = self.axs_dict['ROC']
        ax.set_title(f'n={len(self.evaluator.labels)} (pos={np.sum(self.evaluator.labels)})')
        lbl = 'AUC={:.2f}'.format(  # self.fig_name,
            self.evaluator.metrics['AUCROC'])
        ax.plot(self.evaluator.fpr, self.evaluator.tpr, label=lbl,
                color=self.fig_color
                )
    def plot_LLR_histograms(self, N_bins=100):
        ax = self.axs_dict['LLR_HISTOGRAM']
        h_orange = ax.hist(self.evaluator.LLR[self.evaluator.labels.astype(int) == 1],
                           color='orange', density=True,
                           bins=N_bins, histtype='step')
        h_blue = ax.hist(self.evaluator.LLR[self.evaluator.labels.astype(int) == 0],
                         color='blue', density=True,
                         bins=N_bins, histtype='step')

        #ax.set_title("{}".format(self.fig_name))

    def add_EER_point(self):
        # Add to ROC curve
        ax = self.axs_dict['ROC']
        EER_ROC, EER_idx_ROC = self.evaluator.metrics_calculator.get_EER()
        lbl = "EER={:.2f} (theta={:.2f})".format(  # self.fig_name,
            EER_ROC, self.evaluator.thresholds_ROC[EER_idx_ROC])
        ax.scatter(EER_ROC, self.evaluator.tpr[EER_idx_ROC],
                   marker='x', label=lbl,
                   color=self.fig_color)

        # Add to PR curve
        ax = self.axs_dict['PR']
        EER_idx_PR = 0
        for j, recall in enumerate(self.evaluator.recalls):
            if recall <= self.evaluator.metrics['EER']:
                EER_idx_PR = j
                break
        EER_PR = self.evaluator.recalls[EER_idx_PR]
        print("EER recall ", self.evaluator.recalls[EER_idx_PR],
              "EER precision ", self.evaluator.precisions[EER_idx_PR])
        ax.scatter(EER_PR, self.evaluator.precisions[EER_idx_PR],
                   marker='x', label=lbl,
                   color=self.fig_color)

    def add_threshold_point(self):
        # Plot scatter points on specific threshold

        # Add to ROC curve
        ax = self.axs_dict['ROC']
        lbl = "{} with theta={:.2f}. FPR={:.2f} TPR={:.2f}  (actual theta={:.2f})".format(  # self.fig_name,
            self.evaluator.current_threshold_name, self.evaluator.current_theta,
            self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_FPR'],
            self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_TPR'],
            self.evaluator.thresholds_ROC[self.evaluator.current_theta_idx_ROC])
        ax.scatter(self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_FPR'],
                   self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_TPR'],
                   marker='o', label=lbl,
                   color=self.fig_color)
        # Add to PR curve
        ax = self.axs_dict['PR']
        lbl = self.evaluator.current_threshold_name + " (actual theta={:.2f})".format(
            self.evaluator.thresholds_PR[self.evaluator.current_theta_idx_PR])
        ax.scatter(self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_recall'],
                   self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_precision'],
                   marker='o', label=lbl,
                   color=self.fig_color)

        # Add to cost curve
        ax = self.axs_dict['DCF']
        lbl = '{} cost={:.2f} (theta={:.2f})'.format(
            # self.fig_name,
            self.evaluator.current_threshold_name,
            self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_cost'],
            self.evaluator.current_theta, self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_cost'])
        ax.scatter(self.evaluator.thresholds_ROC[self.evaluator.current_theta_idx_ROC],
                   self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_cost'],
                   marker='o', label=lbl,
                   color=self.fig_color)

        ax = self.axs_dict['LLR_HISTOGRAM']
        ax.axvline(self.evaluator.current_theta, color='gray', ls='--',
                   label="predefined theta={:.2f}".format(self.evaluator.current_theta))

    def prepare_empty_axs_dict(self):
        fig = plt.figure(figsize=(12, 18))
        gs = fig.add_gridspec(4, 2)
        axs_dict = {'fig': fig}

        axs_dict['ROC'] = fig.add_subplot(gs[0, 0])
        axs_dict['ROC'].set_ylabel('Recall (TPR)')
        axs_dict['ROC'].set_xlim((0, 1))
        axs_dict['ROC'].set_ylim((0, 1))
        axs_dict['ROC'].set_xlabel('1 - specificity (FPR)')
        axs_dict['ROC'].set_title('ROC curve')

        axs_dict['PR'] = fig.add_subplot(gs[0, 1])
        axs_dict['PR'].set_xlim((0, 1))
        axs_dict['PR'].set_ylim((0, 1))
        axs_dict['PR'].set_ylabel('Precision')
        axs_dict['PR'].set_xlabel('Recall')
        axs_dict['PR'].set_title('Precision-recall curve')

        axs_dict['DCF'] = fig.add_subplot(gs[1, 0])
        axs_dict['DCF'].set_title('Detection cost function')
        axs_dict['DCF'].set_ylabel('Cost')
        axs_dict['DCF'].set_xlabel('LLR threshold (theta)')

        axs_dict['CALIBRATION_CURVE_NAEINI'] = fig.add_subplot(gs[2, 0])
        axs_dict['CALIBRATION_CURVE_NAEINI'].set_xlim((0, 1))
        axs_dict['CALIBRATION_CURVE_NAEINI'].set_ylim((0, 1))


        axs_dict['CALIBRATION_CURVE_GUO'] = fig.add_subplot(gs[2, 1])
        axs_dict['CALIBRATION_CURVE_GUO'].set_xlim((0, 1))
        axs_dict['CALIBRATION_CURVE_GUO'].set_ylim((0, 1))

        axs_dict['LLR_HISTOGRAM'] = fig.add_subplot(gs[3, :])

        return axs_dict

class MultipleRunsPlotter():
    def __init__(self):
        self.axs_dict = self.prepare_empty_axs_dict()
        self.evaluators = []

    def save_fig(self, path, save_editable=False):
        for name, ax in self.axs_dict.items():
            ax.legend()
        self.axs_dict['fig'].savefig(path, dpi=500)
        if save_editable:
            self.axs_dict['fig'].savefig(path[:-4]+'.svg')

    def add_evaluator(self, evaluator, color, evaluation_name):
        assert isinstance(evaluator, Evaluator)
        self.evaluators.append(evaluator)
        self.fig_color = color
        self.fig_name = evaluation_name

    def plot_multiple_runs(self):
        metrics_arrays = {}

        for metric in self.evaluators[0].metrics.keys():
            metrics_arrays[metric] = np.zeros(len(self.evaluators))

        for j, evaluator in enumerate(self.evaluators):
            for metric in metrics_arrays.keys():
                metrics_arrays[metric][j] = evaluator.metrics[metric]

        for metric in metrics_arrays.keys():
            metrics_arrays[metric+'_mean'] = metrics_arrays[metric].mean()
            metrics_arrays[metric+'_std'] = metrics_arrays[metric].std()

        # Plot curves
        self.plot_PR()
        self.plot_DCF()
        self.plot_CALIBRATION_CURVE()
        self.plot_ROC()
        self.plot_LLR_histograms()

        self.add_EER_point()

    def plot_PR(self):
        ax = self.axs_dict['PR']
        lbl = 'AUC={:.2f} (adj={:.2f})'.format(
            self.evaluator.metrics['AUCPR'], self.evaluator.metrics['adjusted_AUCPR'])
        ax.plot(self.evaluator.recalls, self.evaluator.precisions,
                color=self.fig_color, label=lbl
                )
        ax.axhline(self.evaluator.real_positive_prior, color='gray', ls='--', label='Positive prior')

    def plot_DCF(self):
        ax = self.axs_dict['DCF']
        ax.plot(self.evaluator.thresholds_ROC,
                self.evaluator.detection_cost_function,
                # label=f"{self.fig_name}",
                color=self.fig_color)
        min_cost, min_cost_idx = self.evaluator.metrics_calculator.get_DCFmin()
        lbl = 'min cost={:.2f} (theta={:.2f})'.format(  # self.fig_name,
            min_cost,
            self.evaluator.thresholds_ROC[min_cost_idx])
        ax.scatter(self.evaluator.thresholds_ROC[min_cost_idx],
                   min_cost, marker='x',
                   label=lbl,
                   color=self.fig_color)

    def plot_CALIBRATION_CURVE(self):
        ax = self.axs_dict['CALIBRATION_CURVE']
        positives_proportion_per_bin, mean_predicted_score_per_bin = calibration_curve(self.evaluator.labels,
                                                                                       self.evaluator.positive_posteriors,
                                                                                       n_bins=10)
        calibration_plot = CalibrationDisplay(positives_proportion_per_bin,
                                              mean_predicted_score_per_bin,
                                              self.evaluator.positive_posteriors)

        calibration_plot.plot(ax=ax, color=self.fig_color,
                              # label=self.fig_name
                              )
        # bins_centers = np.linspace(0, 1, len(self.positives_proportion_per_bin))
        bins_centers = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) + 0.05

        # Error bars
        ax.bar(bins_centers, bins_centers, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')
        # Draw bars and identity line
        ECE, MCE, counts_per_bin, positives_proportion_per_bin = self.evaluator.metrics_calculator.get_ECE()
        ax.bar(bins_centers, positives_proportion_per_bin, width=0.1, alpha=1, edgecolor='black', color='b')
        for j, c in enumerate(counts_per_bin):
            ax.text(bins_centers[j], positives_proportion_per_bin[j] + 0.01, str(int(c)))
        ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.1, linewidth=1)

    def plot_ROC(self):
        ax = self.axs_dict['ROC']
        ax.set_title(f'n={len(self.evaluator.labels)} (pos={np.sum(self.evaluator.labels)})')
        lbl = 'AUC={:.2f}'.format(  # self.fig_name,
            self.evaluator.metrics['AUCROC'])
        ax.plot(self.evaluator.fpr, self.evaluator.tpr, label=lbl,
                color=self.fig_color
                )
    def plot_LLR_histograms(self, N_bins=100):
        ax = self.axs_dict['LLR_HISTOGRAM']
        h_orange = ax.hist(self.evaluator.LLR[self.evaluator.labels.astype(int) == 1],
                           color='orange', density=True,
                           bins=N_bins, histtype='step')
        h_blue = ax.hist(self.evaluator.LLR[self.evaluator.labels.astype(int) == 0],
                         color='blue', density=True,
                         bins=N_bins, histtype='step')

        #ax.set_title("{}".format(self.fig_name))

    def add_EER_point(self):
        # Add to ROC curve
        ax = self.axs_dict['ROC']
        EER_ROC, EER_idx_ROC = self.evaluator.metrics_calculator.get_EER()
        lbl = "EER={:.2f} (theta={:.2f})".format(  # self.fig_name,
            EER_ROC, self.evaluator.thresholds_ROC[EER_idx_ROC])
        ax.scatter(EER_ROC, self.evaluator.tpr[EER_idx_ROC],
                   marker='x', label=lbl,
                   color=self.fig_color)

        # Add to PR curve
        ax = self.axs_dict['PR']
        EER_idx_PR = 0
        for j, recall in enumerate(self.evaluator.recalls):
            if recall <= self.evaluator.metrics['EER']:
                EER_idx_PR = j
                break
        EER_PR = self.evaluator.recalls[EER_idx_PR]
        print("EER recall ", self.evaluator.recalls[EER_idx_PR],
              "EER precision ", self.evaluator.precisions[EER_idx_PR])
        ax.scatter(EER_PR, self.evaluator.precisions[EER_idx_PR],
                   marker='x', label=lbl,
                   color=self.fig_color)

    def add_threshold_point(self):
        # Plot scatter points on specific threshold

        # Add to ROC curve
        ax = self.axs_dict['ROC']
        lbl = "{} with theta={:.2f}. FPR={:.2f} TPR={:.2f}  (actual theta={:.2f})".format(  # self.fig_name,
            self.evaluator.current_threshold_name, self.evaluator.current_theta,
            self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_FPR'],
            self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_TPR'],
            self.evaluator.thresholds_ROC[self.evaluator.current_theta_idx_ROC])
        ax.scatter(self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_FPR'],
                   self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_TPR'],
                   marker='o', label=lbl,
                   color=self.fig_color)
        # Add to PR curve
        ax = self.axs_dict['PR']
        lbl = self.evaluator.current_threshold_name + " (actual theta={:.2f})".format(
            self.evaluator.thresholds_PR[self.evaluator.current_theta_idx_PR])
        ax.scatter(self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_recall'],
                   self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_precision'],
                   marker='o', label=lbl,
                   color=self.fig_color)

        # Add to cost curve
        ax = self.axs_dict['DCF']
        lbl = '{} cost={:.2f} (theta={:.2f})'.format(
            # self.fig_name,
            self.evaluator.current_threshold_name,
            self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_cost'],
            self.evaluator.current_theta, self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_cost'])
        ax.scatter(self.evaluator.thresholds_ROC[self.evaluator.current_theta_idx_ROC],
                   self.evaluator.metrics[f'{self.evaluator.current_threshold_name}_cost'],
                   marker='o', label=lbl,
                   color=self.fig_color)

        ax = self.axs_dict['LLR_HISTOGRAM']
        ax.axvline(self.evaluator.current_theta, color='gray', ls='--',
                   label="predefined theta={:.2f}".format(self.evaluator.current_theta))

    def prepare_empty_axs_dict(self):
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(3, 2)
        axs_dict = {'fig': fig}

        axs_dict['ROC'] = fig.add_subplot(gs[0, 0])
        axs_dict['ROC'].set_ylabel('Recall (TPR)')
        axs_dict['ROC'].set_xlim((0, 1))
        axs_dict['ROC'].set_ylim((0, 1))
        axs_dict['ROC'].set_xlabel('1 - specificity (FPR)')
        axs_dict['ROC'].set_title('ROC curve')

        axs_dict['PR'] = fig.add_subplot(gs[0, 1])
        axs_dict['PR'].set_xlim((0, 1))
        axs_dict['PR'].set_ylim((0, 1))
        axs_dict['PR'].set_ylabel('Precision')
        axs_dict['PR'].set_xlabel('Recall')
        axs_dict['PR'].set_title('Precision-recall curve')

        axs_dict['DCF'] = fig.add_subplot(gs[1, 0])
        axs_dict['DCF'].set_title('Detection cost function')
        axs_dict['DCF'].set_ylabel('Cost')
        axs_dict['DCF'].set_xlabel('LLR threshold (theta)')

        axs_dict['CALIBRATION_CURVE'] = fig.add_subplot(gs[1, 1])
        axs_dict['CALIBRATION_CURVE'].set_xlim((0, 1))
        axs_dict['CALIBRATION_CURVE'].set_ylim((0, 1))

        axs_dict['LLR_HISTOGRAM'] = fig.add_subplot(gs[2, :])
        fig.close()

        return axs_dict

