# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfcinv


def _get_tar_and_nontar_scores(scores: np.ndarray, labels: List[str], tar_label: str, nontar_label: str) -> Tuple[np.ndarray, np.ndarray]:

    if not tar_label:
        tar_labels = set(('Target', 'target', 'TARGET', '1', 't', 'T'))
    else:
        tar_labels = set([tar_label])

    if not nontar_label:
        nontar_labels = set(('Nontarget', 'nontarget', 'NONTARGET', 'Non-target', 'non-target', 'NON-TARGET', '0', 'n', 'N', 'f', 'F'))
    else:
        nontar_labels = set([nontar_label])

    #print(labels[0])

    target_scores = []
    nontarget_scores = []
    for index, score in enumerate(scores):
        if labels[index] in tar_labels:
            target_scores.append(score)
        elif labels[index] in nontar_labels:
            nontarget_scores.append(score)

    return np.asarray(target_scores), np.asarray(nontarget_scores)


def _compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(scores: np.ndarray, labels: List[str], tar_label: str = None, nontar_label: str = None):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    target_scores, nontarget_scores = _get_tar_and_nontar_scores(scores, labels, tar_label, nontar_label)
    print('{} target and {} non-target scores'.format(target_scores.size, nontarget_scores.size))
    return _compute_eer(target_scores, nontarget_scores)

def _compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    frr, far, thresholds = _compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_min_dcf(scores: np.ndarray, labels: List[str], p_target: float, c_miss: float, c_fa: float, tar_label: str = None, nontar_label: str = None):
    target_scores, nontarget_scores = _get_tar_and_nontar_scores(scores, labels, tar_label, nontar_label)
    frr, far, thresholds = _compute_det_curve(target_scores, nontarget_scores)
    return _compute_min_dcf(frr, far, thresholds, p_target, c_miss, c_fa)

def _compute_min_dcf_tar_nontar(target_scores: np.ndarray, nontarget_scores: np.ndarray, p_target: float, c_miss: float, c_fa: float):
    frr, far, thresholds = _compute_det_curve(target_scores, nontarget_scores)
    return _compute_min_dcf(frr, far, thresholds, p_target, c_miss, c_fa)

# Adapted from KALDI:
def _compute_min_dcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def _icdf(x):
   # computes the inverse of cumulative distribution function in x
    x = -np.sqrt(2) * erfcinv(2 * (x + np.finfo(float).eps))
    return x

def _prepare_det_plot(ticks, limits):
    tick_labels = ['{0:g}'.format(float(x)) for x in (ticks*100).tolist()]
    ticks = _icdf(ticks)
    limits = _icdf(limits)
    return ticks, tick_labels, limits

def _convert_rates(frr, far):
    frr = _icdf(frr)
    far = _icdf(far)
    return frr, far



class DetPlot():

    def __init__(self, ticks: List[float] = None, axis_min: float = 0.001, axis_max: float = 0.5):
        self.target_scores = None
        self.nontarget_scores = None
        if not ticks:
            ticks = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
        else:
            ticks = np.asarray(ticks)
        limits = np.asarray([axis_min, axis_max])
        plt.xlabel('False Acceptance Rate [%]')
        plt.ylabel('False Rejection Rate [%]')
        ticks, tick_labels, limits = _prepare_det_plot(ticks, limits)
        plt.xticks(ticks, tick_labels)
        plt.yticks(ticks, tick_labels)
        plt.axis('square')
        plt.xlim(limits)
        plt.ylim(limits)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        diag_line = np.asarray([0, 0.99])
        diag_line = _convert_rates(diag_line, diag_line)[0]
        plt.plot(diag_line, diag_line, color='lightgray', linestyle='--', linewidth=0.5)

    def set_scores_and_labels(self, scores: np.ndarray, labels: List[str], tar_label: str = None, nontar_label: str = None):
        self.target_scores, self.nontarget_scores = _get_tar_and_nontar_scores(scores, labels, tar_label, nontar_label)

    def plot_det_curve(self, **plt_kwargs):
        frr, far, _ = _compute_det_curve(self.target_scores, self.nontarget_scores)
        frr, far = _convert_rates(frr, far)
        plt.plot(far, frr, **plt_kwargs)

    def plot_eer(self, **plt_kwargs):
        eer = _compute_eer(self.target_scores, self.nontarget_scores)[0]
        eer = _icdf(eer)
        if 'marker' not in plt_kwargs:
            plt_kwargs['marker'] = 'o'
        plt.plot(eer, eer, **plt_kwargs)

    def plot_min_dcf(self, p_target: float, c_miss: float, c_fa: float, **plt_kwargs):
        frr, far, thresholds = _compute_det_curve(self.target_scores, self.nontarget_scores)
        min_dcf_threshold = _compute_min_dcf_tar_nontar(self.target_scores, self.nontarget_scores, p_target, c_miss, c_fa)[1]
        idx = (np.abs(thresholds - min_dcf_threshold)).argmin()
        frr, far = _convert_rates(frr, far)
        if 'marker' not in plt_kwargs:
            plt_kwargs['marker'] = 'd'
        plt.plot(far[idx], frr[idx], **plt_kwargs)

