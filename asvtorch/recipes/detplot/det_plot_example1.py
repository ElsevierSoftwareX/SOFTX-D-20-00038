# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import sys
import os
# Adding the project root to the path to make imports to work regardless from where this file was executed:
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('asvtorch', 1)[0])

from scipy import stats
import numpy as np 
import matplotlib.pyplot as plt

import asvtorch.src.misc.fileutils as fileutils
import asvtorch.src.evaluation.eval_metrics as eval_metrics

# Generating random sets of target and nontarget scores for two systems:
N = 40000
scale = 1.9
sys1_tar_samples = stats.norm.rvs(loc=6, scale=scale, size=N)
sys1_non_samples = stats.norm.rvs(loc=-1, scale=scale*0.8, size=N)
sys2_tar_samples = stats.norm.rvs(loc=8.5, scale=scale, size=N)
sys2_non_samples = stats.norm.rvs(loc=-2, scale=scale*1.5, size=N)

# Stack target and nontarget scores together:
sys1_scores = np.hstack((sys1_tar_samples, sys1_non_samples))
sys2_scores = np.hstack((sys2_tar_samples, sys2_non_samples))
# Generate list of labels correspoding to the above scores:
labels = ['target'] * N + ['nontarget'] * N

# Set DCF parameters (here, two different settings):
mindcf1_params = [0.01, 1, 1]
mindcf2_params = [0.99, 1, 10]

# DET plot configuration:

fig = plt.figure(figsize=(3.9,3.9))
ticks = [0.0002, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4]
det_plot = eval_metrics.DetPlot(ticks=ticks, axis_min=0.00025)

# Plotting system 1:
det_plot.set_scores_and_labels(sys1_scores, labels)
det_plot.plot_det_curve(color='navy', label='System 1')
det_plot.plot_eer(marker='o', fillstyle='none', markeredgewidth=1, color='navy', markersize=8)
det_plot.plot_min_dcf(*mindcf1_params, marker='p', fillstyle='none', markeredgewidth=1, color='navy', markersize=8)
det_plot.plot_min_dcf(*mindcf2_params, marker='d', fillstyle='none', markeredgewidth=1, color='navy', markersize=8)

# Plotting system 2:
det_plot.set_scores_and_labels(sys2_scores, labels)
det_plot.plot_det_curve(color='darkorange', linestyle='--',  label='System 2')
det_plot.plot_eer(marker='o', fillstyle='none', markeredgewidth=1, color='darkorange', markersize=8)
det_plot.plot_min_dcf(*mindcf1_params, marker='p', fillstyle='none', markeredgewidth=1, color='darkorange', markersize=8)
det_plot.plot_min_dcf(*mindcf2_params, marker='d', fillstyle='none', markeredgewidth=1, color='darkorange', markersize=8)

# This is just to show the markers in the legend (the point (100, 100) is outside the DET plot so nothing gets drawn)
plt.plot(100, 100, marker='o', fillstyle='none', markeredgewidth=1, color='black', markersize=8, label='EER')
plt.plot(100, 100, marker='p', fillstyle='none', markeredgewidth=1, color='black', markersize=8, label='MinDCF1')
plt.plot(100, 100, marker='d', fillstyle='none', markeredgewidth=1, color='black', markersize=8, label='MinDCF2')

plt.legend(edgecolor='black')
plt.tight_layout()
plt.savefig(os.path.join(fileutils.get_folder_of_file(__file__), 'det_plot1.pdf'))
plt.show()
