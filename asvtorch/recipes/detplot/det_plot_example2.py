# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import sys
import os
# Adding the project root to the path to make imports to work regardless from where this file was executed:
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('asvtorch', 1)[0])

import numpy as np 
import matplotlib.pyplot as plt

import asvtorch.src.misc.fileutils as fileutils
import asvtorch.src.evaluation.eval_metrics as eval_metrics

mindcf_params = [0.05, 1, 1]
colors = {'xvector': 'dodgerblue', 'neural_ivector': 'green', 'ivector': 'orangered'}
legend_labels = {'xvector': 'x-vector', 'neural_ivector': 'neural i-vector', 'ivector': 'i-vector'}
linestyles = {'xvector': '-', 'neural_ivector': '--', 'ivector': ':'}

titles = {'voxcelebEC': 'VoxCeleb-E (cleaned)', 'sitw': 'SITW core-core'}

fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.4), sharey=True)

for list_index, trial_list in enumerate(('voxcelebEC', 'sitw')):

    plt.sca(axes[list_index])

    key_file = os.path.join(fileutils.get_folder_of_file(__file__), 'keyfiles', '{}_key_file.txt'.format(trial_list))
    labels = np.loadtxt(key_file, usecols=2, dtype=str).tolist()

    ticks = [0.0002, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4]
    det_plot = eval_metrics.DetPlot(ticks=ticks, axis_min=0.0005, axis_max=0.6)       

    for system in ('xvector', 'neural_ivector', 'ivector'):
        score_file = os.path.join(fileutils.get_folder_of_file(__file__), 'scorefiles', '{}_scores_{}.txt'.format(trial_list, system))
        scores = np.loadtxt(score_file)

        det_plot.set_scores_and_labels(scores, labels)
        if list_index == 0:
            det_plot.plot_det_curve(color=colors[system], linestyle=linestyles[system], label=legend_labels[system])
        else:
            det_plot.plot_det_curve(color=colors[system], linestyle=linestyles[system])
        det_plot.plot_eer(marker='o', fillstyle='none', markeredgewidth=1, color=colors[system], markersize=8)
        det_plot.plot_min_dcf(*mindcf_params, marker='p', fillstyle='none', markeredgewidth=1, color=colors[system], markersize=9)
    
    if list_index == 1:
        plt.plot(100, 100, marker='o', linestyle='None', fillstyle='none', markeredgewidth=1, color='black', markersize=8, label='EER point')
        plt.plot(100, 100, marker='p', linestyle='None', fillstyle='none', markeredgewidth=1, color='black', markersize=9, label='MinDCF point')
        plt.ylabel('')
        plt.tick_params(axis='y', which='both', left=False)

    plt.legend(edgecolor='black')
    plt.title(titles[trial_list], fontsize=10)

plt.subplots_adjust(wspace=0.05)
#plt.tight_layout()

plt.savefig(os.path.join(fileutils.get_folder_of_file(__file__), 'det_plot2.pdf'))
plt.show()