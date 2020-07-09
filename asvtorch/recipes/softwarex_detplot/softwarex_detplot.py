import sys
import os
# Adding the project root to the path to make imports to work regardless from where this file was executed:
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('asvtorch', 1)[0])

from scipy import stats
import numpy as np 
import matplotlib.pyplot as plt

import asvtorch.src.misc.fileutils as fileutils
import asvtorch.src.evaluation.eval_metrics as eval_metrics

mindcf_params = [0.05, 1, 1]
colors = {'xvector': 'dodgerblue', 'neural_ivector': 'green', 'ivector': 'orangered'}
legend_labels = {'xvector': 'x-vector', 'neural_ivector': 'neural i-vector', 'ivector': 'i-vector'}
linestyles = {'xvector': '-', 'neural_ivector': '--', 'ivector': ':'}
#markers = {'xvector': 'o', 'neural_ivector': 'D', 'ivector': '>'}

titles = {'voxcelebEC': 'VoxCeleb-E (cleaned)', 'sitw': 'SITW core-core'}

#figure = plt.figure(figsize=(3.9, 3.9))

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True)

# add a big axis, hide frame

for list_index, trial_list in enumerate(('voxcelebEC', 'sitw')):

    plt.sca(axes[list_index])

    #if list_index == 0:
    #    ax1 = plt.subplot(1, 2, list_index+1)
    #else:
    #    plt.subplot(1, 2, list_index+1, sharex=ax1)
    
    #ax2 = plt.subplot(212, sharex = ax1)

    key_file = os.path.join(fileutils.get_folder_of_file(__file__), '{}_key_file.txt'.format(trial_list))
    np_labels = np.loadtxt(key_file, usecols=2, dtype=str)
    labels = np_labels.tolist()

    ticks = [0.0002, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4]
    det_plot = eval_metrics.DetPlot(ticks=ticks, axis_min=0.00025)       

    for index, system in enumerate(('xvector', 'neural_ivector', 'ivector')):
        score_file = os.path.join(fileutils.get_folder_of_file(__file__), '{}_scores_{}.txt'.format(trial_list, system))
        scores = np.loadtxt(score_file)

        scores[np_labels == '1'] += index*3
        #print(scores.shape)

        det_plot.set_scores_and_labels(scores, labels)
        det_plot.plot_det_curve(color=colors[system], linestyle=linestyles[system], label=legend_labels[system])
        det_plot.plot_eer(marker='o', fillstyle='none', markeredgewidth=1, color=colors[system], markersize=8)
        det_plot.plot_min_dcf(*mindcf_params, marker='p', fillstyle='none', markeredgewidth=1, color=colors[system], markersize=9)

    plt.plot(100, 100, marker='o', linestyle='None', fillstyle='none', markeredgewidth=1, color='black', markersize=8, label='EER point')
    plt.plot(100, 100, marker='p', linestyle='None', fillstyle='none', markeredgewidth=1, color='black', markersize=9, label='MinDCF point')

    if list_index == 0:
        plt.legend(edgecolor='black')

    #plt.xlabel('')
    if list_index == 1:
        plt.ylabel('')
        plt.tick_params(axis='y', which='both', left=False)
    plt.title(titles[trial_list], fontsize=10)

    #plt.tight_layout()


#fig.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axis
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#plt.xlabel('False Acceptance Rate (FAR) [%]')
#plt.ylabel('False Rejection Rate (FRR) [%]')

plt.subplots_adjust(wspace=0.05)
#plt.tight_layout()

plt.savefig(os.path.join(fileutils.get_folder_of_file(__file__), 'det_subplots.pdf'))
plt.show()