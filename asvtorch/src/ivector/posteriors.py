# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import time
import os

import numpy as np
import torch

from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.ivector.featureloader import get_feature_loader
from asvtorch.src.ivector.gmm import DiagGmm, Gmm
from asvtorch.src.ivector.posterior_io import PosteriorWriter
from asvtorch.src.settings.settings import Settings
import asvtorch.src.misc.fileutils as fileutils


def extract_posteriors(data: UtteranceList):

    print('{}: Extracting posteriors for {} utterances using ubm "{}"...'.format(data.name, len(data), Settings().ivector.ubm_name))

    ubm = Gmm.from_kaldi(fileutils.get_ubm_file(), Settings().computing.device)
    diag_ubm = DiagGmm.from_full_gmm(ubm, Settings().computing.device)

    dataloader = get_feature_loader(data)

    # To sub-batching to reduce GPU memory usage
    sub_batch_count = int(np.ceil(ubm.means.size()[0] / ubm.means.size()[1]))

    output_filename = os.path.join(fileutils.get_posterior_folder(), data.name)
    wspecifier_top_posterior = "ark,scp:{0}.ark,{0}.scp".format(output_filename)
    scp_file = output_filename + '.scp'
    posterior_writer = PosteriorWriter(wspecifier_top_posterior)

    posterior_buffer = torch.Tensor()
    top_buffer = torch.LongTensor()
    count_buffer = torch.LongTensor()

    start_time = time.time()
    frame_counter = 0
    utterance_counter = 0

    start_time = time.time()
    for batch_index, batch in enumerate(dataloader):

        frames, end_points = batch
        frames = frames.to(Settings().computing.device)
        frames_in_batch = frames.size()[0]

        chunks = torch.chunk(frames, sub_batch_count, dim=0)
        top_gaussians = []
        for chunk in chunks:
            posteriors = diag_ubm.compute_posteriors(chunk)
            top_gaussians.append(torch.topk(posteriors, Settings().posterior_extraction.n_top_gaussians, dim=0, largest=True, sorted=False)[1])

        top_gaussians = torch.cat(top_gaussians, dim=1)

        posteriors = ubm.compute_posteriors_top_select(frames, top_gaussians)

        # Posterior thresholding:
        max_indices = torch.argmax(posteriors, dim=0)
        mask = posteriors.ge(Settings().posterior_extraction.posterior_threshold)
        top_counts = torch.sum(mask, dim=0)
        posteriors[~mask] = 0
        divider = torch.sum(posteriors, dim=0)
        mask2 = divider.eq(0) # For detecting special cases
        posteriors[:, ~mask2] = posteriors[:, ~mask2] / divider[~mask2]
        # Special case that all the posteriors are discarded (force to use 1):
        posteriors[max_indices[mask2], mask2] = 1
        mask[max_indices[mask2], mask2] = 1
        top_counts[mask2] = 1

        # Vectorize the data & move to cpu memory
        posteriors = posteriors.t().masked_select(mask.t())
        top_gaussians = top_gaussians.t().masked_select(mask.t())
        posteriors = posteriors.cpu()
        top_gaussians = top_gaussians.cpu()
        top_counts = top_counts.cpu()

        end_points = end_points - frame_counter  # relative end_points in a batch

        if end_points.size != 0:
            # Save utterance data that continues from the previous batch:
            psave = torch.cat([posterior_buffer, posteriors[:torch.sum(top_counts[:end_points[0]])]])
            tsave = torch.cat([top_buffer, top_gaussians[:torch.sum(top_counts[:end_points[0]])]])
            csave = torch.cat([count_buffer, top_counts[:end_points[0]]])
            posterior_writer.write(data[utterance_counter].utt_id, csave, psave, tsave)
            utterance_counter += 1

            # Save utterance data that is fully included in this batch:
            for start_point, end_point in zip(end_points[:-1], end_points[1:]):
                psave = posteriors[torch.sum(top_counts[:start_point]):torch.sum(top_counts[:end_point])]
                tsave = top_gaussians[torch.sum(top_counts[:start_point]):torch.sum(top_counts[:end_point])]
                csave = top_counts[start_point:end_point]
                posterior_writer.write(data[utterance_counter].utt_id, csave, psave, tsave)
                utterance_counter += 1

            # Buffer partial data to be used in the next batch:
            posterior_buffer = posteriors[torch.sum(top_counts[:end_points[-1]]):]
            top_buffer = top_gaussians[torch.sum(top_counts[:end_points[-1]]):]
            count_buffer = top_counts[end_points[-1]:]
        else:
            # Buffer the whole data for the next batch (if the utterance is longer than the current batch (special case)):
            posterior_buffer = torch.cat([posterior_buffer, posteriors])
            top_buffer = torch.cat([top_buffer, top_gaussians])
            count_buffer = torch.cat([count_buffer, top_counts])

        frame_counter += frames_in_batch

        print('{:.0f} seconds elapsed, batch {}/{}: {}, utterance count (roughly) = {}'.format(time.time() - start_time, batch_index+1, dataloader.__len__(), frames.size(), len(end_points)))

    posterior_writer.close()
    print('Posterior computation completed in {:.3f} seconds'.format(time.time() - start_time))

    print('Storing scp-file pointers to UtteranceList {}...'.format(data.name))
    rxspecifiers = []
    with open(scp_file) as f:
        for line in f:
            rxspecifiers.append(line.split()[-1].strip())
    for utterance, rxspecifier in zip(data, rxspecifiers):
        utterance.posterior_location = rxspecifier
    print('Done!')
