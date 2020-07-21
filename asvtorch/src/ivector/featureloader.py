# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import numpy as np
import torch
from torch.utils import data

from asvtorch.src.settings.settings import Settings
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader


def _get_clip_indices(utt_start, utt_end, batch_start, batch_end):
    """ Cuts the parts of the utterance that do not fit into the batch window.

    Arguments:
        utt_start {int} -- start point of the utterance
        utt_end {int} -- end point of the utterance
        batch_start {int} -- start point of the batch window
        batch_end {int} -- end point of the batch window

    Returns:
        (int, int), bool -- a tuple containing clipped start and end point of an utterance, the boolean flag is True if the end of the utterance is inside the batch window.
    """
    if utt_end <= batch_start:
        return None
    if utt_start >= batch_end:
        return None
    start = 0
    end = utt_end - utt_start
    if utt_start < batch_start:
        start = batch_start - utt_start
    if utt_end > batch_end:
        end = batch_end - utt_start
    ends = utt_end <= batch_end
    return (start, end), ends

class _Kaldi_Dataset(data.Dataset):
    def __init__(self, dataset: UtteranceList):

        self.dataset = dataset

        frames_per_batch = Settings().posterior_extraction.batch_size_in_frames
        break_points = dataset.get_breakpoints_after_frame_selection()
        n_active_frames = break_points[-1]
        n_batches = int(np.ceil(n_active_frames / frames_per_batch))

        utt_index = 0
        self.batches = []

        for i in range(n_batches):
            batch_indices = []
            batch_endpoints = []
            window_start = i * frames_per_batch
            window_end = (i + 1) * frames_per_batch
            while utt_index < len(dataset):
                clip_indices = _get_clip_indices(break_points[utt_index], break_points[utt_index + 1], window_start, window_end)
                utt_index += 1
                if clip_indices is not None:
                    batch_indices.append((utt_index - 1, clip_indices[0]))
                    if clip_indices[1]:
                        batch_endpoints.append(break_points[utt_index])
                    else:
                        break
                else:
                    if batch_indices:
                        break
            self.batches.append((batch_indices, np.asarray(batch_endpoints)))
            batch_indices = []
            batch_endpoints = []
            utt_index -= 1

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch_indices, batch_endpoints = self.batches[index]
        frames = []
        for utt_indices in batch_indices:
            utt_index, selection_indices = utt_indices
            feats = FeatureLoader().load_features(self.dataset[utt_index])
            frames.append(feats[selection_indices[0]:selection_indices[1], :])
        frames = torch.from_numpy(np.vstack(frames))
        return frames, batch_endpoints


def _collater(batch):
    """ In this "hack" batches are already formed in the DataSet object (batch consists of a single element, which is actually the batch). 
    """
    return batch[0]

def get_feature_loader(dataset):
    dataset = _Kaldi_Dataset(dataset)
    return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=Settings().computing.posterior_extraction_dataloader_workers, collate_fn=_collater)
