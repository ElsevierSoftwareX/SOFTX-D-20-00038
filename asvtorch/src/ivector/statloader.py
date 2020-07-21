
# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import numpy as np
import torch
from torch.utils import data

import asvtorch.src.ivector.posterior_io
from asvtorch.src.settings.settings import Settings
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader

class _StatDataset(data.Dataset):
    def __init__(self, dataset: UtteranceList, data_dims, second_order: bool, centering_means: torch.Tensor = None):
        # self.feat_rxspecifiers = rxspecifiers[0]
        # self.vad_rxspecifiers = rxspecifiers[1]
        # self.posterior_rxspecifiers = rxspecifiers[2]
        # self.feature_loader = feature_loader
        self.dataset = dataset
        if centering_means is not None:
            self.centering_means = centering_means.cpu().numpy()
        else:
            self.centering_means = None
        self.second_order = second_order
        self.data_dims = data_dims
        if second_order:
            self.second_order_sum = np.zeros((data_dims[0], data_dims[1], data_dims[1]), dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def accumulate_stats(self, feats, counts, posteriors, indices):
        """Computes 0th and 1st order statistics from the selected posteriors.
        
        Arguments:
            feats {ndarray} -- Feature array (feature vectors as rows).
            counts {ndarray} -- Array containing the numbers of selected posteriors for each frame.
            posteriors {ndarray} -- Array containing posteriors (flattened).
            indices {ndarray} -- Array containing Gaussian indices (flattened).
        
        Returns:
            ndarray -- 0th order statistics (row vector).
            ndarray -- 1st order statistics (row index = component index).
        """

        n = np.zeros(self.data_dims[0], dtype=np.float32)
        f = np.zeros(self.data_dims, dtype=np.float32)
        posterior_count = 0
        for frame_index in range(counts.size):
            end = posterior_count+counts[frame_index]
            gaussian_indices = indices[posterior_count:end]
            frame_posteriors = posteriors[posterior_count:end]
            n[gaussian_indices] += frame_posteriors
            f[gaussian_indices, :] += np.outer(frame_posteriors, feats[frame_index, :])
            if self.second_order:
                if self.centering_means is not None:
                    feats_centered = np.atleast_3d(np.atleast_2d(feats[frame_index, :]) - self.centering_means[gaussian_indices, :]) # Ok: (atleast_2d and atleast_3d prepend and append dimensions, respectively)
                    feat_outer = np.matmul(feats_centered, np.transpose(feats_centered, (0, 2, 1)))
                else:
                    feat_outer = np.outer(feats[frame_index, :], feats[frame_index, :])
                self.second_order_sum[gaussian_indices, :, :] += frame_posteriors[:, np.newaxis, np.newaxis] * feat_outer
            posterior_count += counts[frame_index]            
        return n, f

    def __getitem__(self, index):
        feats = FeatureLoader().load_features(self.dataset[index])
        counts, posteriors, indices = asvtorch.src.ivector.posterior_io.load_posteriors(self.dataset[index].posterior_location)
        n, f = self.accumulate_stats(feats, counts, posteriors, indices)
        if self.centering_means is not None:
            f -= n[:, None] * self.centering_means
        return n, f

    def collater(self, batch):
        """Collates sufficient statistics from many utterances to form a batch.
        
        Returns:
            Tensor -- 0th order statistics (number of utterances x number of components)
            Tensor -- 1st order statistics (#components x feat_dim x #utterances)
            Tensor -- Sum of 2nd order statistics (#components x feat_dim x feat_dim)
        """
        n, f = zip(*batch)
        n = np.stack(n, axis=0) 
        f = np.stack(f, axis=2)

        if self.second_order:
            s = self.second_order_sum
            self.second_order_sum = np.zeros(self.second_order_sum.shape, dtype=np.float32)  # Zero the accumulator
            return torch.from_numpy(n), torch.from_numpy(f), torch.from_numpy(s)
        else:
            return torch.from_numpy(n), torch.from_numpy(f)
            

def get_stat_loader(dataset: UtteranceList, data_dims, second_order, centering_means):
    dataset = _StatDataset(dataset, data_dims, second_order, centering_means)
    return data.DataLoader(dataset, batch_size=Settings().ivector.batch_size_in_utts, shuffle=False, num_workers=Settings().computing.ivector_dataloader_workers, collate_fn=dataset.collater)