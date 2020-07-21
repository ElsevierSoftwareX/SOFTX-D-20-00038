
# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import torch
from torch.utils import data

from asvtorch.src.settings.settings import Settings
from asvtorch.src.utterances.utterance_list import UtteranceList

class _NeuralStatDataset(data.Dataset):
    def __init__(self, utterance_list: UtteranceList, centering_means=None):
        self.utterance_list = utterance_list
        if centering_means is not None:
            self.centering_means = centering_means.cpu()
        else:
            self.centering_means = None

    def __len__(self):
        return self.utterance_list.stats.zeroth.size()[0]

    def __getitem__(self, index):
        z = self.utterance_list.stats.zeroth[index, :]
        f = self.utterance_list.stats.first[index, :, :]
        return z, f

    def collater(self, batch):
        """Collates sufficient statistics from many utterances to form a batch.
        
        Returns:
            Tensor -- 0th order statistics (number of utterances x number of components)
            Tensor -- 1st order statistics (#components x feat_dim x #utterances)
        """
        n, f = zip(*batch)
        n = torch.stack(n, dim=0)
        f = torch.stack(f, dim=2)
        return n, f

def get_stat_loader(utterance_list: UtteranceList, centering_means: torch.Tensor):
    dataset = _NeuralStatDataset(utterance_list, centering_means)
    return data.DataLoader(dataset, batch_size=Settings().ivector.batch_size_in_utts, shuffle=False, num_workers=Settings().computing.ivector_dataloader_workers, collate_fn=dataset.collater)
