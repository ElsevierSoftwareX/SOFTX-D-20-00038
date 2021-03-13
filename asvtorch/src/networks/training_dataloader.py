# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from collections import defaultdict, deque
from typing import NamedTuple
import random
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader
from asvtorch.src.settings.settings import Settings

class SpeakerData(NamedTuple):
    utterance_list: UtteranceList
    utterance_queue: deque

class TrainingDataset(Dataset):
    def __init__(self, data: UtteranceList):
        self.data = data
        self.batch_data = None

    def _shuffle(self, seed):
        np.random.seed(seed)
        indices = np.random.permutation(np.arange(len(self.data.utterances)))
        self.data = UtteranceList([self.data[i] for i in np.nditer(indices)], self.data.name)

    def shuffle_and_rebatch(self, seed):
        self._shuffle(seed)
        self.create_batches(seed)

    def create_batches(self, seed):
        minibatch_size = Settings().network.minibatch_size // Settings().computing.world_size  ## Keep effective minibatch size the same in multiprocessing
        np.random.seed(seed)
        random.seed(seed)
        print('Creating minibatches...')
        spk2data = defaultdict(UtteranceList)
        for utt in self.data:
            spk2data[utt.spk_id].utterances.append(utt)
        for key in spk2data:
            spk2data[key].name = key
            spk2data[key] = SpeakerData(spk2data[key], deque(spk2data[key].utterances))
        spks = list(spk2data.keys())
        number_of_minibathes = math.ceil(Settings().network.utts_per_speaker_in_epoch * len(spks) / minibatch_size)
        print('One epoch contains...')
        print('  - {} speech clips from each of {} training speakers'.format(Settings().network.utts_per_speaker_in_epoch, len(spks)))
        print('  - {} minibatches of size {}'.format(number_of_minibathes, minibatch_size))
        spk_deque = deque()
        self.batch_data = []
        for _ in range(number_of_minibathes):
            self.batch_data.append([])
            clip_length = np.random.randint(Settings().network.min_clip_size, Settings().network.max_clip_size + 1)
            for _ in range(minibatch_size):
                if not spk_deque:
                    shuffled_speakers = spks.copy()
                    random.shuffle(shuffled_speakers)
                    spk_deque.extend(shuffled_speakers)
                next_utt = None
                while next_utt is None:
                    spk = spk_deque.popleft()
                    next_utt, clip_indices = _select_next_clip(spk2data[spk], clip_length)
                self.batch_data[-1].append((next_utt, clip_indices))

        # Make the number of minibatches divisable by world size:
        remainder = len(self.batch_data) % Settings().computing.world_size
        if remainder > 0:
            self.batch_data = self.batch_data[:-remainder]

        # Split minibatches to different processes:
        self.batch_data = self.batch_data[Settings().computing.local_process_rank::Settings().computing.world_size]
        
        print('Minibatches created!')

    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, index):
        feat_list = []
        labels = []
        for utt, clip_indices in self.batch_data[index]:
            feats = FeatureLoader().load_features(utt)
            feat_list.append(feats[clip_indices[0]:clip_indices[1], :])
            labels.append(utt.spk_id)
        features = torch.from_numpy(np.transpose(np.dstack(feat_list), (2, 1, 0)))
        return features, torch.LongTensor(labels)

def _select_next_clip(speaker_data: SpeakerData, clip_length: int): ## REMEMBER VAD TOLERANCE
    for _ in range(len(speaker_data.utterance_list.utterances) + len(speaker_data.utterance_queue)):
        if not speaker_data.utterance_queue:
            shuffled_utts = speaker_data.utterance_list.utterances.copy()
            random.shuffle(shuffled_utts)
            speaker_data.utterance_queue.extend(shuffled_utts)
        utt = speaker_data.utterance_queue.popleft()
        frame_count = utt.get_minimum_selected_frame_count()
        if frame_count >= clip_length:
            start_point = np.random.randint(0, frame_count - clip_length + 1)
            end_point = start_point + clip_length
            return utt, (start_point, end_point)
    print('Warning: speaker "{}" does not have training utterances of length {} or longer. Skipping!'.format(speaker_data.utterance_list.name, clip_length))
    return None, None

def _collater(batch):
    # Batch is already formed in the DataSet object (batch consists of a single element, which is actually the batch itself).
    return batch[0]

def get_dataloader(data: UtteranceList):
    dataset = TrainingDataset(data)
    print('Feature loader initialized!')
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=Settings().computing.network_dataloader_workers, collate_fn=_collater)
