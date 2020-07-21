# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union, Optional
from collections import defaultdict, Counter
import time
import os
import pickle
import random

import numpy as np
import torch

from asvtorch.src.utterances.utterance import Utterance
from asvtorch.src.ivector.sufficient_stats import SufficientStats
import asvtorch.src.misc.fileutils as fileutils

@dataclass
class UtteranceList:
    utterances: List[Utterance] = field(default_factory=list)
    name: str = ''
    embeddings: torch.Tensor = field(init=False)
    neural_features: List[torch.Tensor] = field(init=False)
    adapted_means: torch.Tensor = field(init=False)
    stats: SufficientStats = field(init=False)

    def __len__(self):
        return len(self.utterances)

    # combine utterance list with one or more utterancelists
    def combine(self, *args: UtteranceList):
        utt_set = set(self.utterances)
        for arg in args:
            for utt in arg:
                if not utt in utt_set:
                    utt_set.add(utt)
                    self.utterances.append(utt)

    def subtract(self, utterance_list: UtteranceList):
        utt_set = set(utterance_list.utterances)
        utts = []
        for utterance in self.utterances:
            if utterance not in utt_set:
                utts.append(utterance)
        print('{} - {}: {} utterances subtracted from {} utterances'.format(self.name, utterance_list.name, len(self.utterances) - len(utts), len(self.utterances)))
        self.utterances = utts

    def choose_random(self, n_utterances: int, seed: int = 0):
        if len(self.utterances) <= n_utterances:
            print('{}: Choosing {} utterances...'.format(self.name, n_utterances))
            print('Dataset has only {} utterances! All utterances retained!'.format(len(self.utterances)))
            return
        np.random.seed(seed)
        self.utterances = random.sample(self.utterances, n_utterances)


    def clip_utterances_to_length(self, min_n_frames: int, max_n_frames: int, seed: int = 0):
        np.random.seed(seed)
        for utterance in self.utterances:
            n_frames = np.random.randint(min_n_frames, max_n_frames + 1)
            utterance.clip_to_length(n_frames)
        print('{}: Utterances were clipped (randomly) to max length of {} - {} frames!'.format(self.name, min_n_frames, max_n_frames))

    def remove_short_utterances(self, min_length: int):
        orig_count = len(self.utterances)
        self.utterances = [x for x in self.utterances if x.get_minimum_selected_frame_count() >= min_length]
        print('{}: {}/{} utterances were removed because they were shorter than {} frames!'.format(self.name, orig_count - len(self.utterances), orig_count, min_length))

    def remove_speakers_with_few_utterances(self, min_utt_count: int):
        orig_count = len(self.utterances)
        utt_per_spk_counter = Counter()
        for utt in self.utterances:
            utt_per_spk_counter[utt.spk_id] += 1
        self.utterances = [x for x in self.utterances if utt_per_spk_counter[x.spk_id] >= min_utt_count]
        num_removed = 0
        for key in utt_per_spk_counter:
            if utt_per_spk_counter[key] < min_utt_count:
                num_removed += 1
        print('{}: {} speakers ({} utterances) were removed because they did not have at least {} utterances!'.format(self.name, num_removed, orig_count - len(self.utterances), min_utt_count))

    def select_random_speakers(self, n_speakers: int, seed: int = 0):
        spks = set()
        for utt in self.utterances:
            spks.add(utt.spk_id)
        if len(spks) <= n_speakers:
            print('{}: Selecting {} speakers...'.format(self.name, n_speakers))
            print('Dataset has only {} speakers! No speakers removed!'.format(len(spks)))
            return
        random.seed(seed)
        selected_spks = set(random.sample(spks, n_speakers))
        self.utterances = [x for x in self.utterances if x.spk_id in selected_spks]
        print('{}: Randomly selected {} speakers from {} speakers!'.format(self.name, n_speakers, len(spks)))

    def get_spk_labels(self) -> Union[List[str], List[int]]:
        labels = []
        for utt in self.utterances:
            labels.append(utt.spk_id)
        return labels

    def get_utt_labels(self) -> Union[List[str]]:
        labels = []
        for utt in self.utterances:
            labels.append(utt.utt_id)
        return labels

    def convert_labels_to_numeric(self):
        labels = self.get_spk_labels()
        index_dict = defaultdict(list)
        for index, label in enumerate(labels):
            index_dict[label].append(index)
        for label, key in enumerate(index_dict):
            for index in index_dict[key]:
                self.utterances[index].spk_id = label

    def get_number_of_speakers(self):
        return len(set(self.get_spk_labels()))

            
    # Used in the i-vector code only (The last breakpoint is the total selected frame count)
    def get_breakpoints_after_frame_selection(self) -> np.ndarray:
        counts = []
        for utterance in self.utterances:
            counts.append(utterance.frame_selector.selected_count)
        cs = np.cumsum(np.asarray(counts), dtype=int)
        break_points = np.concatenate((np.atleast_1d(np.asarray(0, dtype=int)), cs))
        return break_points

    # def get_total_number_of_selected_frames(self) -> int:
    #     count = 0
    #     for utterance in self.utterances:
    #         count += utterance.frame_selector.selected_count
    #     return count

    def __getitem__(self, item):
        return self.utterances[item]

    def save(self, filename, folder: Optional[str] = None):
        if not folder:
            folder = fileutils.get_utterance_folder()
        output_file = os.path.join(folder, fileutils.ensure_ext(filename, 'pickle'))
        start_time = time.time()
        print('Saving: {}'.format(output_file))
        with open(output_file, 'wb') as f:
            pickle.dump(self, f, protocol=4)
        print('Saved ({:.3f} s): {}'.format(time.time() - start_time, output_file))

    @classmethod
    def load(cls, filename, folder: Optional[str] = None):
        if not folder:
            folder = fileutils.get_utterance_folder()
        input_file = os.path.join(folder, fileutils.ensure_ext(filename, 'pickle'))
        start_time = time.time()
        print('Loading: {}'.format(input_file))
        with open(input_file, 'rb') as f:
            utt_list = pickle.load(f)
        print('Loaded ({:.3f} s): {}'.format(time.time() - start_time, input_file))
        return utt_list
