# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import Type, List
import os
import sys

from asvtorch.src.misc.singleton import Singleton
import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.utterances.abstract_utterance_selector import AbstractUtteranceSelector
from asvtorch.src.utterances.kaldi_utterance_selector import KaldiUtteranceSelector
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.settings.settings import Settings


class UtteranceSelector(AbstractUtteranceSelector, metaclass=Singleton):

    def choose_all(self, dataset: str) -> UtteranceList:
        check_dataset_folder_exists(dataset)
        selector = _detect_utterance_selector(dataset)
        return selector.choose_all(dataset)

    def choose_longest(self, dataset: str, n: int) -> UtteranceList:
        check_dataset_folder_exists(dataset)
        selector = _detect_utterance_selector(dataset)
        return selector.choose_longest(dataset, n)

    def choose_random(self, dataset: str, n: int, seed: int = 0) -> UtteranceList:
        check_dataset_folder_exists(dataset)
        selector = _detect_utterance_selector(dataset)
        return selector.choose_random(dataset, n, seed)

    def choose_from_trials(self, dataset: str, trial_file: str, side: str = 'both') -> UtteranceList:  # both, enroll, or test
        check_dataset_folder_exists(dataset)
        selector = _detect_utterance_selector(dataset)
        return selector.choose_from_trials(dataset, trial_file, side)

    def choose_from_list(self, dataset: str, id_list: List[str]) -> UtteranceList:
        check_dataset_folder_exists(dataset)
        selector = _detect_utterance_selector(dataset)
        return selector.choose_from_list(dataset, id_list)

    def choose_regex(self, dataset: str, regex: str, id_type: str = 'utt') -> UtteranceList:  # utt or spk
        check_dataset_folder_exists(dataset)
        selector = _detect_utterance_selector(dataset)
        return selector.choose_regex(dataset, regex, id_type)

    def choose_random_regex(self, dataset: str, n: int, regex: str, seed: int = 0) -> UtteranceList:
        check_dataset_folder_exists(dataset)
        selector = _detect_utterance_selector(dataset)
        return selector.choose_random_regex(dataset, n, regex, seed)

def _detect_utterance_selector(dataset: str) -> Type[AbstractUtteranceSelector]:
    list_folder = fileutils.get_list_folder(dataset)
    if Settings().features.use_kaldi:
        return KaldiUtteranceSelector()
    raise NotImplementedError

def check_dataset_folder_exists(dataset: str) -> bool: 
    dataset_folder = os.path.join(Settings().paths.output_folder, Settings().paths.feature_and_list_folder, dataset)
    if not os.path.exists(dataset_folder):
        sys.exit('Error: dataset folder "{}" does not exist. Is the dataset name typed correctly? Is the dataset already prepared?'.format(dataset_folder))