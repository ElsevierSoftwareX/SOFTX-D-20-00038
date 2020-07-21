# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import List
import abc

from asvtorch.src.utterances.utterance_list import UtteranceList

class AbstractUtteranceSelector():

    @abc.abstractmethod
    def choose_all(self, dataset: str) -> UtteranceList:
        pass

    @abc.abstractmethod
    def choose_longest(self, dataset: str, n: int) -> UtteranceList:
        pass

    @abc.abstractmethod
    def choose_random(self, dataset: str, n: int, seed: int = 0) -> UtteranceList:
        pass

    @abc.abstractmethod
    def choose_from_trials(self, dataset: str, trial_file: str, side: str = 'both') -> UtteranceList:  # both, enroll, or test
        pass

    @abc.abstractmethod
    def choose_from_list(self, dataset: str, id_list: List[str]) -> UtteranceList:
        pass

    @abc.abstractmethod
    def choose_regex(self, dataset: str, regex: str, id_type: str = 'utt') -> UtteranceList:  # utt or spk
        pass

    @abc.abstractmethod
    def choose_random_regex(self, dataset: str, n: int, regex: str, seed: int = 0) -> UtteranceList:
        pass
