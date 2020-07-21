# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from dataclasses import dataclass, field
from typing import List, Union, Tuple, Generator
import sys

from asvtorch.src.frontend.frame_selector import FrameSelector
from asvtorch.src.settings.settings import Settings

@dataclass
class Utterance:
    feature_location: Union[str, List[str]]     # If list is given, utterance is formed by combining multiple files
    frame_selector: Union[FrameSelector, List[FrameSelector]] # If list is given, utterance is formed by combining multiple files
    utt_id: str
    spk_id: Union[str, int]
    posterior_location: str = field(init=False)
    

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.utt_id == other.utt_id
        else:
            return False

    def __hash__(self):
        return hash(self.utt_id)

    def __post_init__(self):
        if isinstance(self.feature_location, list) and isinstance(self.frame_selector, list):
            if len(self.feature_location) != len(self.frame_selector):
                sys.exit('Error: the number of feature locations should match the number of frame selectors!')
        else:
            if not (isinstance(self.feature_location, str) and isinstance(self.frame_selector, FrameSelector)):
                print(type(self.feature_location), type(self.frame_selector))
                sys.exit('Error: both feature location and frame selector should be lists or not at the same time!')
        if self.utt_id == self.spk_id:
            del self.spk_id  # Save space (probably a testing utterance --> no spk labels needed)

    def generator(self) -> Generator[Tuple[str, FrameSelector], None, None]:
        if isinstance(self.feature_location, list):
            for feat_loc, frame_selector in zip(self.feature_location, self.frame_selector):
                yield feat_loc, frame_selector
        else:
            yield self.feature_location, self.frame_selector

    def clip_to_length(self, n_frames: int):
        self.frame_selector.clip_to_length(n_frames)

    def get_minimum_selected_frame_count(self):
        frame_count = 0
        for _, frame_selector in self.generator():
            frame_count += frame_selector.selected_count - Settings().features.vad_mismatch_tolerance
        return max((frame_count, 0))

    def get_frame_count(self):
        frame_count = 0
        for _, frame_selector in self.generator():
            frame_count += frame_selector.frame_count
        return frame_count
