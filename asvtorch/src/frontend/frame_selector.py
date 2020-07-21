# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import sys

import numpy as np

from asvtorch.src.settings.settings import Settings

# This class is used to for storing and applying VAD and diarization labels.
class FrameSelector:
    def __init__(self, boolean_selectors: np.ndarray):
        self.frame_count = boolean_selectors.size
        self.selected_count = np.sum(boolean_selectors)
        self.bits = np.packbits(boolean_selectors)

    def select(self, frames: np.ndarray, id_for_error_message: str = '') -> np.ndarray:
        boolean_selectors = np.unpackbits(self.bits, count=self.frame_count).astype(bool)
        size_diff = boolean_selectors.size - frames.shape[0]
        if size_diff != 0:
            if abs(size_diff) > Settings().features.vad_mismatch_tolerance:
                if size_diff > 0:
                    sys.exit('[ERROR] {}: frame selector has {} extra values'.format(id_for_error_message, size_diff))
                else:
                    sys.exit('[ERROR] {}: {} values are missing from frame selector'.format(id_for_error_message, abs(size_diff)))
            elif size_diff < 0:
                boolean_selectors = np.hstack((boolean_selectors, np.asarray([False]*abs(size_diff), dtype=bool)))
                print('[WARNING] {}: frame selector was missing {} values'.format(id_for_error_message, abs(size_diff)))
            else:
                boolean_selectors = boolean_selectors[:-size_diff]
                print('[WARNING] {}: frame selector had {} extra values'.format(id_for_error_message, size_diff))
        return frames[boolean_selectors, :]

    def intersect(self, boolean_selectors: np.ndarray):
        if self.frame_count != boolean_selectors.size:
            sys.exit('ERROR: Cannot intersect selectors of different sizes')
        self_selectors = np.unpackbits(self.bits, count=self.frame_count).astype(bool)
        intersection = np.logical_and(self_selectors, boolean_selectors)
        self.__init__(intersection)

    def clip_to_length(self, n_frames: int):
        if self.selected_count <= n_frames:
            return
        startpos = np.random.randint(self.selected_count - n_frames + 1)
        boolean_selectors = np.unpackbits(self.bits, count=self.frame_count).astype(bool)
        indices_of_selected = np.where(boolean_selectors)[0]
        indices_to_set_zero = np.concatenate((indices_of_selected[:startpos], indices_of_selected[startpos+n_frames:]))
        boolean_selectors[indices_to_set_zero] = 0
        self.__init__(boolean_selectors)