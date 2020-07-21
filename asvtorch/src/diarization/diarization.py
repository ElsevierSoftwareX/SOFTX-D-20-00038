# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import sys
from collections import defaultdict

import numpy as np

from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.settings.settings import Settings

def apply_diarization_file(utterance_list: UtteranceList, filename: str):
    """Applies a diarization file given in NIST SRE 2019 format. Columns of this file are [uttID, start time (in seconds), end time (in seconds)].

    Arguments:
        utterance_list {UtteranceList} -- [description]
        filename {str} -- Diarization file
    """

    id2utt = {}
    for utt in utterance_list.utterances:
        id2utt[utt.utt_id] = utt

    id2segments = defaultdict(list)
    with open(filename) as f:
        for line in f:
            utt_id, start, end = line.split()           
            start_frame_index = _time_to_frame_index(float(start))
            end_frame_index = _time_to_frame_index(float(end)) + 1
            id2segments[utt_id].append((start_frame_index, end_frame_index))

    for key in id2segments:
        num_frames = id2utt[key].get_frame_count()
        diar_labels = np.zeros(num_frames, dtype=bool)
        for start, end in id2segments[key]:
            if start >= num_frames:
                sys.exit('ERROR: Diarization segment starts after the end of the utterance')
            if end > num_frames:
                if end - num_frames > 200:
                    print('WARNING: {} has {} extra diarization labels'.format(key, end - num_frames))
                end = num_frames
            diar_labels[start:end] = 1      
        id2utt[key].frame_selector.intersect(diar_labels)

def _time_to_frame_index(time: float):
    return int(time * 1000 / Settings().features.frame_shift)
