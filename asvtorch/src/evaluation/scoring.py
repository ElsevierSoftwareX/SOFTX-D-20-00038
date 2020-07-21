# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import Tuple, List

import torch
import numpy as np

from asvtorch.src.misc.ordered_set import OrderedSet
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.backend.plda import Plda
from asvtorch.src.settings.settings import Settings

def prepare_scoring(data: UtteranceList, trial_file: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Returns trial labels and indices to find correct embeddings from embedding array for each trial.

    Arguments:
        data {UtteranceList} -- An utteranceList containing all enrollment and test segments
        trial_file {str} -- Filename of trial file

    Returns:
        Tuple[List[str], List[Tuple[int, int]]] -- Trial labels in a list, A list containg indices of left embeddings and indices of right embeddings for each trial.
    """

    utt2index = {}
    for index, segment in enumerate(data.get_utt_labels()):
        utt2index[segment] = index

    labels = []
    indices = []
    with open(trial_file) as f:
        for line in f:
            parts = line.split()
            labels.append(parts[2].strip())
            indices.append((utt2index[parts[0]], utt2index[parts[1]]))
    return labels, indices


def score_trials_plda(data: UtteranceList, indices: List[Tuple[int, int]], plda: Plda) -> Tuple[np.ndarray]:
    """Scores trials using PLDA. If the number of all enrollment-test embedding pairs is less than Settings().backend.max_all_vs_all_score_count, then all-vs-all scoring strategy is used; otherwise, pairwise scoring is used.

    Arguments:
        data {UtteranceList} -- An utteranceList containing all enrollment and test segments
        indices {List[Tuple[int, int]]} -- A list containg indices of left embeddings and indices of right embeddings for each trial.
        plda {Plda} -- PLDA model used for scoring

    Returns:
        Tuple[np.ndarray] -- PLDA scores
    """

    left_indices = OrderedSet()
    right_indices = OrderedSet()   
    for left_index, right_index in indices:     
        left_indices.add(left_index)
        right_indices.add(right_index)
    n_left = len(left_indices)
    n_right = len(right_indices)

    scores = []
    if n_left * n_right < Settings().backend.max_all_vs_all_score_count:
        print('Less than {} enroll-test utterance pairs --> using all-vs-all scoring strategy ({} x {})!'.format(Settings().backend.max_all_vs_all_score_count, n_left, n_right))
        left_vectors = torch.zeros(n_left, data.embeddings.shape[1], dtype=data.embeddings.dtype, device=data.embeddings.device)
        right_vectors = torch.zeros(n_right, data.embeddings.shape[1], dtype=data.embeddings.dtype, device=data.embeddings.device)
        left2index = {}
        for matrix_index, embedding_index in enumerate(left_indices):
            left_vectors[matrix_index, :] = data.embeddings[embedding_index, :]
            left2index[embedding_index] = matrix_index
        right2index = {}
        for matrix_index, embedding_index in enumerate(right_indices):
            right_vectors[matrix_index, :] = data.embeddings[embedding_index, :]
            right2index[embedding_index] = matrix_index
        score_matrix = plda.score_all_vs_all(left_vectors, right_vectors, Settings().backend.plda_dim)
        print('Scoring completed! Organizing scores...')
        for left_index, right_index in indices:
            scores.append(score_matrix[left2index[left_index], right2index[right_index]])
        scores = np.asarray(scores)
        print('Scores ready!')
    else:
        print('More than {} enroll-test utterance pairs ({} x {}) --> using chunked pairwise scoring of trials (chunk size = {})!'.format(Settings().backend.max_all_vs_all_score_count, n_left, n_right, Settings().backend.pairwise_scoring_chunk_size))
        print('Scoring {} trials... '.format(len(indices)), end='')
        i = 0
        while i < len(indices):
            chunk_indices = indices[i:i+Settings().backend.pairwise_scoring_chunk_size]
            i += Settings().backend.pairwise_scoring_chunk_size
            left_vectors = torch.zeros(len(chunk_indices), data.embeddings.shape[1], dtype=data.embeddings.dtype, device=data.embeddings.device)
            right_vectors = torch.zeros(len(chunk_indices), data.embeddings.shape[1], dtype=data.embeddings.dtype, device=data.embeddings.device)
            for index, (left_index, right_index) in enumerate(chunk_indices):
                left_vectors[index, :] = data.embeddings[left_index, :]
                right_vectors[index, :] = data.embeddings[right_index, :]
            scores.append(plda.score_trials(left_vectors, right_vectors, Settings().backend.plda_dim))
            print('{:.1f}M'.format(i/1e6), end=' ')
        print()
        scores = np.hstack(scores)
    return scores
