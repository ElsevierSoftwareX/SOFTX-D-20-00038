# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import Tuple, List

import numpy as np
import torch

from asvtorch.src.backend.plda import Plda

def compute_adaptive_snorm_stats(xvectors: torch.Tensor, cohort_xvectors: torch.Tensor, plda: Plda, plda_dim: int, cohort_size: int):
    print('Computing adaptive s-norm statistics...')
    normalization_scores = plda.score_all_vs_all_chunked(xvectors, cohort_xvectors, plda_dim)

    # Adaptive cohort selection (selecting the closest matches)
    normalization_scores = np.partition(normalization_scores, -cohort_size)[:, -cohort_size:]
    means = np.mean(normalization_scores, axis=1)
    stds = np.std(normalization_scores, axis=1)
    print('s-norm statistics computed!')
    return means, stds

# When scores are orginized pairwise manner:
def apply_snorm(scores: np.ndarray, snorm_stats: Tuple[np.ndarray, np.ndarray], indices: List[Tuple[int, int]]):
    print('Applying s-norm...')
    for index, score in enumerate(scores):
        scores[index] = 0.5 * ((score - snorm_stats[0][indices[index][0]]) / snorm_stats[1][indices[index][0]] +  # Enrollment normalization
                (score - snorm_stats[0][indices[index][1]]) / snorm_stats[1][indices[index][1]])  # Test normalization
    print('s-norm applied!')
    return scores

# When scores are in all-vs-all matrix:
def apply_snorm_square_matrix(scores: np.ndarray, snorm_stats: Tuple[np.ndarray, np.ndarray]):
    print('Applying s-norm...')
    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            scores[i, j] = 0.5 * ((scores[i, j] - snorm_stats[0][i]) / snorm_stats[1][i] +  # Enrollment normalization
                (scores[i, j] - snorm_stats[0][j]) / snorm_stats[1][j])  # Test normalization
    print('s-norm applied!')
    return scores
        
                