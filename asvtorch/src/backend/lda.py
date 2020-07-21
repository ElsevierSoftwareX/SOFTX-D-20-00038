# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import torch

from asvtorch.src.backend.plda import _rearrange_data, _compute_within_cov, _compute_between_cov

class Lda:
    def __init__(self, W):      
        self.W = W

    @classmethod
    def train(cls, data, speaker_labels, device): ## Expects centered data
        print('Training LDA...')
        data = data.to(device)
        data, class_boundaries = _rearrange_data(data, speaker_labels)
        print('Computing within class covariance...')
        Sw = _compute_within_cov(data, class_boundaries)
        print('Computing between class covariance...')
        Sb = _compute_between_cov(data, class_boundaries)
        print('Simultaneous diagonalization...')
        l, U = torch.symeig(Sw, eigenvectors=True)
        W = torch.rsqrt(l) * U
        Sbb = torch.chain_matmul(W.t(), Sb, W)
        l, U = torch.symeig(Sbb, eigenvectors=True)
        W = torch.matmul(W, U) 
        print('LDA trained!...')
        return Lda(W)

    def project(self, vectors, lda_dim):
        return torch.matmul(vectors, self.W[:, -lda_dim:])