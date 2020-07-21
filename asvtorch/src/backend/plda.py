# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from collections import defaultdict
import time
import math

import numpy as np
import torch
from asvtorch.src.settings.settings import Settings


class Plda:
    def __init__(self, St: torch.Tensor, Sb: torch.Tensor):      
        self.St = St
        self.Sb = Sb
        self.plda_dim = 0
        self.l = None
        self.uk = None
        self.qhat = None
    
    @classmethod
    def train_closed_form(cls, data: torch.Tensor, speaker_labels, device):
        print('Training PLDA...')
        data = data.to(device)
        data, class_boundaries = _rearrange_data(data, speaker_labels)
        print('Computing within class covariance...')
        Sw = _compute_within_cov(data, class_boundaries)
        print('Computing data covariance...')
        St = _compute_cov(data)
        Sb = St - Sw
        print('PLDA trained!...')
        return Plda(St, Sb)

    @classmethod
    def train_em(cls, data, speaker_labels, plda_dim, iterations, device):
        print('Initializing simplified PLDA...')
        data = data.to(device)     
        n_total_sessions, data_dim = data.size()
        F = torch.randn(data_dim, plda_dim, device=device)
        F = _orthogonalize_columns(F)
        S = 1000 * torch.randn(data_dim, data_dim, device=device)
        data_covariance = torch.matmul(data.t(), data)
        data_list, count_list = _arrange_data_by_counts(data, speaker_labels)
        eye_matrix = torch.eye(plda_dim, device=device)

        for iteration in range(1, iterations+1):
            print('Iteration {}...'.format(iteration), end='')
            iter_start_time = time.time()
            
            FS = torch.solve(F, S.t())[0].t()
            FSF = torch.matmul(FS, F) 
                
            dataEh = torch.zeros(data_dim, plda_dim, device=device)
            Ehh = torch.zeros(plda_dim, plda_dim, device=device)

            for count_data, count in zip(data_list, count_list):
                Sigma = torch.inverse(eye_matrix + count * FSF)
                my = torch.chain_matmul(Sigma, FS.repeat(1, count), count_data.view(-1, data_dim * count).t())            
                dataEh += torch.matmul(count_data.t(), my.repeat(count, 1).t().reshape(count_data.size()[0], -1))
                Ehh += count * (my.size()[1] * Sigma + torch.matmul(my, my.t()))              
            
            F = torch.solve(dataEh.t(), Ehh.t())[0].t()
            S = (data_covariance - torch.chain_matmul(F, Ehh, F.t())) / n_total_sessions

            Sb = torch.matmul(F, F.t())
            St = Sb + S

            print(' [elapsed time = {:0.1f} s]'.format(time.time() - iter_start_time))
            yield Plda(St, Sb)

    def _compute_scoring_matrices(self, plda_dim):
        if self.plda_dim != plda_dim:
            self.plda_dim = plda_dim
            #iSt = torch.pinverse(self.St, Settings().backend.plda_rcond)
            #iS = torch.pinverse(self.St - torch.chain_matmul(self.Sb, iSt, self.Sb), Settings().backend.plda_rcond)
            iSt = torch.inverse(self.St)
            iS = torch.inverse(self.St - torch.chain_matmul(self.Sb, iSt, self.Sb))
            Q = iSt - iS
            P = torch.chain_matmul(iSt, self.Sb, iS)
            U, s = torch.svd(P)[:2]
            self.l = s[:plda_dim]
            self.uk = U[:, :plda_dim]
            self.qhat = torch.chain_matmul(self.uk.t(), Q, self.uk)

    def score_trials(self, left_vectors, right_vectors, plda_dim):
        self._compute_scoring_matrices(plda_dim)
        left_vectors = left_vectors.to(self.uk.device)
        right_vectors = right_vectors.to(self.uk.device)
        left_vectors = torch.matmul(left_vectors, self.uk)
        right_vectors  = torch.matmul(right_vectors, self.uk)
        score_h1 = torch.sum(torch.matmul(left_vectors, self.qhat) * left_vectors, 1)
        score_h2 = torch.sum(torch.matmul(right_vectors, self.qhat) * right_vectors, 1)
        score_h1h2 = 2 * torch.sum(left_vectors * self.l * right_vectors, 1)
        scores = score_h1h2 + score_h1 + score_h2
        return scores.cpu().numpy()

    def _score_all_vs_all(self, vectors1, vectors2):
        vectors1 = vectors1.to(self.uk.device)
        vectors2 = vectors2.to(self.uk.device)
        vectors1 = torch.matmul(vectors1, self.uk)
        vectors2  = torch.matmul(vectors2, self.uk)
        score_h1 = torch.sum(torch.matmul(vectors1, self.qhat) * vectors1, 1, keepdim=True)
        score_h2 = torch.sum(torch.matmul(vectors2, self.qhat) * vectors2, 1)
        score_h1h2 = 2 * torch.matmul(vectors1 * self.l, vectors2.t())
        scores = score_h1h2 + score_h1 + score_h2
        return scores.cpu().numpy()

    def score_all_vs_all(self, vectors1, vectors2, plda_dim):
        self._compute_scoring_matrices(plda_dim)
        return self._score_all_vs_all(vectors1, vectors2)

    def score_all_vs_all_chunked(self, vectors1, vectors2, plda_dim):
        n_scores = vectors1.size()[0] * vectors2.size()[0]
        if n_scores <= Settings().backend.max_all_vs_all_score_count:
            return self.score_all_vs_all(vectors1, vectors2, plda_dim)       
        print('More than {} all-vs-all pairs ({}) --> chunked PLDA scoring...'.format(Settings().backend.max_all_vs_all_score_count, n_scores))
        self._compute_scoring_matrices(plda_dim)
        n_chunks = math.ceil(n_scores / Settings().backend.max_all_vs_all_score_count)
        vecs = (vectors1, vectors2)
        larger = 1
        if vecs[0].size()[0] > vecs[1].size()[0]:
            larger = 0
        chunk_size = math.ceil(vecs[larger].size()[0] / n_chunks)
        scores = []
        i = 0
        c = 1
        while i < vecs[larger].size()[0]:
            print('Chunk {}/{} ...'.format(c, n_chunks))
            chunk_vecs = vecs[larger][i:i+chunk_size, :]
            i += chunk_size
            c += 1
            if larger == 0:
                scores.append(self._score_all_vs_all(chunk_vecs, vectors2))
            else:
                scores.append(self._score_all_vs_all(vectors1, chunk_vecs))
        if larger == 0:
            scores = np.vstack(scores)
        else:
            scores = np.hstack(scores)
        print('PLDA scoring done!')
        return scores


    def compress(self, vectors, plda_dim):
        self._compute_scoring_matrices(plda_dim)
        return torch.matmul(vectors, self.uk.to(vectors.device))

    def save(self, filename):
        print('Saving PLDA to file {}'.format(filename))
        np.savez(filename, St=self.St.cpu().numpy(), Sb=self.Sb.cpu().numpy())

    @classmethod
    def load(cls, filename, device):
        print('Loading PLDA from file {}'.format(filename))
        holder = np.load(filename)
        St, Sb = holder['St'], holder['Sb']
        return Plda(torch.from_numpy(St).to(device), torch.from_numpy(Sb).to(device))


def _compute_cov(data):
    data -= torch.mean(data, dim=0)
    cov = torch.matmul(data.t(), data) / (data.size()[0] - 1)
    return cov

def _compute_within_cov(data, class_boundaries):
    data = data.clone()
    for start, end in zip(class_boundaries[:-1], class_boundaries[1:]):
        data[start:end, :] -= data[start:end, :].mean(dim=0)        
    return _compute_cov(data)     

def _compute_between_cov(data, class_boundaries):
    Sb = torch.zeros(data.size()[1], data.size()[1], dtype=data.dtype, device=data.device)
    for start, end in zip(class_boundaries[:-1], class_boundaries[1:]):
        class_mean = data[start:end, :].mean(dim=0, keepdims=True)
        Sb += (end-start) * torch.matmul(class_mean.t(), class_mean)
    return Sb / data.size()[0]

def _rearrange_data(data, speaker_labels):
    print('Rearranging data for LDA/PLDA training...')
    index_dict = defaultdict(list)
    for index, label in enumerate(speaker_labels):
        index_dict[label].append(index)
    new_data = torch.zeros(*data.size(), dtype=data.dtype, device=data.device)
    class_boundaries = [0]
    counter = 0
    for key in index_dict:
        indices = index_dict[key]
        new_data[counter:counter + len(indices), :] = data[indices, :]
        counter += len(indices)
        class_boundaries.append(counter)
    return new_data, class_boundaries

def _orthogonalize_columns(matrix):
    matrix -= torch.mean(matrix, 1).unsqueeze(1)
    D, V = torch.svd(matrix)[1:]
    W = torch.matmul(V, torch.diag((1./(torch.sqrt(D) + 1e-10))))
    return torch.matmul(matrix, W)

def _arrange_data_by_counts(data, labels):
    spk2indices = defaultdict(list)
    for index, label in enumerate(labels):
        spk2indices[label].append(index)

    count2spks = defaultdict(list)
    for spk in spk2indices:
        count2spks[len(spk2indices[spk])].append(spk)

    data_list = []
    count_list = []
    for count in count2spks:
        count_list.append(count)
        count_indices = []
        for spk in count2spks[count]:
            count_indices.extend(spk2indices[spk])
        data_list.append(data[count_indices, :])

    return data_list, count_list