# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import numpy as np
import torch
import kaldi.util.io as kio
from kaldi.gmm import FullGmm as KaldiFullGmm
from kaldi.matrix import Matrix as KaldiMatrix

import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.utterances.utterance_list import UtteranceList

class Gmm():
    def __init__(self, means, covariances, weights, device=torch.device("cpu")):
        self.means = means.to(device)
        self.covariances = covariances.to(device)
        self.weights = weights.to(device)
        # Preparation for posterior computation:
        const = torch.tensor([-0.5 * self.means.size()[1] * np.log(2 * np.pi)], dtype=means.dtype, device=self.means.device)  # pylint: disable=not-callable
        self._inv_covariances = torch.inverse(self.covariances)
        self._component_constants = torch.zeros(self.weights.numel(), dtype=means.dtype, device=self.means.device)
        for i in range(self.weights.numel()):
            self._component_constants[i] = -0.5 * torch.logdet(self.covariances[i, :, :]) + const + torch.log(self.weights[i])

    def to_device(self, device):
        return Gmm(self.means, self.covariances, self.weights, device=device)

    def compute_posteriors_top_select(self, frames, top_indices):
        logprob = torch.zeros(top_indices.size(), device=self.means.device)
        for i in range(self.weights.numel()):
            indices_of_component = (top_indices == i)
            frame_selection = torch.any(indices_of_component, 0)
            post_index = torch.argmax(indices_of_component.int(), 0)[frame_selection]
            centered_frames = frames[frame_selection, :] - self.means[i, :]        
            logprob[post_index, frame_selection] = self._component_constants[i] - 0.5 * torch.sum(torch.mm(centered_frames, self._inv_covariances[i, :, :]) * centered_frames, 1)
        llk = torch.logsumexp(logprob, dim=0)
        return torch.exp(logprob - llk)

    def compute_posteriors(self, frames):
        logprob = torch.zeros(self.weights.numel(), frames.size()[0], device=self.means.device)
        for i in range(self.weights.numel()):
            centered_frames = frames - self.means[i, :]
            logprob[i, :] = self._component_constants[i] - 0.5 * torch.sum(torch.mm(centered_frames, self._inv_covariances[i, :, :]) * centered_frames, 1)
        llk = torch.logsumexp(logprob, dim=0)
        return torch.exp(logprob - llk)

    def compute_llk(self, frames, means):
        logprob = torch.zeros(self.weights.numel(), frames.size()[0], device=self.means.device)
        for i in range(self.weights.numel()):
            centered_frames = frames - means[i, :]
            logprob[i, :] = self._component_constants[i] - 0.5 * torch.sum(torch.mm(centered_frames, self._inv_covariances[i, :, :]) * centered_frames, 1)
        return torch.logsumexp(logprob, dim=0)

    def save_npz(self, filename):
        np.savez(filename, weights=self.weights.cpu().numpy(), means=self.means.cpu().numpy(), covariances=self.covariances.cpu().numpy())
        print('GMM saved to {}'.format(fileutils.ensure_ext(filename, 'npz')))

    @classmethod
    def from_npz(cls, filename, device):
        data = np.load(fileutils.ensure_ext(filename, 'npz'))
        weights = torch.from_numpy(data['weights'])
        means = torch.from_numpy(data['means'])
        covariances = torch.from_numpy(data['covariances'])
        return Gmm(means, covariances, weights, device)

    @classmethod
    def from_kaldi(cls, filename, device):
        ubm = KaldiFullGmm()
        with kio.xopen(filename) as f:
            ubm.read(f.stream(), f.binary)
        means = torch.from_numpy(ubm.get_means().numpy())
        weights = torch.from_numpy(ubm.weights().numpy())
        n_components = weights.numel()
        feat_dim = means.size()[1]
        covariances = torch.zeros([n_components, feat_dim, feat_dim], device='cpu', dtype=torch.float32)
        for index, kaldicovar in enumerate(ubm.get_covars()):
            covariances[index, :, :] = torch.from_numpy(KaldiMatrix(kaldicovar).numpy())
        return Gmm(means, covariances, weights, device=device)

    @classmethod
    def from_stats(cls, utterance_list: UtteranceList, means=None, device=torch.device("cpu")):
        counts = torch.sum(utterance_list.stats.zeroth, dim=0)
        weights = counts / torch.sum(counts)
        if means is None:
            means = torch.sum(utterance_list.stats.first, dim=0) / counts.unsqueeze(1)
        covariances = utterance_list.stats.second_sum / counts.unsqueeze(1).unsqueeze(2) - torch.bmm(means.unsqueeze(2), means.unsqueeze(1))
        return Gmm(means, covariances, weights, device=device)




class DiagGmm():
    def __init__(self, means, covariances, weights, device=torch.device("cpu")):
        self.means = means.to(device)
        self.covariances = covariances.to(device)
        self.weights = weights.to(device)
        # Preparation for posterior computation:
        const = torch.Tensor([self.means.size()[1] * np.log(2 * np.pi)]).to(self.means.device)
        self.posterior_constant = torch.sum(self.means * self.means / self.covariances, 1) + torch.sum(torch.log(self.covariances), 1) + const
        self.posterior_constant = self.posterior_constant.unsqueeze(1)
        self.precisions = (1 / self.covariances)
        self.mean_pres = (self.means / self.covariances)

    def compute_posteriors(self, frames):
        logprob = torch.mm(self.precisions, (frames * frames).t()) - 2 * torch.mm(self.mean_pres, frames.t())
        logprob = -0.5 * (logprob + self.posterior_constant)
        logprob = logprob + torch.log(self.weights.unsqueeze(1))
        llk = torch.logsumexp(logprob, 0)
        return torch.exp(logprob - llk)

    def compute_llk(self, frames):
        logprob = torch.mm(self.precisions, (frames * frames).t()) - 2 * torch.mm(self.mean_pres, frames.t())
        logprob = -0.5 * (logprob + self.posterior_constant)
        logprob = logprob + torch.log(self.weights.unsqueeze(1))
        return torch.logsumexp(logprob, 0)

    def map_adapt(self, utterance_list: UtteranceList, relevance_factor: float):
        print('Adapting to obtain {} GMMs...'.format(len(utterance_list.utterances)))
        adapted_gmms = []
        for index in range(utterance_list.stats.zeroth.size()[0]):
            alpha = utterance_list.stats.zeroth[index] / (utterance_list.stats.zeroth[index] + relevance_factor)
            ml_means = utterance_list.stats.first[index] / utterance_list.stats.zeroth[index].unsqueeze(1)
            ml_means = ml_means.to(self.means.device)
            alpha = alpha.to(self.means.device)
            #print(utterance_list.stats.zeroth[index] / torch.sum(utterance_list.stats.zeroth[index]))
            new_means = alpha.unsqueeze(1) * ml_means + (1 - alpha).unsqueeze(1) * self.means
            if torch.sum(torch.isnan(new_means)) > 0:
                print('NaN error, TODO: implement special case')
            #adapted_means[index, :, :] = new_means
            adapted_gmms.append(DiagGmm(new_means, self.covariances, self.weights, device=self.means.device))
        utterance_list.adapted_gmms = adapted_gmms
        print('Adaptation completed!')

    def score_all_vs_all(self, enroll_gmms, test_features):
        count = 0
        max_frames_per_chunk = 500000
        scores = torch.zeros(len(enroll_gmms), len(test_features))
        while count < len(test_features):
            feature_list = []
            break_points = []
            frame_count = 0
            start_utt = count
            while count < len(test_features) and frame_count + test_features[count].size()[0] < max_frames_per_chunk:
                break_points.append((frame_count, frame_count + test_features[count].size()[0]))
                frame_count += test_features[count].size()[0]
                feature_list.append(test_features[count])
                #print(test_features[count].size())
                count += 1             
            features = torch.cat(feature_list, dim=0)
            features = features.to(self.means.device)
            print('Scoring test utterances {} - {}'.format(start_utt + 1, count))
            ubm_llks = self.compute_llk(features)
            gmm_llks = torch.zeros(len(enroll_gmms), ubm_llks.numel(), device=self.means.device)
            for enroll_index in range(len(enroll_gmms)):
                gmm_llks[enroll_index, :] = enroll_gmms[enroll_index].compute_llk(features)
            frame_scores = (gmm_llks - ubm_llks.unsqueeze(0)).cpu()
            for index, (start, end) in enumerate(break_points):
                scores[:, start_utt + index] = torch.mean(frame_scores[:, start:end], dim=1)
                #print(torch.mean(frame_scores[:, start:end], dim=1))
        
        #print(scores)
        return scores

    @classmethod
    def from_full_gmm(cls, full_gmm, device):
        means = full_gmm.means.clone()
        weights = full_gmm.weights.clone()
        covariances = torch.zeros(means.size(), device=full_gmm.covariances.device, dtype=full_gmm.covariances.dtype)
        for index in range(weights.numel()):
            covariances[index, :] = full_gmm.covariances[index, :, :].diag()
        return DiagGmm(means, covariances, weights, device=device)
           