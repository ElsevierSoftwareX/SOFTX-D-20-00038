# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import time
import os
import sys

import torch
import numpy as np

from asvtorch.src.ivector.gmm import Gmm
from asvtorch.src.settings.settings import Settings
from asvtorch.src.utterances.utterance_list import UtteranceList
import asvtorch.src.ivector.statloader as statloader
import asvtorch.src.ivector.statloader_neural as statloader_neural
import asvtorch.src.misc.fileutils as fileutils

class IVectorExtractor():
    def __init__(self, t_matrix, ubm: Gmm, prior_offset, device):
        # When prior offset is zero, standard (non-augmented) i-vector formulation is used.
        self.covariance_type = Settings().ivector.covariance_type
        self.t_matrix = t_matrix.to(device)
        self.prior_offset = prior_offset.to(device)
        self.means = ubm.means.clone().to(device)
        self.covariances = ubm.covariances.clone().to(device)
        self.weights = ubm.weights.clone().to(device)
        self.covariances = self._constraint_covariances(self.covariances)
        self.inv_covariances = torch.inverse(self.covariances)
        self.n_components, self.ivec_dim, self.feat_dim = self.t_matrix.size()
        self.identity = torch.eye(self.ivec_dim, dtype=self.means.dtype, device=device).unsqueeze(0)
        self.bias_offset = None

    def _compute_posterior_means_and_covariances(self, n_all, f_all, batch_size, component_batches):
        covariances = torch.zeros(self.ivec_dim, batch_size, self.ivec_dim, dtype=self.means.dtype, device=self.t_matrix.device)
        means = torch.zeros(self.ivec_dim, batch_size, dtype=self.means.dtype, device=self.t_matrix.device)
        for bstart, bend in component_batches:
            n = n_all[:, bstart:bend]
            f = f_all[bstart:bend, :, :]
            sub_t = self.t_matrix[bstart:bend, :, :]
            sub_inv_covars = self.inv_covariances[bstart:bend, :, :]
            sub_tc = torch.bmm(sub_t, sub_inv_covars)
            tt = torch.bmm(sub_tc, torch.transpose(sub_t, 1, 2))
            tt.transpose_(0, 1)
            covariances += torch.matmul(n, tt)
            means = torch.addbmm(means, sub_tc, f)
        covariances.transpose_(0, 1)
        covariances += self.identity
        covariances = torch.inverse(covariances)
        means.t_()
        means[:, 0] += self.prior_offset
        means.unsqueeze_(2)
        means = torch.bmm(covariances, means)
        means = means.view((means.size()[:2]))
        return means, covariances

    def _get_component_batches(self, n_component_batches):
        cbatch_size = self.n_components // n_component_batches
        component_batches = []
        for cbatch_index in range(n_component_batches):
            bstart = cbatch_index * cbatch_size
            bend = (cbatch_index + 1) * cbatch_size
            component_batches.append((bstart, bend))
        return component_batches

    def _get_stat_loader(self, data, second_order):
        data_dims = (self.n_components, self.feat_dim)
        if hasattr(data, 'stats'):
            if self.prior_offset == 0:
                stat_loader = statloader_neural.get_stat_loader(data, self.means)
            else:
                stat_loader = statloader_neural.get_stat_loader(data, None)
        else:
            if self.prior_offset == 0:
                stat_loader = statloader.get_stat_loader(data, data_dims, second_order, self.means)
            else:  # Kaldi style i-vector (augmented form) --> No centering required
                stat_loader = statloader.get_stat_loader(data, data_dims, second_order, None)
        return stat_loader

    def get_updated_ubm(self, ubm, device):
        if self.prior_offset == 0:
            means = self.means.clone()
        else:
            means = self.t_matrix[:, 0, :] * self.prior_offset
        covariances = ubm.covariances.clone()
        weights = ubm.weights.clone()
        return Gmm(means, covariances, weights, device)
 
    def extract(self, utterance_list: UtteranceList):
        stat_loader = self._get_stat_loader(utterance_list, False)
        component_batches = self._get_component_batches(Settings().ivector.n_component_batches)
        print('Extracting i-vectors for {} utterances...'.format(len(utterance_list.utterances)))
        start_time = time.time()
        ivectors = torch.zeros(len(utterance_list.utterances), self.ivec_dim, dtype=self.means.dtype, device=self.t_matrix.device)
        counter = 0
        for batch_index, batch in enumerate(stat_loader):
            n_all, f_all = batch
            batch_size = n_all.size()[0]
            print('{:.0f} seconds elapsed, Batch {}/{}: utterance count = {}'.format(time.time() - start_time, batch_index+1, stat_loader.__len__(), batch_size))
            n_all = n_all.to(self.t_matrix.device)
            f_all = f_all.to(self.t_matrix.device)
            means = self._compute_posterior_means_and_covariances(n_all, f_all, batch_size, component_batches)[0]
            ivectors[counter:counter+batch_size, :] = means
            counter += batch_size
        ivectors[:, 0] -= self.prior_offset
        print('I-vector extraction completed in {:.0f} seconds.'.format(time.time() - start_time))
        utterance_list.embeddings = ivectors

    def extract_with_covariances(self, utterance_list: UtteranceList):
        stat_loader = self._get_stat_loader(utterance_list, False)
        component_batches = self._get_component_batches(Settings().ivector.n_component_batches)
        print('Extracting i-vectors for {} utterances...'.format(len(utterance_list.utterances)))
        start_time = time.time()
        ivectors = torch.zeros(len(utterance_list.utterances), self.ivec_dim, dtype=self.means.dtype, device=self.t_matrix.device)
        covariances = torch.zeros(len(utterance_list.utterances), self.ivec_dim, self.ivec_dim, dtype=self.means.dtype, device=self.t_matrix.device)
        counter = 0
        for batch_index, batch in enumerate(stat_loader):
            n_all, f_all = batch
            batch_size = n_all.size()[0]
            print('{:.0f} seconds elapsed, Batch {}/{}: utterance count = {}'.format(time.time() - start_time, batch_index+1, stat_loader.__len__(), batch_size))
            n_all = n_all.to(self.t_matrix.device)
            f_all = f_all.to(self.t_matrix.device)
            means, covs = self._compute_posterior_means_and_covariances(n_all, f_all, batch_size, component_batches)
            ivectors[counter:counter+batch_size, :] = means
            covariances[counter:counter+batch_size, :, :] = covs
            counter += batch_size
        ivectors[:, 0] -= self.prior_offset
        print('I-vector extraction completed in {:.0f} seconds.'.format(time.time() - start_time))
        utterance_list.embeddings = ivectors
        utterance_list.covariances = covariances

    def train(self, utterance_list: UtteranceList, resume=0):
        if resume < 0:
            resume = 0
        elif resume > 0:
            print('Resuming i-vector extractor training from iteration {}...'.format(resume))
            extractor = IVectorExtractor.from_npz(resume, self.t_matrix.device)
            self.t_matrix = extractor.t_matrix
            self.means = extractor.means
            self.inv_covariances = extractor.inv_covariances
            self.prior_offset = extractor.prior_offset

        print('Training i-vector extractor ({} iterations)...'.format(Settings().ivector.n_iterations))

        n_utts = len(utterance_list.utterances)
        component_batches = self._get_component_batches(Settings().ivector.n_component_batches)
        
        print('Allocating memory for accumulators...')
        z = torch.zeros(self.n_components, dtype=self.means.dtype, device=self.t_matrix.device)
        S = torch.zeros(self.n_components, self.feat_dim, self.feat_dim, dtype=self.means.dtype, device=self.t_matrix.device)
        Y = torch.zeros(self.n_components, self.feat_dim, self.ivec_dim, dtype=self.means.dtype, device=self.t_matrix.device)
        R = torch.zeros(self.n_components, self.ivec_dim, self.ivec_dim, dtype=self.means.dtype, device=self.t_matrix.device)  # The biggest memory consumer!
        h = torch.zeros(self.ivec_dim, dtype=self.means.dtype, device=self.t_matrix.device)
        H = torch.zeros(self.ivec_dim, self.ivec_dim, dtype=self.means.dtype, device=self.t_matrix.device)

        if hasattr(utterance_list, 'stats'):
            S = utterance_list.stats.second_sum.to(self.t_matrix.device)
            accumulate_2nd_stats = False

        iteration_times = []
        start_time = time.time()
        for iteration in range(1, Settings().ivector.n_iterations + 1):
            iter_start_time = time.time()

            print('Initializing statistics loader...')
            if not hasattr(utterance_list, 'stats'):
                accumulate_2nd_stats = Settings().ivector.update_covariances and iteration == 1 # 2nd order stats need to be accumulated only once
            stat_loader = self._get_stat_loader(utterance_list, accumulate_2nd_stats)

            print('Iterating over batches of utterances...')
            for batch_index, batch in enumerate(stat_loader):            

                if accumulate_2nd_stats:
                    n_all, f_all, s_batch_sum = batch
                    S += s_batch_sum.to(self.t_matrix.device)  
                else:
                    n_all, f_all = batch

                batch_size = n_all.size()[0]
                print('Iteration {} ({:.0f} seconds), Batch {}/{}: utterance count = {}'.format(iteration + resume, time.time() - iter_start_time, batch_index+1, stat_loader.__len__(), batch_size))

                n_all = n_all.to(self.t_matrix.device)
                f_all = f_all.to(self.t_matrix.device)
                if iteration == 1:  # Need to be accumulated only once
                    z += torch.sum(n_all, dim=0)

                means, covariances = self._compute_posterior_means_and_covariances(n_all, f_all, batch_size, component_batches)

                # Accumulating...
                h += torch.sum(means, dim=0)
                yy = torch.baddbmm(covariances, means.unsqueeze(2), means.unsqueeze(1))
                H += torch.sum(yy, dim=0)
                yy = yy.permute((1, 2, 0))
                for bstart, bend in component_batches: # Batching over components saves GPU memory
                    n = n_all[:, bstart:bend]
                    f = f_all[bstart:bend, :, :]
                    Y[bstart:bend, :, :] += torch.matmul(f, means)
                    R[bstart:bend, :, :] += torch.matmul(yy, n).permute((2, 0, 1))

            self.weights = z / torch.sum(z)
            h = h / n_utts
            H = H / n_utts
            H = H - torch.ger(h, h)

            # Updating:
            if Settings().ivector.update_projections: self._update_projections(Y, R, component_batches)
            if Settings().ivector.update_covariances: self._update_covariances(Y, z, S, component_batches)
            if Settings().ivector.minimum_divergence: self._minimum_divergence_whitening(h, H, component_batches)
            if Settings().ivector.update_means:       self._minimum_divergence_centering(h, component_batches)

            print('Zeroing accumulators...')
            Y.zero_()
            R.zero_()
            h.zero_()
            H.zero_()

            self.save_npz(iteration + resume)
            print('Iteration {} completed in {:.1f} seconds!'.format(iteration + resume, time.time() - iter_start_time))

        print('Training completed in {:.0f} seconds.'.format(time.time() - start_time))
        return iteration_times

    def _update_projections(self, Y, R, component_batches):
        print('Updating projections...')
        for bstart, bend in component_batches:
            self.t_matrix[bstart:bend, :, :] = torch.cholesky_solve(Y[bstart:bend, :, :].transpose(1, 2), torch.cholesky(R[bstart:bend, :, :], upper=True), upper=True)

    def _update_covariances(self, Y, z, S, component_batches):
        print('Updating covariances...')
        for bstart, bend in component_batches:
            crossterm = torch.matmul(Y[bstart:bend, :, :], self.t_matrix[bstart:bend, :, :])
            crossterm = crossterm + crossterm.transpose(1, 2)
            self.inv_covariances[bstart:bend, :, :] = S[bstart:bend, :, :] - 0.5 * crossterm

        var_floor = torch.sum(self.inv_covariances, dim=0)
        var_floor *= 0.1 / torch.sum(z)

        self.inv_covariances = self.inv_covariances / z.unsqueeze(1).unsqueeze(1)
        self.inv_covariances = self._constraint_covariances(self.inv_covariances)
        if self.covariance_type.endswith('full'):
            self._apply_floor_(self.inv_covariances, var_floor, component_batches)
        self.covariances = (self.inv_covariances).clone()
        self.inv_covariances = torch.inverse(self.inv_covariances)

    def _constraint_covariances(self, covariance_matrices):
        #print(covariance_matrices.diagonal(dim1=-2, dim2=-1))
        if self.covariance_type.startswith('shared'):
            covariance_matrices = torch.sum(covariance_matrices * self.weights.unsqueeze(1).unsqueeze(1), dim=0).expand(*covariance_matrices.size())
            #covariance_matrices = torch.mean(covariance_matrices, dim=0).expand(*covariance_matrices.size())
            #print(covariance_matrices.diagonal(dim1=-2, dim2=-1))
        if self.covariance_type.endswith('spherical'):
            covariance_matrices = torch.mean(covariance_matrices.diagonal(dim1=-2, dim2=-1), dim=1)
            covariance_matrices = torch.diag_embed(torch.ones(self.means.size()[1], device=Settings().computing.device).unsqueeze(0) * covariance_matrices.unsqueeze(1))
        elif self.covariance_type.endswith('diagonal'):
            covariance_matrices = covariance_matrices.diagonal(dim1=-2, dim2=-1)
            covariance_matrices = torch.diag_embed(covariance_matrices)
            #print(covariance_matrices.diagonal(dim1=-2, dim2=-1))
        elif not self.covariance_type.endswith('full'):
            sys.exit('Covariance type {} is not valid'.format(self.covariance_type))
        return covariance_matrices


    def _apply_floor_(self, A, B, component_batches):
        #B = self._apply_floor_scalar(B, self._max_abs_eig(B) * 1e-4)[0]  # To prevent Cholesky from failing
        L = torch.cholesky(B)
        L_inv = torch.inverse(L)
        num_floored = 0
        batch_size = component_batches[0][1] - component_batches[0][0]
        l = torch.zeros(batch_size, self.feat_dim, dtype=self.means.dtype, device=self.t_matrix.device)
        U = torch.zeros(batch_size, self.feat_dim, self.feat_dim, dtype=self.means.dtype, device=self.t_matrix.device) 
        for bstart, bend in component_batches:
            D = L_inv.matmul(A[bstart:bend, :, :]).matmul(L_inv.t())   
            for c in range(batch_size):
                l[c, :], U[c, :, :] = torch.symeig(D[c, :, :], eigenvectors=True)
            num_floored += torch.sum(l < 1).item()
            l = torch.clamp(l, min=1)
            D = U.matmul(l.unsqueeze(2) * U.transpose(1,2))
            A[bstart:bend, :, :] = L.matmul(D.transpose(1, 2)).matmul(L.t())
        print('Floored {:.1%} of the eigenvalues...'.format(num_floored / (self.n_components * self.feat_dim)))

    def _max_abs_eig(self, A):
        l = torch.symeig(A)[0]
        return torch.max(torch.abs(l))

    def _apply_floor_scalar(self, A, b):
        l, U = torch.symeig(A, eigenvectors=True)
        num_floored = torch.sum(l < b).item()
        l = torch.clamp(l, min=b)
        A = torch.matmul(U, l.unsqueeze(1) * U.t())
        return A, num_floored

    def _minimum_divergence_whitening(self, h, H, component_batches):
        print('Minimum divergence re-estimation...')
        l, U = torch.symeig(H, eigenvectors=True)
        l = torch.clamp(l, min=1e-7)
        P1 = torch.rsqrt(l) * U  # transposed
        torch.matmul(h, P1, out=h)  # In place operation, so that the result is available for update_means()
        if self.prior_offset != 0:  # Augmented formulation
            self.prior_offset = h[0]
            print('Prior offset: {}'.format(self.prior_offset))
        P1 = torch.inverse(P1)
        #P1 = torch.pinverse(P1, 0.0001)
        for bstart, bend in component_batches:
            self.t_matrix[bstart:bend, :, :] = P1.matmul(self.t_matrix[bstart:bend, :, :])

    def _minimum_divergence_centering(self, h, component_batches):
        if self.prior_offset == 0:
            self.means += torch.sum(self.t_matrix * h.unsqueeze(0).unsqueeze(2), dim=1)
        else:  # Augmented formulation uses the Householder transformation
            x = h / h.norm()
            alpha = torch.rsqrt(2 * (1 - x[0]))
            print('Alpha: {}'.format(alpha))
            a = x * alpha
            a[0] -= alpha
            P2 = self.identity.squeeze(0) - 2 * torch.ger(a, a)
            self.prior_offset = torch.dot(h, P2[:, 0].squeeze())
            print('Prior offset: {}'.format(self.prior_offset))
            P2 = torch.inverse(P2)
            #P2 = torch.pinverse(P2, 0.0001)
            for bstart, bend in component_batches:
                self.t_matrix[bstart:bend, :, :] = P2.matmul(self.t_matrix[bstart:bend, :, :])

    def save_npz(self, iteration: int):
        extractor_folder = fileutils.get_ivector_extractor_folder()
        filename = os.path.join(extractor_folder, 'iter.{}.npz'.format(iteration))
        np.savez(filename, t_matrix=self.t_matrix.cpu().numpy(), means=self.means.cpu().numpy(), covariances=self.covariances.cpu().numpy(), weights=self.weights.cpu().numpy(), prior_offset=self.prior_offset.cpu().numpy())
        print('I-vector extractor saved to {}'.format(filename))

    @classmethod
    def random_init(cls, ubm: Gmm, device, seed=0):
        torch.manual_seed(seed)
        t_matrix = torch.randn(ubm.covariances.size()[0], Settings().ivector.ivec_dim, ubm.covariances.size()[1], dtype=ubm.means.dtype)
        if Settings().ivector.ivec_type == 'augmented':
            prior_offset = torch.tensor([float(Settings().ivector.initial_prior_offset)], dtype=ubm.means.dtype) # pylint: disable=not-callable
            t_matrix[:, 0, :] = ubm.means.clone().cpu() / prior_offset
        else:
            prior_offset = torch.tensor([float(0)], dtype=ubm.means.dtype)  # pylint: disable=not-callable
        return IVectorExtractor(t_matrix, ubm, prior_offset, device)

    @classmethod
    def from_npz(cls, iteration: int, device):
        extractor_folder = fileutils.get_ivector_extractor_folder()
        filename = os.path.join(extractor_folder, 'iter.{}.npz'.format(iteration))
        data = np.load(filename)
        t_matrix = torch.from_numpy(data['t_matrix'])
        means = torch.from_numpy(data['means'])
        covariances = torch.from_numpy(data['covariances'])
        weights = torch.from_numpy(data['weights'])
        prior_offset = torch.from_numpy(data['prior_offset'])
        gmm = Gmm(means, covariances, weights, device='cpu')
        return IVectorExtractor(t_matrix, gmm, prior_offset, device)
