# Copyright 2020 Ville Vestman
#           2020 Kong Aik Lee
# This file is licensed under the MIT license (see LICENSE.txt).

from functools import partial
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from asvtorch.src.settings.settings import Settings


class CnnLayer(nn.Module):
    def __init__(self, D_in: int, D_out: int, filter_reach: int, ser: int = 0):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(D_out, momentum=Settings().network.bn_momentum, affine=Settings().network.bn_affine)
        self.activation = get_activation_function()
        self.cnn = torch.nn.Conv1d(D_in, D_out, filter_reach*2+1, padding=filter_reach, padding_mode=Settings().network.cnn_padding_mode)
        self.ser = ser
        if ser > 0:
           self.se = SeLayer(D_out, ser)

    def forward(self, x):
        x = self.cnn(x)
        if self.ser > 0:
            x = self.se(x)
        return self.batchnorm(self.activation(x))

class PlainCnnLayer(nn.Module):
    def __init__(self, D_in: int, D_out: int, filter_reach: int, ser: int = 0):
        super().__init__()
        self.cnn = torch.nn.Conv1d(D_in, D_out, filter_reach*2+1, padding=filter_reach, padding_mode=Settings().network.cnn_padding_mode)
        self.ser = ser
        if ser > 0:
           self.se = SeLayer(D_out, ser)

    def forward(self, x):
        x = self.cnn(x)
        if self.ser > 0:
            x = self.se(x)
        return x


# For utterance-level layers
class LinearReluBatchNormLayer(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.activation = get_activation_function()
        self.linear = torch.nn.Linear(D_in, D_out)
        self.batchnorm = nn.BatchNorm1d(D_out, momentum=Settings().network.bn_momentum, affine=Settings().network.bn_affine)

    def forward(self, x):
        return self.batchnorm(self.activation(self.linear(x)))


class LinearBatchNormLayer(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.linear = torch.nn.Linear(D_in, D_out)
        self.batchnorm = nn.BatchNorm1d(D_out, momentum=Settings().network.bn_momentum, affine=Settings().network.bn_affine)

    def forward(self, x):
        return self.batchnorm(self.linear(x))

        
class ResCnnLayer(nn.Module):
    def __init__(self, D: int, filter_reach: int, ser: int = 0):
        super().__init__()
        self.activation = get_activation_function()

        # Fully connected layer as 1-by-1 CNN
        self.fc = torch.nn.Conv1d(D, D, 1, bias=True, padding=0)
        self.cnn = torch.nn.Conv1d(D, D, filter_reach*2+1, padding=filter_reach, padding_mode=Settings().network.cnn_padding_mode)
        self.fc_norm = nn.BatchNorm1d(D, momentum=Settings().network.bn_momentum, affine=Settings().network.bn_affine)
        self.cnn_norm = nn.BatchNorm1d(D, momentum=Settings().network.bn_momentum, affine=Settings().network.bn_affine)
        self.ser = ser
        if ser > 0:
           self.se = SeLayer(D, ser)

    def forward(self, x):
        y = self.fc_norm(self.activation(self.fc(x)))
        y = self.cnn(y)
        if self.ser > 0:
            y = self.se(y) 
        return self.cnn_norm(self.activation(y + x))


# Squeeze-and-excite module
class SeLayer(nn.Module):
    def __init__(self, D: int, ser: int):
        super().__init__()
        self.pooling_layer = MeanStdPoolingLayer()
        self.linear_relu_bn = LinearReluBatchNormLayer(D * 2, int(D / ser))
        self.fc = torch.nn.Linear(int(D / ser), D, bias=True)
        
    def forward(self, x):
        W = self.pooling_layer(x)
        W = self.fc(self.linear_relu_bn(W))
        W = torch.sigmoid(W)
        W = torch.unsqueeze(W, dim=2)
        return x * W


class MeanStdPoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        m = torch.mean(x, dim=2)
        sigma = torch.sqrt(torch.clamp(torch.mean(x ** 2, dim=2) - m ** 2, min=1e-6))
        return torch.cat((m, sigma), 1)



class ClusteringLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.centroids = nn.Parameter(torch.rand(Settings().network.n_clusters + Settings().network.n_ghost_clusters, input_dim))
        if Settings().network.cluster_assignment_mode == 'net_vlad':
            self.linear = nn.Linear(input_dim, Settings().network.n_clusters + Settings().network.n_ghost_clusters)
            self.linear.weight = nn.Parameter((2.0 * Settings().network.net_vlad_alpha * self.centroids))
        elif Settings().network.cluster_assignment_mode == 'lde':
            if Settings().network.lde_covariance_type == 'spherical':
                self.precision = nn.Parameter(torch.ones(Settings().network.n_clusters))  
                self.einsum_string = 'utcf,utcf,c->utc'
            elif Settings().network.lde_covariance_type == 'shared_spherical':
                self.precision = nn.Parameter(torch.ones(1))
                self.einsum_string = 'utcf,utcf,a->utc'
            elif Settings().network.lde_covariance_type == 'diagonal':          
                self.precision = nn.Parameter(torch.ones(Settings().network.n_clusters, input_dim))
                self.einsum_string = 'utcf,utcf,cf->utc'
            elif Settings().network.lde_covariance_type == 'shared_diagonal':
                self.precision = nn.Parameter(torch.ones(input_dim))
                self.einsum_string = 'utcf,utcf,f->utc'
            else:
                # self.scaling_factors = nn.Parameter(torch.diag_embed(torch.ones(input_dim))) * 0.5 shared full
                sys.exit('LDE cov type {} not implemented'.format(Settings().network.lde_covariance_type))
            self.bias_terms = nn.Parameter(torch.randn(Settings().network.n_clusters))
        self.centroids = nn.Parameter(self.centroids[:(-Settings().network.n_ghost_clusters or None), :]) # Remove the 'ghost' clusters
       

    def forward(self, x, forward_mode='train'):
        n_utts, feat_dim, duration = x.size()
        x = x.transpose(1, 2)
        residuals = x.unsqueeze(2).expand(-1, -1, Settings().network.n_clusters, -1) - self.centroids.expand(n_utts, duration, -1, -1)
        if Settings().network.cluster_assignment_mode == 'net_vlad':
            soft_assignments = self.linear(x)
            soft_assignments = F.softmax(soft_assignments, dim=2)
            soft_assignments = soft_assignments[:, :, :(-Settings().network.n_ghost_clusters or None)]
        elif Settings().network.cluster_assignment_mode == 'lde':
            res2 = residuals.clone()  # to allow gradient computation
            scaled_neg_distances = -0.5 * torch.einsum(self.einsum_string, res2, res2, self.precision) + self.bias_terms.unsqueeze(0).unsqueeze(0)
            soft_assignments = F.softmax(scaled_neg_distances, dim=2)  # Softmax over clusters        

        # Stats are for neural i-vector:
        if forward_mode == 'extract_training_stats':
            zeroth = soft_assignments.sum(dim=1)
            first = torch.bmm(soft_assignments.transpose(1, 2), x)
            second_sum = torch.einsum('abc,abd,abe->cde', soft_assignments, x, x)  # Handy command :)
            return zeroth, first, second_sum
        if forward_mode == 'extract_testing_stats':
            zeroth = soft_assignments.sum(dim=1)
            first = torch.bmm(soft_assignments.transpose(1, 2), x)
            return zeroth, first

        residuals *= soft_assignments.unsqueeze(3)
        component_vectors = residuals.sum(dim=1)  # Sum over time

        if Settings().network.supervector_mode == 'net_vlad':
            component_vectors = F.normalize(component_vectors, p=2, dim=2)  # intra-normalization
        elif Settings().network.supervector_mode == 'lde':
            component_vectors /= soft_assignments.sum(dim=1).unsqueeze(2)

        supervector = component_vectors.view(n_utts, -1)
        if Settings().network.normalize_supervector:
            supervector = F.normalize(supervector, p=2, dim=1)  # L2 normalize

        return supervector


def get_activation_function():
    if Settings().network.activation == 'relu':
        return F.relu
    elif Settings().network.activation == 'lrelu':
        return partial(F.leaky_relu, negative_slope=Settings().network.lrelu_slope)
    elif Settings().network.activation == 'selu':
        return F.selu
