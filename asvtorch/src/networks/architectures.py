# Copyright 2020 Ville Vestman
#           2020 Kong Aik Lee
# This file is licensed under the MIT license (see LICENSE.txt).

import torch
import torch.nn as nn

from asvtorch.src.networks.modules import *
from asvtorch.src.settings.settings import Settings

class BaseNet(nn.Module):
    def __init__(self, feat_dim, n_speakers):
        super().__init__()
        self.feat_dim_param = torch.nn.Parameter(torch.LongTensor([feat_dim]), requires_grad=False)
        self.n_speakers_param = torch.nn.Parameter(torch.LongTensor([n_speakers]), requires_grad=False)
        self.training_loss = torch.nn.Parameter(torch.Tensor([torch.finfo().max]), requires_grad=False)
        self.consecutive_lr_updates = torch.nn.Parameter(torch.LongTensor([0]), requires_grad=False)

class StandardNetTemplate(BaseNet):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)
        self.feat_dim = feat_dim
        self.n_speakers = n_speakers
        self.dim_featlayer = Settings().network.frame_layer_size
        self.dim_statlayer = Settings().network.stat_size
        self.dim_uttlayer = Settings().network.embedding_size
        self.tdnn_layers = nn.ModuleList()
        self.utterance_layers = nn.ModuleList()
        self.pooling_layer, self.pooling_output_dim = _init_pooling_layer()

    def forward(self, x, forward_mode='train'):
        for layer in self.tdnn_layers:
            x = layer(x)

        # To extract "neural features"
        if forward_mode == 'extract_features':
            return x

        # To extract "neural stats" for neural i-vector
        if forward_mode in ('extract_training_stats', 'extract_testing_stats'):
            return self.pooling_layer(x, forward_mode)

        x = self.pooling_layer(x)

        if forward_mode == 'extract_embeddings': # Embedding extraction
            return self.utterance_layers[0].linear(x)
            #return self.utterance_layers[0].activation(self.utterance_layers[0].linear(x))

        for layer in self.utterance_layers:
            x = layer(x)

        return x

class StandardNet(StandardNetTemplate):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)

        self.tdnn_layers.append(CnnLayer(self.feat_dim, self.dim_featlayer, 2))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 2))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 3))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 0))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_statlayer, 0))

        # Pooling layer

        self.utterance_layers.append(LinearBatchNormLayer(self.pooling_output_dim, self.dim_uttlayer))
        #self.utterance_layers.append(LinearReluBatchNormLayer(self.dim_uttlayer, self.dim_uttlayer))
        self.utterance_layers.append(nn.Linear(self.dim_uttlayer, self.n_speakers))


class StandardSeNet(StandardNetTemplate):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)

        ser = Settings().network.ser

        self.tdnn_layers.append(CnnLayer(self.feat_dim, self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 3, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 0, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_statlayer, 0, ser))

        # Pooling layer

        self.utterance_layers.append(LinearBatchNormLayer(self.pooling_output_dim, self.dim_uttlayer))
        #self.utterance_layers.append(LinearReluBatchNormLayer(self.dim_uttlayer, self.dim_uttlayer))
        self.utterance_layers.append(nn.Linear(self.dim_uttlayer, self.n_speakers))



class StandardResSeNet(StandardNetTemplate):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)

        ser = Settings().network.ser

        self.tdnn_layers.append(CnnLayer(self.feat_dim, self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 3, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 0, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_statlayer, 0, ser))

        # Pooling layer

        self.utterance_layers.append(LinearBatchNormLayer(self.pooling_output_dim, self.dim_uttlayer))
        #self.utterance_layers.append(LinearReluBatchNormLayer(self.dim_uttlayer, self.dim_uttlayer))
        self.utterance_layers.append(nn.Linear(self.dim_uttlayer, self.n_speakers))


class LargeResSeNet(StandardNetTemplate):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)

        ser = Settings().network.ser

        self.tdnn_layers.append(CnnLayer(self.feat_dim, self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 3, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 4, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 1, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 3, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 4, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 0, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_statlayer, 0, ser))

        # Pooling layer

        self.utterance_layers.append(LinearBatchNormLayer(self.pooling_output_dim, self.dim_uttlayer))
        #self.utterance_layers.append(LinearReluBatchNormLayer(self.dim_uttlayer, self.dim_uttlayer))
        self.utterance_layers.append(nn.Linear(self.dim_uttlayer, self.n_speakers))



def _init_pooling_layer():
    if Settings().network.pooling_layer_type == 'clustering':
        pooling_layer = ClusteringLayer(Settings().network.stat_size)
        pooling_output_dim = Settings().network.n_clusters * Settings().network.stat_size
    elif Settings().network.pooling_layer_type == 'default':
        pooling_layer = MeanStdPoolingLayer()
        pooling_output_dim = Settings().network.stat_size * 2
    return pooling_layer, pooling_output_dim
