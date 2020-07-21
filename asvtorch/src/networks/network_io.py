# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
import importlib

import torch

from asvtorch.src.settings.settings import Settings
import asvtorch.src.misc.fileutils as fileutils

def load_network(epoch: int, device):
    model_filepath = os.path.join(fileutils.get_network_folder(), 'epoch.{}.pt'.format(epoch))
    loaded_states = torch.load(model_filepath, map_location=device)
    state_dict = loaded_states['model_state_dict']
    key1 = 'feat_dim_param'
    key2 = 'n_speakers_param'
    feat_dim = state_dict[key1].item()
    n_speakers = state_dict[key2].item()
    net = initialize_net(feat_dim, n_speakers)
    net.to(device)
    net.load_state_dict(state_dict)
    return net

def save_state(filename, epoch, net, optimizer):
    model_dict = {'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    filename = fileutils.ensure_ext('{}.{}'.format(fileutils.remove_ext(filename, '.pt'), epoch), '.pt')
    torch.save(model_dict, filename)
    print('x-vector extractor model saved to: {}'.format(filename))

def load_state(filename, epoch, net, optimizer, device):
    filename = fileutils.ensure_ext('{}.{}'.format(fileutils.remove_ext(filename, '.pt'), epoch), '.pt')
    loaded_states = torch.load(filename, map_location=device)
    net.load_state_dict(loaded_states['model_state_dict'])
    optimizer.load_state_dict(loaded_states['optimizer_state_dict'])

# This allows to select the network class by using the class name in Settings
def initialize_net(feat_dim: int, n_speakers: int):
    module, class_name = Settings().network.network_class.rsplit('.', 1)
    FooBar = getattr(importlib.import_module(module), class_name)
    return FooBar(feat_dim, n_speakers)
