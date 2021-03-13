# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import TextIO
import torch

def test_finiteness(tensor, description):
    if (~torch.isfinite(tensor)).sum() > 0:
        print('{}: NOT FINITE!'.format(description))

def print_embedding_stats(embeddings: torch.Tensor):
    means = embeddings.mean(dim=0)
    stds = embeddings.std(dim=0)
    print('Embedding statistics: ')
    for i in range(means.numel()):
        print('mean =  {:.10f}  std = {:.10f}   [Dim {}]'.format(means[i], stds[i], i+1))


def dual_print(print_file: TextIO, text: str):
    print(text)
    print_file.write(text + '\n')
    print_file.flush()
