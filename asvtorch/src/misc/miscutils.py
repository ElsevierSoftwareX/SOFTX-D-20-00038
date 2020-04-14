from typing import TextIO
import torch

def test_finiteness(tensor, description):
    if (~torch.isfinite(tensor)).sum() > 0:
        print('{}: NOT FINITE!'.format(description))

def dual_print(print_file: TextIO, text: str):
    print(text)
    print_file.write(text + '\n')
    print_file.flush()
