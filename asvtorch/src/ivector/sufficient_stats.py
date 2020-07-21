# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class SufficientStats():
    zeroth: torch.Tensor
    first: torch.Tensor
    second_sum: Optional[torch.Tensor]
