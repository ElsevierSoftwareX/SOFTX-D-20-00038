from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class SufficientStats():
    zeroth: torch.Tensor
    first: torch.Tensor
    second_sum: Optional[torch.Tensor]
