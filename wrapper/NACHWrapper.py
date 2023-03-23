from typing import Dict, Tuple, List
import torch
import torch.nn as nn

from model.nach import NACH

__all__ = [
    "NACHWrapper"
]


class NACHWrapper(nn.Module):

    def __init__(self,
                 cfg: Dict,
                 model: NACH,
                 ) -> None:
        super().__init__()
        # cfg = cfg

        self.model = model

        # ------------------- Loss ------------------ #

    def forward(self,
                img1: torch.Tensor,
                label: torch.Tensor,
                img2: torch.Tensor = None,
                aug_weak: torch.Tensor = None,
                aug_strong: torch.Tensor = None,
                iter: int = 0):
        return img1
