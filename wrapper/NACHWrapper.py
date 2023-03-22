from typing import Dict, Tuple
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
