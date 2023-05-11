from typing import Dict
import torch
import torch.nn as nn

from model.consistency import Consistency
from model.others.functions import AverageMeter

__all__ = [
    "ConsistencyWrapper"
]


class ConsistencyWrapper(nn.Module):

    def __init__(self,
                 cfg: Dict,
                 cfg_loss: Dict,
                 model: Consistency
                 ) -> None:
        super().__init__()
        # cfg = cfg

        self.model = model
        self.reg_weight = cfg_loss["reg_weight"]
        self.sup_weight = cfg_loss["sup_weight"]

    def forward(self,
                img1: torch.Tensor,
                label: torch.Tensor,
                ulabel: torch.Tensor,
                img2: torch.Tensor = None,
                aug_weak: torch.Tensor = None,
                aug_strong: torch.Tensor = None,
                iter: int = 0,
                epoch: int = -1):
        '''

        :param img1: (b, 3, h, w)
        :param label: (b * label_ratio, 3, h, w)
        :param img2:
        :param aug_weak:
        :param aug_strong:
        :param iter:
        :return:
        '''

        feat, results = self.model(img=img1, label=label, aug_weak=img2, aug_strong=aug_strong,
                                   iter=iter, epoch=epoch)
        if not self.training:
            return feat

        model_loss = self.reg_weight * results["reg-loss"] + self.sup_weight * results["sup-loss"]
        results["loss"] = model_loss

        return feat, results
