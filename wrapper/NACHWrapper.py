from typing import Dict, Tuple, List
import torch
import torch.nn as nn

from model.nach import NACH
from model.functions import AverageMeter

__all__ = [
    "NACHWrapper"
]


class NACHWrapper(nn.Module):

    def __init__(self,
                 cfg: Dict,
                 model: NACH
                 ) -> None:
        super().__init__()
        # cfg = cfg

        self.mean_uncertainty = torch.ones(1)
        self.model = model

    def _initial_update(self):
        self.seen_probs = AverageMeter('id_prob', ':.4e')
        self.unseen_probs = AverageMeter('ood_prob', ':.4e')

    def _update_mean_uncertainty(self):
        self.mean_uncertainty = self.seen_probs.avg - self.unseen_probs.avg

    def forward(self,
                img1: torch.Tensor,
                label: torch.Tensor,
                img2: torch.Tensor = None,
                aug_weak: torch.Tensor = None,
                aug_strong: torch.Tensor = None,
                iter: int = 0,
                max_iter: int = 0):
        '''

        :param img1: (b, 3, h, w)
        :param label: (b * label_ratio, 3, h, w)
        :param img2:
        :param aug_weak:
        :param aug_strong:
        :param iter:
        :return:
        '''

        if iter == 0:  # Initialize for every epoch
            self._initial_update()
            self.mean_uncertainty = self.mean_uncertainty.to(img1.device)

        feat, results, seen_prob, unseen_prob = self.model(img=img1, label=label, aug_weak=img2, aug_strong=aug_strong,
                                                           iter=iter,
                                                           mean_uncertainty=self.mean_uncertainty)
        if not self.training:
            return feat

        self.seen_probs.update(seen_prob, n=img1.shape[0])
        self.unseen_probs.update(unseen_prob, n=img1.shape[0])

        if iter == max_iter - 1:
            self._update_mean_uncertainty()

        results["mean-uncertainty"] = self.mean_uncertainty

        model_loss = -1 * results["entropy-loss"] + results["bse-loss"] + \
                     results["cross-entropy-loss"] + results["fixmatch-loss"]
        results["loss"] = model_loss

        return feat, results
