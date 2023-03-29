import torch
import torch.nn as nn
import torch.nn.functional as F


class Entropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        x_ = torch.clamp(x, min=self.eps)
        b = x_ * torch.log(x_)

        if len(b.size()) == 2:  # Sample-wise entropy
            return - b.sum(dim=1).mean()
        elif len(b.size()) == 1:  # Distribution-wise entropy
            return - b.sum()
        else:
            raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class Fixmatch(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.eps = 1e-8
        self.num_classes = num_classes

    def forward(self, mask, pseudo_label, logit_strong, targets_u):

        if torch.sum(mask) > 0:
            p = torch.sum(pseudo_label[mask.bool()], dim=0)
            p = p / torch.sum(pseudo_label[mask.bool()])
            q = torch.ones(self.num_classes, device=mask.device)
            q = q / self.num_classes
            align = torch.log(p / q)
            loss = (F.cross_entropy(logit_strong + align, targets_u, reduction="none") * mask).mean()

        if torch.sum(mask) == 0:
            loss = (F.cross_entropy(logit_strong, targets_u, reduction="none") * mask).mean()

        return loss
