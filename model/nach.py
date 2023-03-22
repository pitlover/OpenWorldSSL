from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_s import build_resnet


class NACH(nn.Module):
    def __init__(self, cfg: Dict, cfg_los: Dict, num_classes: int):
        super().__init__()
        # cfg["model"], cfg["loss"]

        # ------------------- Model ------------------ #
        self.model = build_resnet(model_name=cfg["backbone"]["name"].lower(), num_classes=num_classes)
        self.model.load_state_dict(state_dict=torch.load(cfg["pretrained"]),
                                   strict=False)  # Load Imagenet pretrained weights

        for name, param in self.model.named_parameters():
            if 'linear' not in name and 'layer4' not in name:
                param.requires_grad = False

        # ------------------- Loss ------------------ #

    def forward(self, img):
        return img
