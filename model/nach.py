from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_s import build_resnet
from model.functions import freeze_layers


class NACH(nn.Module):
    def __init__(self, cfg: Dict, cfg_los: Dict, num_classes: int):
        super().__init__()
        # cfg["model"], cfg["loss"]

        # ------------------- Model ------------------ #
        self.model = build_resnet(model_name=cfg["backbone"]["name"].lower(), num_classes=num_classes)
        self.model.load_state_dict(state_dict=torch.load(cfg["pretrained"]), strict=False)  # Load pretrain with SimCLR
        self.model = freeze_layers(self.model)  # (linear or fc) and layer4

        # ------------------- Loss ------------------ #

    def forward(self, img):
        return img
