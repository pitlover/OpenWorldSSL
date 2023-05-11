from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import build_resnet
from model.dino import DinoFeaturizer
from model.head import DINOHead
from model.others.functions import freeze_layers
from model.others.loss import Entropy, Fixmatch


class Basic(nn.Module):
    def __init__(self, cfg: Dict, cfg_loss: Dict, num_classes: int, num_seen: int = 5):
        super().__init__()
        # cfg["model"], cfg["loss"]
        self.threshold = 0.95
        self.eps = 1e-10
        self.num_classes = num_classes
        self.num_seen = num_seen
        model_name = cfg["backbone"]["name"].lower()
        self.n_mlp_layers = cfg["backbone"]["n_mlp_layers"]
        self.output_dim = num_classes
        self.mlp_dim = cfg["backbone"]["mlp_dim"]

        # ------------------- Model ------------------ #
        if "resnet" in model_name:
            self.backbone = build_resnet(model_name=model_name, num_classes=num_classes)
            self.backbone.load_state_dict(state_dict=torch.load(cfg["backbone"]["pretrained"]),
                                          strict=False)  # Load pretrain with SimCLR
            if cfg["backbone"]["is_freeze"]:
                self.backbone = freeze_layers(self.backbone, model_name)  # (linear or fc) and layer4

        elif "dino" in model_name:
            self.backbone = DinoFeaturizer(cfg["backbone"], cfg["backbone"]["is_freeze"])
        else:
            raise ValueError(f"Not supported {model_name} model.")

        # ------------------- Loss ------------------ #
        self.entropy_loss = Entropy()

        # # -------- Head -------- #
        self.head = DINOHead(in_dim=self.backbone.feat_dim, out_dim=self.mlp_dim, nlayers=self.n_mlp_layers)
        self.classifier = nn.Linear(self.mlp_dim, self.num_classes)

    def forward(self,
                img: torch.Tensor,
                label: torch.Tensor,
                aug_weak: torch.Tensor = None,
                aug_strong: torch.Tensor = None,
                iter: int = 0):
        '''

        :param img: (b, 3, 32, 32)
        :param label: (b * label_ratio)
        :param aug_weak: (b, 3, 32, 32)  different from fixmatch_weak aug.
        :param aug_strong: (b, 3, 32, 32)
        :param iter:
        :return:
        '''
        num_label = len(label)
        b, c, h, w, = img.shape
        results = {}

        # dino (b, 768) , head (b, hidden_dim, h, w), classifier (b, num_classes)
        dino_feat, attn = self.backbone(img)
        head = self.head(dino_feat)
        out = self.classifier(head)

        if not self.training:
            return F.softmax(out, dim=1), None

        # (b, num_class)
        prob_out = F.softmax(out, dim=1)

        entropy_loss = self.entropy_loss(torch.mean(prob_out, 0))
        results["reg-loss"] = entropy_loss  # Regularization Term

        cross_entropy_loss = (F.cross_entropy(out[:num_label], label, reduction="none")).mean()

        results["sup-loss"] = cross_entropy_loss  # Supervised Loss

        return prob_out, results
