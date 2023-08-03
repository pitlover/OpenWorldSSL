from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import build_resnet
from model.dino import DinoFeaturizer
from model.others.functions import freeze_layers
from model.others.loss import Entropy, Fixmatch


class Consistency(nn.Module):
    def __init__(self, cfg: Dict, cfg_loss: Dict, num_classes: int, num_seen: int = 5):
        super().__init__()
        # cfg["model"], cfg["loss"]
        self.threshold = 0.95
        self.eps = 1e-10
        self.num_classes = num_classes
        self.num_seen = num_seen
        self.pretrain_classifier_epoch = cfg["pretrain_classifier_epoch"]
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
            self.embeded_dim = 512
        elif "dino" in model_name:
            self.backbone = DinoFeaturizer(cfg["backbone"], cfg["backbone"]["is_freeze"])
            self.embeded_dim = self.backbone.feat_dim
        else:
            raise ValueError(f"Not supported {model_name} model.")

        # ------------------- Loss ------------------ #
        self.entropy_loss = Entropy()

        self.labeled_correct = 0
        self.unlabeled_correct = 0
        self.unlabeled_incorrect = 0
        # self.proxies = torch.nn.Parameter(torch.randn(num_seen, self.embeded_dim).cuda())
        # nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self,
                img: torch.Tensor,
                label: torch.Tensor,
                ulabel: torch.Tensor,
                aug_weak: torch.Tensor = None,
                aug_strong: torch.Tensor = None,
                iter: int = 0,
                epoch: int = -1):
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

        output, feat = self.backbone(img)  # fc, flatten, shape : (b, num_classes), (b, 512)
        assert feat.shape[-1] == self.embeded_dim

        if not self.training:
            return F.softmax(output, dim=1), None

        logit_aug_weak, feat_aug_weak = self.backbone(aug_weak)
        logit_aug_strong, feat_aug_strong = self.backbone(aug_strong)

        # (b, num_class)
        prob_out = F.softmax(output, dim=1)
        prob_aug_weak, prob_aug_strong = F.softmax(logit_aug_weak, dim=1), F.softmax(logit_aug_strong, dim=1)

        # loss
        cross_entropy_loss = (F.cross_entropy(output[:num_label], label, reduction="none")).mean()
        results["sup-loss"] = cross_entropy_loss  # Supervised Loss

        if epoch > self.pretrain_classifier_epoch:
            entropy_loss = self.entropy_loss(torch.mean(prob_out, 0))
            results["reg-loss"] = entropy_loss  # Regularization Term

            # labeled data
            _, labeled_img = F.softmax(output[:num_label], dim=1).max(1)
            _, top1_labeled_aug_weak = F.softmax(logit_aug_weak[:num_label], dim=1).max(1)
            _, top1_labeled_aug_strong = F.softmax(logit_aug_strong[:num_label], dim=1).max(1)

            # unlabeled data
            ulabel_seen_mask = ulabel.lt(self.num_seen)
            ulabel_unseen_mask = ~ulabel_seen_mask
            _, top1_unlabeled_seen_weak = F.softmax(logit_aug_weak[num_label:][ulabel_seen_mask],
                                                    dim=1).max(1)
            _, top1_unlabeled_unseen_weak = F.softmax(logit_aug_weak[num_label:][ulabel_unseen_mask], dim=1).max(1)

            _, top1_unlabeled_seen_strong = F.softmax(logit_aug_strong[num_label:][ulabel_seen_mask], dim=1).max(1)
            _, top1_unlabeled_unseen_strong = F.softmax(logit_aug_strong[num_label:][ulabel_unseen_mask], dim=1).max(1)

            results["label_seen"] = torch.sum(
                (top1_labeled_aug_weak == top1_labeled_aug_strong) & (top1_labeled_aug_strong < self.num_seen)) \
                                    / len(top1_labeled_aug_weak)
            results["unlabel_unseen"] = torch.sum(
                (top1_unlabeled_seen_weak == top1_unlabeled_seen_strong) & (top1_unlabeled_seen_weak < self.num_seen)) \
                                        / len(top1_unlabeled_unseen_weak)
            results["unlabel_seen"] = torch.sum(
                (top1_unlabeled_seen_weak == top1_unlabeled_seen_strong) & (top1_unlabeled_seen_weak < self.num_seen)) \
                                      / len(top1_unlabeled_seen_weak)

        return prob_out, results
