from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.resnet_s import build_resnet
from model.functions import freeze_layers
from model.others.loss import Entropy, Fixmatch


class NACH(nn.Module):
    def __init__(self, cfg: Dict, cfg_los: Dict, num_classes: int, num_seen: int = 5):
        super().__init__()
        # cfg["model"], cfg["loss"]
        self.threshold = 0.95
        self.eps = 1e-10
        self.num_classes = num_classes
        self.num_seen = num_seen
        model_name = cfg["backbone"]["name"].lower()

        # ------------------- Model ------------------ #
        self.backbone = build_resnet(model_name=model_name, num_classes=num_classes)
        self.backbone.load_state_dict(state_dict=torch.load(cfg["pretrained"]),
                                      strict=False)  # Load pretrain with SimCLR
        if cfg["backbone"]["is_freeze"]:
            self.backbone = freeze_layers(self.backbone, model_name)  # (linear or fc) and layer4

        # ------------------- Loss ------------------ #
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.entropy_loss = Entropy()
        self.fixmatch_loss = Fixmatch(num_classes)

    def forward(self,
                img: torch.Tensor,
                label: torch.Tensor,
                aug_weak: torch.Tensor = None,
                aug_strong: torch.Tensor = None,
                iter: int = 0,
                mean_uncertainty: float = 1):
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

        output, feat = self.backbone(img)  # fc, flatten (b, num_classes), (b, 512)

        if not self.training:
            return F.softmax(output, dim=1), None, None, None

        logit_aug_weak, feat_aug_weak = self.backbone(aug_weak)
        logit_aug_strong, feat_aug_strong = self.backbone(aug_strong)

        # (b, num_class)
        prob_out = F.softmax(output, dim=1)
        prob_aug_weak, prob_aug_strong = F.softmax(logit_aug_weak, dim=1), F.softmax(logit_aug_strong, dim=1)

        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)  # normalize

        cos_distance = torch.mm(feat_norm, feat_norm.t())  # (b, b)

        pseudo_label = torch.softmax(logit_aug_weak.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)

        index_seen, index_unseen = targets_u.lt(self.num_seen).float(), targets_u.ge(self.num_seen).float()
        seen_count, unseen_count = torch.sum(index_seen), torch.sum(index_unseen)

        # Adaptive Threshold
        mask_seen = (index_seen * max_probs).ge(self.threshold).float()
        mask_unseen = (index_unseen * max_probs).ge(self.threshold - min(2 * mean_uncertainty, 0.5)).float()

        seen_prob = torch.sum(index_seen * max_probs) / seen_count + self.eps
        if unseen_count != 0:
            unseen_prob = torch.sum(index_unseen * max_probs) / unseen_count + self.eps
        elif unseen_count == 0:
            unseen_prob = seen_prob - seen_prob

        mask = mask_seen + mask_unseen

        positive_pairs = []
        label_np = label.cpu().numpy()

        # Labeled Data Pair
        for i in range(num_label):
            label_i = label_np[i]
            idxs = np.where(label_np == label_i)[0]
            if len(idxs) == 1:
                positive_pairs.append(idxs[0])
            else:
                selected_idxs = np.random.choice(idxs, 1)

                while selected_idxs == i:
                    selected_idxs = np.random.choice(idxs, 1)
                positive_pairs.append(int(selected_idxs))

        # Unlabeled Data Pair
        unlabel_cosine_distance = cos_distance[num_label:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_distance, 2, dim=1)  # (unlabeled, 2), (unlabeled, 2)

        # FBCE
        choose_k = 2  # TODO should be fine-tuned with different task
        max_pos = torch.topk(cos_distance[:num_label, pos_idx[:, 1]], choose_k, dim=0)[0][choose_k - 1]  # (unlabeled)
        mask1 = (vals[:, 1] - max_pos).ge(0).float()
        mask0 = (vals[:, 1] - max_pos).lt(0).float()

        pos_idx_1 = (pos_idx[:, 1] * mask1).cpu().numpy()
        pos_idx_0 = (pos_idx[:, 0] * mask0).cpu().numpy()
        pos_idx = (pos_idx_1 + pos_idx_0).flatten().tolist()

        positive_pairs.extend(pos_idx)
        pos_prob = prob_aug_weak[positive_pairs, :]
        pos_sim = torch.bmm(prob_out.view(b, 1, -1), pos_prob.view(b, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = self.bce_loss(pos_sim, ones)
        results["bce-loss"] = bce_loss

        entropy_loss = self.entropy_loss(torch.mean(prob_out, 0))
        results["entropy-loss"] = entropy_loss  # Regularization Term

        cross_entropy_loss = (F.cross_entropy(output[:num_label], label, reduction="none")).mean()

        results["cross-entropy-loss"] = cross_entropy_loss  # Supervised Loss

        mask_in_bce = torch.ones_like(mask)
        mask_in_bce[num_label:] = mask_in_bce[num_label:] - mask0

        # Logits Alignment (DTA Loss)
        fixmatch_loss = self.fixmatch_loss(mask=mask, pseudo_label=pseudo_label, logit_strong=logit_aug_strong,
                                           targets_u=targets_u)
        results["fixmatch-loss"] = fixmatch_loss

        return prob_out, results, seen_prob, unseen_prob
