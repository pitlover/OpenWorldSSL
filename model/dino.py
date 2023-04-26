from typing import Tuple
import torch
import torch.nn as nn
import model.vit as vits

__all__ = ["DinoFeaturizer"]


class DinoFeaturizer(nn.Module):

    def __init__(self, cfg: dict):  # cfg["backbone"]
        super().__init__()
        self.cfg = cfg

        arch = self.cfg["model_type"]  # vit_small, vit_base
        patch_size = self.cfg["dino_patch_size"]
        self.patch_size = patch_size
        # self.is_dropout = cfg["dropout"]
        # self.dropout = nn.Dropout2d(p=cfg["drop_prob"])


        # # -------- Backbone -------- #
        self.freeze_backbone: bool = cfg.get("freeze_backbone", True)
        self.backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        self.backbone.requires_grad_(not self.freeze_backbone)
        self.backbone.eval()

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError(f"Unknown arch {arch} and patch size {patch_size}.")

        if cfg["pretrained_weights"] is not None:
            state_dict = torch.load(cfg["pretrained_weights"], map_location="cpu")
            state_dict = state_dict["teacher"]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {cfg["pretrained_weights"]} and loaded with msg: {msg}')
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.backbone.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.feat_dim = 384
        elif arch == "vit_base":
            self.feat_dim = 768

    def train(self, mode: bool = True):
        super().train(mode=mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DINO forward
        :param img:     (batch_size, 3, 224, 224)
        :return:        (batch_size, 384, 28, 28)
        """
        b, c, h, w = img.shape
        assert (h % self.patch_size == 0) and (w % self.patch_size == 0)

        self.backbone.eval()
        if self.freeze_backbone:
            with torch.no_grad():
                dino_feat, attn, qkv = self.backbone(img, return_qkv=True)  # (b, 768)
        else:
            dino_feat, attn, qkv = self.backbone(img, return_qkv=True)  # (b, 768)

        # dino_feat = dino_feat.unsqueeze(-1)  # .unsqueeze(-1)  # (b, 768, 1)
        # dino_feat = dino_feat[:, 1:, :].reshape(b, feat_h, feat_w, -1).permute(0, 3, 1,2).contiguous()  # (b, 384, h //ps, w// ps)

        # dino_feat = self.dropout(dino_feat)

        return dino_feat, attn, qkv
