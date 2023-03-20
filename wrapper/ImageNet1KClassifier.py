from typing import Dict
import torch

from models.base_wrapper import WrapperBase
from models.vision.classifier import VisionClassifier
from losses.cross_entropy import CrossEntropyLoss
from metrics.accuracy import Accuracy


class ImageNet1KClassifier(WrapperBase):

    def __init__(self, cfg: Dict) -> None:
        """Initialize required components."""

        backbone_cfg = cfg["model"]["backbone"]
        head_cfg = cfg["model"]["head"]
        backbone_name = backbone_cfg.pop("name")
        head_name = head_cfg.pop("name")
        model_kwargs = {"backbone": backbone_cfg, "head": head_cfg}

        model = VisionClassifier(backbone_name=backbone_name, head_name=head_name, **model_kwargs)
        super().__init__(model=model)  # self.model = model

        self.loss_fn = CrossEntropyLoss(**cfg["loss"])
        self.metric_fn = Accuracy(**cfg["metric"])

    def forward(self, img: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param img:     (batch_size, 3, h, w)
        :param label:   (batch_size,)
        :return:        {   "loss",
                            "acc1",
                            "acc5",  }
        """
        output = {}

        logit = self.model(img)
        loss = self.loss_fn(logit, label)
        acc1, acc5 = self.metric_fn(logit, label)

        output["loss"] = loss
        output["acc1"] = acc1
        output["acc5"] = acc5
        return output

    def inference(self, img: torch.Tensor) -> torch.Tensor:
        logit = self.model(img)
        prob = torch.softmax(logit, dim=1)
        return prob