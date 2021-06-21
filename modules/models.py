import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torch_models


class EnsembleModel(nn.Module):
    """Some Information about EnsembleModel"""

    def __init__(
        self,
        model_ls=[
            "vit_base_patch16_224",
            "densenet121",
            "resnest14d",
            "efficientnet_b0",
        ],
        pretrained=True,
    ):
        super().__init__()
        self.model_ls = [
            torch_models.resnet18(pretrained=pretrained),
            torch_models.vgg16(pretrained=pretrained),
            torch_models.googlenet(pretrained=pretrained),
        ]

    def forward(self, x):
        res = [torch.softmax(model(x), dim=1) for model in self.model_ls]
        res = torch.cat(res).mean(dim=0)
        return res.unsqueeze(dim=0)
