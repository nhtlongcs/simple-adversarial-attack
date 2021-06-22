# import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torch_models


class EnsembleModel(nn.Module):
    """Some Information about EnsembleModel"""

    def __init__(
        self, device="cpu", pretrained=True,
    ):
        super().__init__()
        self.model_ls = [
            torch_models.resnet18(pretrained=pretrained).to(device),
            torch_models.vgg16(pretrained=pretrained).to(device),
            torch_models.googlenet(pretrained=pretrained).to(device),
        ]

    def forward(self, x):
        res = [
            torch.softmax(model(x), dim=1).unsqueeze(dim=1) for model in self.model_ls
        ]
        res = torch.cat(res, dim=1).mean(dim=1)
        return res
