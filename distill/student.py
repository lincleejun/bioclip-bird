"""The student: a small image encoder whose output lives in BioCLIP 2.5's image space.

Contract (what makes it a drop-in for gardener's BioCLIP adapter): a square crop in, a 1024-d
L2-normalised vector out, comparable by dot product with the teacher's name matrices
(`<bioscan>/data/names/cache/bioclip-2.5-vith14-*.npz`). No text tower, no classifier head: the
11,172-row bird matrix stays the classifier, so a species needs a name, not training photos.
"""
from __future__ import annotations

import timm
import torch
from torch import nn
from torchvision import transforms as T

DIM = 1024                                             # BioCLIP 2.5 ViT-H/14 embedding size
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))   # the teacher's (OpenAI CLIP)
SIGLIP = "siglip2-base"                                # google/siglip2-base-patch16-224's image tower: gardener already has it


def norm(arch: str) -> tuple:
    """(mean, std) the backbone was pretrained with."""
    return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if arch == SIGLIP else CLIP_NORM


class Student(nn.Module):
    def __init__(self, arch: str, pretrained: bool = False) -> None:
        super().__init__()
        self.arch = arch
        if arch == SIGLIP:                              # ViT-B/16, 93 M: the weights come from the local HF cache
            from gardener.service.adapters import siglip2
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(siglip2.MODEL_ID, revision=siglip2.REVISION).vision_model
            width = self.backbone.config.hidden_size
        else:
            self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
            width = self.backbone.num_features
        self.proj = nn.Linear(width, DIM)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=x).pooler_output if self.arch == SIGLIP else self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.proj(self.features(x)), dim=-1)


def preprocess(size: int, arch: str = "") -> T.Compose:
    """The teacher's framing (bicubic resize, centre crop) at the student's input size and normalisation; RGB in."""
    return T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(*norm(arch))])


def params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
