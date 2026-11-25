"""ArcFace additive angular margin head, written from scratch (Deng et al., CVPR 2019).

For the true class y the logit becomes  s * cos(theta_y + m)  instead of  s * cos(theta_y),
all other logits stay  s * cos(theta_j). Cross-entropy on these logits forces embeddings of
the same identity into a tight angular cone with a gap of m radians to other identities.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, emb_size, n_classes, s=64.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, emb_size))
        nn.init.normal_(self.weight, std=0.01)
        self.n_classes = n_classes
        self.s = s
        self.set_margin(m)

    def set_margin(self, m):
        """Callable every iteration so the margin can be warmed up (0 -> m)."""
        self.m = m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        # if theta + m > pi, cos(theta + m) stops being monotonic: fall back to a linear penalty
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels):
        with torch.autocast(device_type=emb.device.type, enabled=False):   # head always in fp32
            cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1, 1)
            sin = (1.0 - cos * cos).clamp(min=1e-7).sqrt()
            phi = cos * self.cos_m - sin * self.sin_m                      # cos(theta + m)
            phi = torch.where(cos > self.th, phi, cos - self.mm)
            onehot = F.one_hot(labels, self.n_classes).bool()
            logits = torch.where(onehot, phi, cos) * self.s
        return logits, cos        # cos (no margin) is handy for training-accuracy logging


class CosFaceHead(ArcFaceHead):
    """CosFace (Wang et al., 2018): additive *cosine* margin, logit = s * (cos(theta_y) - m)."""

    def set_margin(self, m):
        self.m = m

    def forward(self, emb, labels):
        with torch.autocast(device_type=emb.device.type, enabled=False):
            cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1, 1)
            onehot = F.one_hot(labels, self.n_classes).bool()
            logits = torch.where(onehot, cos - self.m, cos) * self.s
        return logits, cos


class SoftmaxHead(nn.Module):
    """Plain linear + softmax (no normalization, no margin): the ablation baseline."""

    def __init__(self, emb_size, n_classes, **_):
        super().__init__()
        self.fc = nn.Linear(emb_size, n_classes, bias=False)
        self.m = 0.0

    def set_margin(self, m):
        pass

    def forward(self, emb, labels):
        with torch.autocast(device_type=emb.device.type, enabled=False):
            emb = emb.float()
            logits = self.fc(emb)
            cos = F.linear(F.normalize(emb), F.normalize(self.fc.weight))
        return logits, cos


def build_head(kind, emb_size, n_classes, s=64.0, m=0.5):
    if kind == "arcface":
        return ArcFaceHead(emb_size, n_classes, s=s, m=m)
    if kind == "cosface":
        return CosFaceHead(emb_size, n_classes, s=s, m=m)
    if kind == "softmax":
        return SoftmaxHead(emb_size, n_classes)
    raise ValueError(kind)
