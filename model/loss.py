import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin, device=None):
        super().__init__()
        self.margin = margin
        self.device = device

    def forward(self, fs, ys):
        anchor, positive, negative = fs
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)

        dist_pos = (anchor - positive).square().sum(axis=-1)
        dist_neg = (anchor - negative).square().sum(axis=-1)

        loss = F.relu(dist_pos - dist_neg + self.margin)

        return loss.mean()
