import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Hard_mining_TripletLoss(nn.Module):
    def __init__(self, margin=0.5, device=None):
        super().__init__()
        self.margin = margin
        self.device = device

    def _get_anchor_positive_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, p] is True if a and p are distinct and have same label.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """

        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size()[0]).bool().to(self.device)
        indices_not_equal = ~indices_equal # flip booleans

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
        # Combine the two masks
        mask = indices_not_equal & labels_equal

        return mask


    def _get_anchor_negative_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True if a and n have distinct labels.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
        mask = ~labels_equal # invert the boolean tensor

        return mask

    def forward(self, embeddings, labels):
        # Get the pairwise distance matrix
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist = torch.max(anchor_positive_dist, 1, keepdim=True)[0]

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = torch.max(pairwise_dist, 1, keepdim=True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * ~(mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = torch.min(anchor_negative_dist, 1, keepdim=True)[0]

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.max(hardest_positive_dist - hardest_negative_dist + self.margin, torch.Tensor([0.0]).to(self.device))

        # Get final mean triplet loss
        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.5, device=None):
        super().__init__()
        self.margin = margin
        self.device = device

    def get_anchor_positive_mask(self, labels):
        indices_eq = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        identity = torch.eye(labels.size(0), dtype=torch.bool, device=self.device)
        return indices_eq & ~identity

    def get_anchor_negative_mask(self, labels):
        return torch.ne(labels.unsqueeze(0), labels.unsqueeze(1))

    def forward(self, embeddings, labels):
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        mask_anchor_positive = self.get_anchor_positive_mask(labels)
        mask_anchor_negative = self.get_anchor_negative_mask(labels)

        # Hardest positive
        anchor_positive_dist = pairwise_dist * mask_anchor_positive.float()
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)

        # Hardest negative
        anchor_negative_dist = pairwise_dist * mask_anchor_negative.float()
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)

        # Расчет Triplet Loss
        triplet_loss = F.relu(
            hardest_positive_dist - hardest_negative_dist + self.margin
        )

        # Усреднение с учетом валидных триплетов
        valid_mask = triplet_loss > 1e-16
        loss = triplet_loss[valid_mask].sum() / (valid_mask.sum().float() + 1e-16)

        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=self.device)


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

