import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class FocalBCEWithLogitsLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalBCEWithLogitsLoss, self).__init__(weight, reduction=reduction)

        if gamma is not None:
            gamma = torch.tensor(gamma, dtype=torch.float32)

        self.register_buffer("gamma", gamma)
        self.register_buffer("weight", weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        # print(pt)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()

        return focal_loss
