import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class FocalBCEWithLogitsLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, alpha=0.25, reduction="mean"):
        super(FocalBCEWithLogitsLoss, self).__init__(weight, reduction=reduction)

        if gamma is not None:
            gamma = torch.tensor(gamma, dtype=torch.float32)

        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32)

        self.register_buffer("gamma", gamma)
        self.register_buffer("alpha", alpha)
        self.register_buffer("weight", weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print("#############")
        ce = F.binary_cross_entropy_with_logits(input, target, reduction="none", weight=self.weight)
        # print(ce)
        pred_prob = torch.sigmoid(input)

        p_t = (target * pred_prob) + ((1 - target) * (1 - pred_prob))
        # print(pt)

        # alpha_factor = 1.0
        # modulating_factor = 1.0

        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)

        modulating_factor = torch.pow((1.0 - p_t), self.gamma)
        loss = alpha_factor * modulating_factor * ce
        # print(loss)
        return loss

    # # compute the final loss and return

    # return tf.reduce_sum(, axis=-1)

    #     focal_loss = (1 - pt) ** self.gamma * ce_loss

    #     if self.reduction == "mean":
    #         focal_loss = focal_loss.mean()

    #     return focal_loss
