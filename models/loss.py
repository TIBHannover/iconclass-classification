import torch
import torch.nn as nn
import torch.distributed as dist
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


def gather_features(
    image_features, text_features, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = 6  # world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss