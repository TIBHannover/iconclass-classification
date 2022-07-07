import logging
import argparse

from tqdm import tqdm

from datasets.utils import read_dict_data, read_line_data

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from torch import Tensor

from utils import world_info_from_env
from .manager import HeadsManager

import open_clip

openai_imagenet_template = [
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
]


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
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        local_rank, global_rank, world_size = world_info_from_env()
        device = image_features.device
        if world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                local_rank,
                world_size,
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
            if world_size > 1 and self.local_loss:
                labels = labels + num_logits * local_rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss


def build_probability_chain(mapping, prediction):
    parent_lut = {x["id"]: x for x in mapping}
    weighted_prediction = torch.zeros_like(prediction)
    for m in mapping:
        if len(m["parents"]) == 0:
            weighted_prediction[:, m["index"]] = prediction[:, m["index"]]
        else:
            parent_prob = weighted_prediction[:, parent_lut[m["parents"][-1]]["index"]]
            weighted_prediction[:, m["index"]] = parent_prob * prediction[:, m["index"]]
    return weighted_prediction


@HeadsManager.export("clip")
class CLIPHead(nn.Module):
    name = "clip"

    def __init__(self, args=None, **kwargs):
        super().__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.labels_path = dict_args.get("labels_path", None)
        self.mapping_path = dict_args.get("mapping_path", None)
        self.labels = {}
        if self.labels_path is not None:
            self.labels = read_dict_data(self.labels_path)

        self.mapping = []
        if self.mapping_path is not None:
            self.mapping = read_line_data(self.mapping_path)

        self.local_loss = dict_args.get("local_loss")
        self.gather_with_grad = dict_args.get("gather_with_grad")
        self.use_probability_chain = dict_args.get("use_probability_chain", None)

        self.clip_loss = ClipLoss(
            local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad,
            cache_labels=True,
            use_horovod=False,
        )

        self.test_label_cache = []
        self.val_label_cache = []

    def loss(self, model, targets, outputs):
        assert "image_features" in outputs, "image_features not in encoder outputs"
        assert "text_features" in outputs, "text_features not in encoder outputs"
        assert "scale" in outputs, "scale not in encoder outputs"

        return self.clip_loss(outputs.get("image_features"), outputs.get("text_features"), outputs.get("scale"))

    def build_weights(self, model):
        labels = []
        for m in self.mapping:
            class_id = m.get("id")
            label = self.labels.get(class_id)
            labels.append(label)

        zeroshot_weights = []
        with torch.no_grad():
            for label in tqdm(labels):
                texts = [template(label) for template in openai_imagenet_template[:1]]
                clip_embedding = open_clip.tokenize(texts).to(model.device)
                class_embeddings = model({"text": clip_embedding})
                # logging.info(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        return torch.stack(zeroshot_weights, dim=1)

    def flat_prediction(self, model, targets, outputs):

        image_features = F.normalize(outputs.get("image_features"), dim=-1)
        logits = 100.0 * image_features @ self.label_cache

        if self.use_probability_chain:
            return build_probability_chain(self.mapping, logits)
        return logits

    def on_test_epoch_start(self, model):
        logging.info("on_test_epoch_start")
        self.label_cache = self.build_weights(model)

    def on_validation_epoch_start(self, model):
        logging.info("on_validation_epoch_start")
        self.label_cache = self.build_weights(model)

    @classmethod
    def add_args(cls, parent_parser):
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--labels_path", type=str)
        parser.add_argument("--mapping_path", type=str)

        parser.add_argument("--use_probability_chain", action="store_true", default=False)
        parser.add_argument(
            "--local_loss",
            default=False,
            action="store_true",
            help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
        )
        parser.add_argument(
            "--gather_with_grad",
            default=False,
            action="store_true",
            help="enable full distributed gradient for feature gather",
        )

        return parser
