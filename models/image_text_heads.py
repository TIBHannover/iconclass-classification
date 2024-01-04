import os
import argparse
import logging
import typing_extensions

import numpy as np

np.set_printoptions(edgeitems=40)

import torch
from torch.nn import functional as F
from torch import nn

torch.set_printoptions(edgeitems=40)

import pytorch_lightning as pl
from models.models import ModelsManager


from models.base_model import BaseModel
from models.utils import gen_filter_mask
from datasets.utils import read_line_data

from models.loss import ClipLoss, FocalBCEWithLogitsLoss
from encoders import EncodersManager
from decoders import DecodersManager
from heads import HeadsManager

from metrics import FBetaMetric, MAPMetric
from torchmetrics import CosineSimilarity, JaccardIndex

from utils.strategy import world_info_from_env

from models import utils
from models.utils import linear_rampup, cosine_rampdown


@ModelsManager.export("image_text_heads")
class ImageTextHeads(BaseModel):
    def __init__(self, args=None, **kwargs):
        super(ImageTextHeads, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.encoder_model = dict_args.get("encoder_model", None)
        self.pretrained = dict_args.get("pretrained", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.ontology_path = dict_args.get("ontology_path", None)
        self.outpath_f2_per_lbs = dict_args.get("outpath_f2_per_lbs", None)

        self.use_label_smoothing = dict_args.get("use_label_smoothing", None)
        self.label_smoothing_factor = dict_args.get("label_smoothing_factor", 0.008)
        self.using_weights = dict_args.get("using_weights", None)

        self.use_focal_loss = dict_args.get("use_focal_loss", None)
        self.focal_loss_gamma = dict_args.get("focal_loss_gamma", None)
        self.focal_loss_alpha = dict_args.get("focal_loss_alpha", None)

        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.best_threshold = dict_args.get("best_threshold", None)

        self.load_model_from_checkpoint = dict_args.get("load_model_from_checkpoint", None)

        self.output_method = dict_args.get("output_method")

        self.mapping_config = []
        self.mask_vec = []
        if self.mapping_path is not None:
            self.mapping_config = read_line_data(self.mapping_path)

        self.classifier = []
        if self.classifier_path is not None:
            self.classifier = read_line_data(self.classifier_path)

        self.ontology = []
        if self.ontology_path is not None:
            self.ontology = read_line_data(self.ontology_path)

        if self.filter_label_by_count is not None:
            self.filter_mask = torch.tensor(
                gen_filter_mask(self.mapping_config, self.filter_label_by_count, key="count.flat")
            )
        else:
            self.filter_mask = torch.ones(len(self.mapping_config), dtype=torch.float32)

        self.num_of_labels = torch.tensor(sum(self.mask_vec))

        if self.output_method == "level":
            self.level_map = utils.build_level_map(self.mapping_config)
            self.output_sizes = [2 + torch.sum(self.level_map == i).item() for i in range(len(self.ontology))]
            self.embedding_sizes = 2 + len(self.mapping_config)
            self.mapper = utils.HierarchicalLevelMapper(mapping=self.mapping_config, classifier=self.classifier)

        elif self.output_method == "minimal":
            self.output_sizes = [len(x["tokenizer"]) for x in self.ontology]  # get from tockenizer
            self.embedding_sizes = max(self.output_sizes)
            self.mapper = None

        elif self.output_method == "global":
            self.output_sizes = [2 + len(self.mapping_config) for i in range(len(self.ontology))]
            self.embedding_sizes = 2 + len(self.mapping_config)
            self.mapper = None

        # if self.use_weights is not None:
        #     self.weights = [x['weight'] for x in self.mapping_config]
        #     self.weights = np.array(self.weights)

        self.encoder = EncodersManager().build_encoder(name=args.encoder, args=args)
        self.heads = torch.nn.ModuleList(HeadsManager().build_heads(names=args.heads, args=args))
        self.decoder = DecodersManager().build_decoder(
            name=args.decoder,
            args=args,
            in_features=self.encoder.dim,
            embedding_size=self.embedding_sizes,
            out_features=len(self.mapping_config),
            vocabulary_sizes=self.output_sizes,
            mapper=self.mapper,
        )

        # if self.using_weights:
        #     logging.info("Using weighting for loss")

        #     # for x in self.mapping_config:
        #     # self.weights[0, x["class_id"]] = x["weight"]
        #     self.weights = [x["weight_pos"] for x in self.mapping_config]
        #     self.weights = torch.Tensor(self.weights)
        # else:
        #     self.weights = torch.ones(len(self.mapping_config))
        # # print(self.weights)
        # if self.use_focal_loss:
        #     self.loss = FocalBCEWithLogitsLoss(
        #         reduction="none", gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha
        #     )
        # else:
        #     self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # self.fbeta = FBetaMetric(num_classes=len(self.mapping_config))
        # self.map = MAPMetric(num_classes=len(self.mapping_config))

        # self.jaccard = JaccardIndex(num_classes=len(self.mapping_config), multilabel=True)
        # self.cosine = CosineSimilarity(reduction="mean")

        # self.cosine = {h.name: CosineSimilarity(reduction="mean") for h in self.heads if hasattr(h, "flat_prediction")}

        if self.load_model_from_checkpoint:
            state_dict = torch.load(self.load_model_from_checkpoint)["state_dict"]
            # for x in state_dict:
            #     print(x)
            # state_dict = {k: v for k, v in state_dict}
            # state_dict = flat_dict(unflat_dict(state_dict)["encoder"])
            # print(state_dict.keys())
            self.load_state_dict(state_dict)
            logging.info(f"Load checkpoint {self.load_model_from_checkpoint}")
            # exit()

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)

        return encoder_outputs

    def training_step(self, batch, batch_idx):
        # for k, v in batch.items():
        #     if hasattr(v, "shape"):
        #         print(k, v.shape)
        # logging.info("training_step")
        # print(f"########## {batch_idx}")
        # print(f"{os.environ.get('LOCAL_RANK')} {batch.get('id')[:5]}")
        image = batch["image"]

        # assert "clip_embedding" in batch, "Expect clip_embedding from dataloader"
        if "clip_embedding" in batch:
            text = batch["clip_embedding"]
        else:
            text = None

        # target = batch["ontology_target"]

        self.image = image

        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image
        encoder_outputs = self.encoder({"image": image, "text": text})

        decoder_outputs = self.decoder({**encoder_outputs, **batch})
        # image_embedding = encoder_outputs.get("image_embedding")
        self.log("train/clip/scale", encoder_outputs.get("scale"))

        losses = []
        for head in self.heads:
            if hasattr(head, "loss"):
                loss = head.loss(self, batch, {**encoder_outputs, **decoder_outputs})
                self.log(f"train/{head.name}/loss", loss, prog_bar=True)
                losses.append(loss)
        return {"loss": torch.sum(torch.stack(losses))}

    def validation_step(self, batch, batch_idx):
        # for k, v in batch.items():
        #     if hasattr(v, "shape"):
        #         print(k, v.shape)
        logging.info("validation_step")
        image = batch["image"]
        target = batch["ontology_target"]

        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image

        # assert "clip_embedding" in batch, "Expect clip_embedding from dataloader"
        if "clip_embedding" in batch:
            text = batch["clip_embedding"]
        else:
            text = None

        encoder_outputs = self.encoder({"image": image, "text": text})
        image_embedding = encoder_outputs.get("image_embedding")
        decoder_outputs = self.decoder({**encoder_outputs, **batch})

        losses = []
        # for head in self.heads:
        #     if hasattr(head, "flat_prediction"):
        #         flat_prediction = head.flat_prediction(self, batch, {**encoder_outputs, **decoder_outputs})
        #         self.cosine[head.name](flat_prediction, batch["yolo_target"])

        losses = []
        for head in self.heads:
            loss = head.loss(self, batch, {**encoder_outputs, **decoder_outputs})
            self.log(f"val/{head.name}/loss", loss, prog_bar=True)
            losses.append(loss)
        return {"loss": torch.sum(torch.stack(losses))}

    def validation_epoch_end(self, outputs):
        logging.info("validation_epoch_end")

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1

        self.log("val/loss", loss / count, prog_bar=True)

        # for head_name, metric in self.cosine.items():
        #     self.log(f"val/{head_name}/cosine", metric.compute(), prog_bar=True)
        #     # self.jaccard.reset()
        #     metric.reset()

    def test_step(self, batch, batch_idx):
        logging.info("test_step")
        image = batch["image"]
        target = batch["ontology_target"]

        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image

        assert "clip_embedding" in batch, "Expect clip_embedding from dataloader"
        text = batch["clip_embedding"]

        encoder_outputs = self.encoder({"image": image, "text": text})
        image_embedding = encoder_outputs.get("image_embedding")
        decoder_outputs = self.decoder({**encoder_outputs, **batch})

        predictions = {}
        for head in self.heads:
            if hasattr(head, "flat_prediction"):
                flat_prediction = head.flat_prediction(self, batch, {**encoder_outputs, **decoder_outputs})
                predictions[head.name] = flat_prediction
                # self.cosine[head.name](flat_prediction, batch["yolo_target"])

        losses = {}
        for head in self.heads:
            loss = head.loss(self, batch, {**encoder_outputs, **decoder_outputs})
            losses[head.name] = flat_prediction
            # self.log(f"val/{head.name}/loss", loss, prog_bar=True)
            # losses.append(loss)
        return {"flat_prediction": predictions, "losses": losses}

    def configure_optimizers(self):
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": self.weight_decay},
            ],
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
        )

        if self.sched_type == "cosine":

            def cosine_lr(step):
                # epoch = step * batch_size / len(train_dataset)

                r = linear_rampup(step, self.lr_rampup)
                lr = r * (1.0 - self.lr_init) + self.lr_init

                if self.lr_rampdown:
                    lr *= cosine_rampdown(step, self.lr_rampdown)

                return lr

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)
        elif self.sched_type == "exponetial":

            def exp_lr(step):
                decayed_learning_rate = self.gamma ** (step / self.step_size)
                return decayed_learning_rate

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp_lr)

        elif self.sched_type == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.step_size)

        # optimizer = torch.optim.SGD(
        #     params=list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1.0, weight_decay=self.params.optimizer.weight_decay,momentum=0.9
        # )
        else:
            return optimizer

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]

    def on_validation_epoch_start(self):
        logging.info("on_validation_epoch_start")
        for head in self.heads:
            if hasattr(head, "on_validation_epoch_start"):
                head.on_validation_epoch_start(self)

    def on_test_epoch_start(self):
        logging.info("on_test_epoch_start")
        for head in self.heads:
            if hasattr(head, "on_test_epoch_start"):
                head.on_test_epoch_start(self)

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("EncoderHierarchicalDecoder::add_args")
        parent_parser = super().add_args(parent_parser)
        parent_parser = EncodersManager.add_args(parent_parser)
        parent_parser = DecodersManager.add_args(parent_parser)
        parent_parser = HeadsManager.add_args(parent_parser)
        print(f"parent {parent_parser}", flush=True)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--ontology_path", type=str)

        parser.add_argument("--outpath_f2_per_lbs", type=str)

        parser.add_argument("--use_diverse_beam_search", action="store_true", default=False)
        parser.add_argument("--div_beam_s_group", type=int, default=2)

        parser.add_argument("--use_label_smoothing", action="store_true", default=False)
        parser.add_argument("--using_weights", action="store_true", default=False)
        parser.add_argument("--use_focal_loss", action="store_true", default=False)
        parser.add_argument("--focal_loss_gamma", type=float, default=2)
        parser.add_argument("--focal_loss_alpha", type=float, default=0.25)

        parser.add_argument("--filter_label_by_count", type=int, default=None)
        parser.add_argument("--output_method", choices=["global", "level", "minimal"], default="level")

        parser.add_argument("--best_threshold", nargs="+", type=float, default=[0.2])
        parser.add_argument("--targets", choices=("flat", "yolo", "clip", "onto"), nargs="+")

        parser.add_argument("--load_model_from_checkpoint", type=str)

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

        print(f"parent {parser}", flush=True)
        return parser
