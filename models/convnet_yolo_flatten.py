import argparse
import re
import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torchvision.models import resnet50, resnet152, densenet161
from models.models import ModelsManager


from datasets.utils import read_jsonl

from models.base_model import BaseModel

from pytorch_lightning.core.decorators import auto_move_data


@ModelsManager.export("convnet_yolo_flatten")
class ConvnetYoloFlatten(BaseModel):
    def __init__(self, args=None, **kwargs):
        super(ConvnetYoloFlatten, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.save_hyperparameters()

        self.encode_model = dict_args.get("encode_model", None)
        self.pretrained = dict_args.get("pretrained", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)

        self.using_weights = dict_args.get("using_weights", False)

        self.mapping_config = []
        if self.mapping_path is not None:
            self.mapping_config = read_jsonl(self.mapping_path)

        self.mapping_lut = {}
        for m in self.mapping_config:
            self.mapping_lut[m["index"]] = m

        self.classifier_config = []
        if self.classifier_path is not None:
            self.classifier_config = read_jsonl(self.classifier_path)

        # Network setup
        if self.encode_model == "resnet152":
            self.net = resnet152(pretrained=self.pretrained)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.dim = 2048
        elif self.encode_model == "densenet161":
            self.net = densenet161(pretrained=self.pretrained)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        else:
            self.net = resnet50(pretrained=self.pretrained)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.dim = 2048
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(self.dim, 1024)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(1024, len(self.mapping_config))

        self.weights = np.ones([1, len(self.mapping_config)])
        if self.using_weights:
            logging.info("Using weighting for loss")

            for x in self.mapping_config:
                self.weights[0, x["index"]] = x["weight"]
        self.weights = torch.tensor(self.weights)

        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.f1_val = pl.metrics.classification.F1(num_classes=len(self.mapping_config), multilabel=True, average=None)
        self.f1_test = pl.metrics.classification.F1(num_classes=len(self.mapping_config), multilabel=True, average=None)

        self.byol_embedding_path = dict_args.get("byol_embedding_path", None)

        if self.byol_embedding_path is not None:
            self.load_pretrained_byol(self.byol_embedding_path)

        self.test_filter_count = dict_args.get("test_filter_count", None)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        ids_vec = batch["ids_vec"]
        cls_vec = batch["cls_vec"]
        cls_ids_mask_vec = batch["cls_ids_mask_vec"]

        self.image = image

        logits = self(image)

        loss = self.loss(logits, ids_vec) * cls_ids_mask_vec
        loss = torch.sum(loss) / torch.sum(cls_ids_mask_vec)
        return {"loss": loss}

    def training_step_end(self, outputs):

        self.log("train/loss", outputs["loss"].mean(), prog_bar=True)

        return {
            "loss": outputs["loss"].mean(),
        }

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        ids_vec = batch["ids_vec"]
        cls_vec = batch["cls_vec"]
        cls_ids_mask_vec = batch["cls_ids_mask_vec"]
        logits = self(image)

        loss = self.loss(logits, ids_vec) * cls_ids_mask_vec
        loss = torch.sum(loss) / torch.sum(cls_ids_mask_vec)

        self.f1_val(torch.sigmoid(logits), ids_vec)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1

        self.log("val/loss", loss / count, prog_bar=True)

        f1_score = self.f1_val.compute().cpu().detach()

        self.log("val/f1", np.nanmean(f1_score), prog_bar=True)

        level_results = {}
        for i, x in enumerate(self.classifier_config):
            if x["depth"] not in level_results:
                level_results[x["depth"]] = []
            level_results[x["depth"]].append(f1_score[x["range"][0] : x["range"][1]])
        for depth, x in sorted(level_results.items(), key=lambda x: x[0]):
            self.log(f"val/f1_{depth}", np.nanmean(torch.cat(x, dim=0)), prog_bar=True)

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        ids_vec = batch["ids_vec"]
        cls_vec = batch["cls_vec"]
        cls_ids_mask_vec = batch["cls_ids_mask_vec"]
        logits = self(image)

        loss = self.loss(logits, ids_vec) * cls_ids_mask_vec
        loss = torch.sum(loss) / torch.sum(cls_ids_mask_vec)

        self.f1_test(torch.sigmoid(logits), ids_vec)

        return {"loss": loss}

    def test_epoch_end(self, outputs):

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1

        self.log("test/loss", loss / count, prog_bar=True)

        f1_score = self.f1_test.compute().cpu().detach()

        mask = np.ones([len(self.mapping_config)])
        if self.test_filter_count is not None and self.test_filter_count > 0:

            logging.info("Using count for filter metrics")

            for x in self.mapping_config:
                mask[x["index"]] = int(x.get("count", 1) > self.test_filter_count)
        print("######")
        print(mask)
        print(np.sum(mask))
        print(np.sum(np.ones([1, len(self.mapping_config)])))

        f1_score = f1_score * mask

        self.log("test/f1", np.nansum(f1_score) / np.nansum(mask), prog_bar=True)

        self.log(
            f"test/num_concepts", np.nansum(mask), prog_bar=True,
        )

        level_results = {}
        mask_results = {}
        for i, x in enumerate(self.classifier_config):
            if x["depth"] not in level_results:
                level_results[x["depth"]] = []
                mask_results[x["depth"]] = []
            level_results[x["depth"]].append(f1_score[x["range"][0] : x["range"][1]])
            mask_results[x["depth"]].append(mask[x["range"][0] : x["range"][1]])
        print(len(mask_results[0][0]))

        for depth, x in sorted(level_results.items(), key=lambda x: x[0]):
            self.log(
                f"test/f1_{depth}",
                np.nansum(torch.cat(x, dim=0)) / np.nansum(np.concatenate(mask_results[depth], axis=0)),
                prog_bar=True,
            )

            self.log(
                f"test/num_concepts_{depth}", np.nansum(np.concatenate(mask_results[depth], axis=0)), prog_bar=True,
            )

    @auto_move_data
    def infer_step(self, batch, k=10):
        image = batch["image"]
        logits = self(image)

        logits_np = logits.cpu().detach().numpy()
        probs_np = torch.sigmoid(logits).cpu().detach().numpy()
        print(probs_np.shape)
        for i in range(probs_np.shape[0]):
            top_index = probs_np[i].argsort()[-k:][::-1]
            # print(top_index.shape)

            classes = [self.mapping_lut[x] for x in top_index.tolist()]
            classes = [
                {"id": x["id"], "kw": x["kw"], "txt": x["txt"], "prob": probs_np[i, top_index[j]]}
                for j, x in enumerate(classes)
            ]
            # print(len(classes))
            # exit()
            yield {"logits": logits_np[i], "probs": probs_np[i], "top_index": top_index, "classes": classes}

        exit()

        return {"logits": logits_np, "probs": probs_np, "top_index": top_index, "classes": classes}

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument("--encode_model", type=str, default="resnet50")
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--byol_embedding_path", type=str)

        parser.add_argument("--using_weights", type=bool, default=False)
        parser.add_argument("--test_filter_count", type=int, default=-1)
        return parser

    def load_pretrained_byol(self, path_checkpoint):
        assert self.encode_model == "resnet50", "BYOL currently working with renset50"
        data = torch.load(path_checkpoint)["state_dict"]

        load_dict = {}
        for name, var in data.items():
            if "model.target_net.0._features" in name:
                new_name = re.sub("^model.target_net.0._features.", "", name)
                load_dict[new_name] = var
        self.net.load_state_dict(load_dict)

