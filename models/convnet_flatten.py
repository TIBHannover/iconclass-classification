import re
import argparse
import logging

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torchvision.models import resnet50, resnet152, densenet161
from models.models import ModelsManager

from models.resnet import ResNet50
from models.base_model import BaseModel
from models.loss import FocalBCEWithLogitsLoss

from datasets.utils import read_jsonl

from models.utils import linear_rampup, cosine_rampdown
from sklearn.metrics import fbeta_score

@ModelsManager.export("convnet_flatten")
class ConvnetFlatten(BaseModel):
    def __init__(self, args=None, **kwargs):
        super(ConvnetFlatten, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.save_hyperparameters()

        self.encoder_model = dict_args.get("encoder_model", None)
        self.pretrained = dict_args.get("pretrained", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)

        self.using_weights = dict_args.get("using_weights", False)
        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)
        
        self.use_focal_loss = dict_args.get("use_focal_loss", None)
        self.focal_loss_gamma = dict_args.get("focal_loss_gamma", None)
        self.focal_loss_alpha = dict_args.get("focal_loss_alpha", None)
        
        self.mapping_config = []
        if self.mapping_path is not None:
            mm = read_jsonl(self.mapping_path)
            for x in mm:
                if x['count'] >self.filter_label_by_count:
                    self.mapping_config.append(x)

        self.classifier_config = []
        if self.classifier_path is not None:
            self.classifier_config = read_jsonl(self.classifier_path)

        if self.encoder_model == "resnet152":
            self.net = resnet152(pretrained=self.pretrained)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.dim = 2048
        elif self.encoder_model == "densenet161":
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
                self.weights[0, x["class_id"]] = x["weight"]
        self.weights = torch.tensor(self.weights)

        if self.use_focal_loss:
            self.loss = FocalBCEWithLogitsLoss(
                reduction="none", gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha
            )
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()
        # self.f1_val = pl.metrics.classification.F1(num_classes=len(self.mapping_config), multilabel=True, average=None)

        self.byol_embedding_path = dict_args.get("byol_embedding_path", None)

        if self.byol_embedding_path is not None:
            self.load_pretrained_byol(self.byol_embedding_path)

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
        target = batch["target"]

        self.image = image

        logits = self(image)
        self.weights = self.weights.to(logits.device)
        loss = self.loss(logits, target) * self.weights
        return {"loss": loss}
        # return {"loss": loss, "prediction": F.sigmoid(logits), "target": target}

    def training_step_end(self, outputs):

        self.log("train/loss", outputs["loss"].mean(), prog_bar=True)
        return {
            "loss": outputs["loss"].mean(),
            # "progress_bar": {"train/loss": outputs["loss"].mean(),},
            # "log": {"train/loss": outputs["loss"].mean(),},
        }

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        target = batch["target"]
        logits = self(image)

        loss = self.loss(logits, target)
        # self.f1_val(torch.sigmoid(logits), target)
        tt = target.detach().cpu().numpy()
        # pp = flat_prediction.detach().cpu().numpy()
        pp = torch.sigmoid(logits).detach().cpu().numpy()
        ll = torch.mean(loss).detach().cpu().numpy()

        self.all_targets.append(tt)
        self.all_predictions.append(pp)
        self.all_losses.append(ll)

        return {"loss": torch.mean(loss)}

    # def validation_step_end(self, outputs):
    #     self.average_precision(outputs["pred"], outputs["target"])
    #     print(self.average_precision)
    def on_validation_epoch_start(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_losses = []

    def validation_epoch_end(self, outputs):

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1
        print(loss.shape)
        self.log("val/loss", loss / count, prog_bar=True)
        
        def binarize_prediction(probabilities, threshold: float, argsorted=None, min_labels=1, max_labels=10):
            """Return matrix of 0/1 predictions, same shape as probabilities."""
            # assert probabilities.shape[1] == self.num_classes
            if argsorted is None:
                argsorted = probabilities.argsort(axis=1)

            max_mask = _make_mask(argsorted, max_labels)
            min_mask = _make_mask(argsorted, min_labels)
            prob_mask = probabilities > threshold
            return (max_mask & prob_mask) | min_mask

        def _make_mask(argsrtd, top_n: int):
            mask = np.zeros_like(argsrtd, dtype=np.uint8)
            col_indices = argsrtd[:, -top_n:].reshape(-1)
            row_indices = [i // top_n for i in range(len(col_indices))]
            mask[row_indices, col_indices] = 1
            return mask

        def get_score(y_pred, all_targets):
            return fbeta_score(all_targets, y_pred, beta=2, average="macro")

        self.all_predictions = np.concatenate(self.all_predictions, axis=0)
        self.all_targets = np.concatenate(self.all_targets, axis=0)
        self.all_losses = np.stack(self.all_losses)

        # nonfiltered_lbs = np.where(~mask.numpy())
        # self.all_predictions = np.delete(self.all_predictions, nonfiltered_lbs, axis=1)
        # self.all_targets = np.delete(self.all_targets, nonfiltered_lbs, axis=1)
        metrics = {}
        arg_sorted = self.all_predictions.argsort(axis=1)
        for threshold in [0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            metrics[f"valid_f2_th_{threshold:.2f}"] = get_score(
                binarize_prediction(self.all_predictions, threshold, arg_sorted), self.all_targets
            )
        metrics["valid_loss"] = np.mean(self.all_losses)
        # print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        #     metrics.items(), key=lambda kv: -kv[1])))
        for kk, vv in sorted(metrics.items(), key=lambda kv: -kv[1]):
            self.log(kk, np.nanmean(round(vv, 3)), prog_bar=True)


        # f1_score = self.f1_val.compute().cpu().detach()

        # self.log("val/f1", np.nanmean(f1_score), prog_bar=True)

        # for i, x in enumerate(self.classifier_config):

        #     self.log(f"val/f1_{i}", np.nanmean(f1_score[x["range"][0] : x["range"][1]]), prog_bar=True)

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument(
            "--encoder_model", choices=("resnet152", "densenet161", "resnet50", "inceptionv3"), default="resnet50"
        )
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--using_weights", type=bool, default=False)
        parser.add_argument("--byol_embedding_path", type=str)
        parser.add_argument("--filter_label_by_count", type=int, default=0)
        
        parser.add_argument("--use_focal_loss", action="store_true", default=False)
        parser.add_argument("--focal_loss_gamma", type=float, default=2)
        parser.add_argument("--focal_loss_alpha", type=float, default=0.25)
        return parser

    def load_pretrained_byol(self, path_checkpoint):
        assert self.encoder_model == "resnet50", "BYOL currently working with renset50"
        data = torch.load(path_checkpoint)["state_dict"]

        load_dict = {}
        for name, var in data.items():
            if "model.target_net.0._features" in name:
                new_name = re.sub("^model.target_net.0._features.", "", name)
                load_dict[new_name] = var
        self.net.load_state_dict(load_dict)
