import argparse

import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torchvision.models import resnet50, resnet152, densenet161
from models.models import ModelsManager

from models.resnet import ResNet50

from datasets.utils import read_jsonl


@ModelsManager.export("convnet_flatten")
class ConvnetFlatten(LightningModule):
    def __init__(self, args=None, **kwargs):
        super(ConvnetFlatten, self).__init__()
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
        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_jsonl(self.mapping_path, dict_key="id")

        self.classifier = {}
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)

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

        self.fc = torch.nn.Linear(self.dim, 1024)
        self.classifier = torch.nn.Linear(1024, len(self.mapping))
        self.loss = torch.nn.BCEWithLogitsLoss()

        print(dir(pl.metrics.classification))
        # self.map_metric = pl.metrics.classification.AveragePrecision()  # len(self.mapping))

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        target = batch["target"]

        self.image = image

        logits = self(image)

        loss = self.loss(logits, target)
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

        return {"loss": loss}
        # return {"loss": loss, "prediction": torch.sigmoid(logits), "target": target}

    # def validation_step_end(self, outputs):
    #     ap = self.map_metric.update(outputs["prediction"], outputs["target"])
    #     return {**outputs, "ap": ap}

    def validation_epoch_end(self, outputs):

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1
        self.log("val/loss", loss / count, prog_bar=True)
        return {
            "loss": loss / count,
            # "progress_bar": {"val/loss": loss / count,},
            # "log": {"val/loss": loss / count,},
        }

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            params=list(self.net.parameters()) + list(self.classifier.parameters()) + list(self.fc.parameters()),
            lr=0.001,
        )

        return optimizer

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument("--encode_model", type=str, default="resnet50")
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        return parser
