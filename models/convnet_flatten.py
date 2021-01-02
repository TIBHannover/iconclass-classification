import re
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

from models.utils import linear_rampup, cosine_rampdown


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

        self.opt_lr = dict_args.get("opt_lr", None)
        self.weight_decay = dict_args.get("weight_decay", None)
        self.opt_type = dict_args.get("opt_type", None)
        self.sched_type = dict_args.get("sched_type", None)
        self.lr_rampup = dict_args.get("lr_rampup", None)
        self.lr_init = dict_args.get("lr_init", None)
        self.lr_rampdown = dict_args.get("lr_rampdown", None)
        self.gamma = dict_args.get("gamma", None)
        self.step_size = dict_args.get("step_size", None)

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
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(self.dim, 1024)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(1024, len(self.mapping))
        self.loss = torch.nn.BCEWithLogitsLoss()

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
        def build_optimizer(model, type, **kwargs):
            parameterwise = {
                "(bn|gn)(\d+)?.(weight|bias)": dict(weight_decay=0.0, lars_exclude=True),
                "bias": dict(weight_decay=0.0, lars_exclude=True),
            }
            if parameterwise is None:
                params = model.parameters()

            else:
                params = []
                for name, param in model.named_parameters():
                    param_group = {"params": [param]}
                    if not param.requires_grad:
                        params.append(param_group)
                        continue

                    for regexp, options in parameterwise.items():
                        if re.search(regexp, name):
                            for key, value in options.items():
                                param_group[key] = value

                    # otherwise use the global settings
                    params.append(param_group)
            if type.lower() == "sgd":
                return torch.optim.SGD(params=params, **kwargs)

            if type.lower() == "adam":
                return torch.optim.AdamW(params=params, **kwargs)

        optimizer = build_optimizer(self, type=self.opt_type, lr=self.opt_lr, weight_decay=self.weight_decay)

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
                decayed_learning_rate = self.lr * self.gamma ** (step / 10000)
                return decayed_learning_rate

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp_lr)

        # optimizer = torch.optim.SGD(
        #     params=list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1.0, weight_decay=self.params.optimizer.weight_decay,momentum=0.9
        # )
        else:
            return optimizer

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument("--encode_model", type=str, default="resnet50")
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--opt_lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--opt_type", choices=["SGD", "LARS", "ADAM"], default="ADAM")

        parser.add_argument("--sched_type", choices=["cosine", "exponetial"])
        parser.add_argument("--lr_rampup", default=10000, type=int)
        parser.add_argument("--lr_init", default=0.0, type=float)
        parser.add_argument("--lr_rampdown", default=60000, type=int)
        parser.add_argument("--gamma", default=0.5, type=float)
        parser.add_argument("--step_size", default=10000, type=int)

        return parser
