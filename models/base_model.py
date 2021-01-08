import re
import argparse

import torch
from pytorch_lightning.core.lightning import LightningModule

from models.utils import linear_rampup, cosine_rampdown


class BaseModel(LightningModule):
    def __init__(self, args=None, **kwargs):
        super(BaseModel, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.lr = dict_args.get("lr", None)
        self.weight_decay = dict_args.get("weight_decay", None)
        self.opt_type = dict_args.get("opt_type", None)
        self.sched_type = dict_args.get("sched_type", None)
        self.momentum = dict_args.get("momentum", 0.9)

        self.lr_rampup = dict_args.get("lr_rampup", None)
        self.lr_init = dict_args.get("lr_init", None)
        self.lr_rampdown = dict_args.get("lr_rampdown", None)
        self.gamma = dict_args.get("gamma", None)
        self.step_size = dict_args.get("step_size", None)

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--opt_type", choices=["SGD", "LARS", "ADAM"], default="ADAM")

        parser.add_argument("--sched_type", choices=["cosine", "exponetial"])

        parser.add_argument("--momentum", default=0.9, type=float)

        parser.add_argument("--lr_rampup", default=10000, type=int)
        parser.add_argument("--lr_init", default=0.0, type=float)
        parser.add_argument("--lr_rampdown", default=60000, type=int)

        parser.add_argument("--gamma", default=0.5, type=float)
        parser.add_argument("--step_size", default=10000, type=int)

        return parser

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

        if self.opt_type == "sgd":
            optimizer = build_optimizer(
                self, type=self.opt_type, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum
            )
        else:
            optimizer = build_optimizer(self, type=self.opt_type, lr=self.lr, weight_decay=self.weight_decay)

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

        # optimizer = torch.optim.SGD(
        #     params=list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1.0, weight_decay=self.params.optimizer.weight_decay,momentum=0.9
        # )
        else:
            return optimizer

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
