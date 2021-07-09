import re
import argparse
import logging

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

import pytorch_lightning as pl
from models.models import ModelsManager

from models.resnet import ResNet50

from models.base_model import BaseModel
from datasets.utils import read_jsonl, read_jsonl_lb_mapping

from models.loss import FocalBCEWithLogitsLoss

from pytorch_lightning.core.decorators import auto_move_data

from models.encoder import Encoder
from sklearn.metrics import fbeta_score


import pandas as pd


@ModelsManager.export("convnet_attn_lstm")
class ConvnetAttnLstm(BaseModel):
    def __init__(self, args=None, **kwargs):
        super(ConvnetAttnLstm, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.encoder_model = dict_args.get("encoder_model", None)
        self.pretrained = dict_args.get("pretrained", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.label_mapping_path = dict_args.get("label_mapping_path", None)
        self.outpath_f2_per_lbs = dict_args.get("outpath_f2_per_lbs", None)

        self.use_diverse_beam_search = dict_args.get("use_diverse_beam_search", None)
        self.div_beam_s_group = dict_args.get("div_beam_s_group", None)

        self.use_label_smoothing = dict_args.get("use_label_smoothing", None)
        self.LABEL_SMOOTHING_factor = 0.008  # TODO
        self.using_weights = dict_args.get("using_weights", None)
        self.use_focal_loss = dict_args.get("use_focal_loss", None)
        self.focal_loss_gamma = dict_args.get("focal_loss_gamma", None)
        self.focal_loss_alpha = dict_args.get("focal_loss_alpha", None)
        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.best_threshold = dict_args.get("best_threshold", None)

        self.mapping_config = []
        if self.mapping_path is not None:
            self.mapping_config = read_jsonl(self.mapping_path)

        self.mapping_lut = {}
        for m in self.mapping_config:

            if len(m["parents"]) < 1:
                p = None
            else:
                p = m["parents"][-1]

            if p not in self.mapping_lut:
                self.mapping_lut[p] = {}
            self.mapping_lut[p][m["token_id_sequence"][-1]] = m["index"]

        self.classifier_config = {}
        if self.classifier_path is not None:
            self.classifier_config = read_jsonl(self.classifier_path)

        if self.label_mapping_path is not None:
            self.label_mapping = read_jsonl_lb_mapping(self.label_mapping_path)

        # if self.use_weights is not None:
        #     self.weights = [x['weight'] for x in self.mapping_config]
        #     self.weights = np.array(self.weights)

        self.max_level = len(self.classifier_config)

        self.vocabulary_size = [len(x["tokenizer"]) for x in self.classifier_config]  # get from tockenizer
        self.max_vocab_size = max(self.vocabulary_size)
        self.embedding_dim = 768#256
        self.attention_dim = 128
        # self.max_vocab_size = max(self.vocabulary_size)
        # self.encoder = Encoder(args, embedding_dim=self.embedding_dim, flatten_embedding=True)
        self.encoder = Encoder(args, embedding_dim=None,flatten_embedding=False )
        self.decoder = Decoder(
            self.vocabulary_size, self.embedding_dim, self.attention_dim, self.embedding_dim, self.max_vocab_size
        )

        if self.using_weights:
            logging.info("Using weighting for loss")

            # for x in self.mapping_config:
            # self.weights[0, x["class_id"]] = x["weight"]
            self.weights = [x["weight_pos"] for x in self.mapping_config]
            self.weights = torch.Tensor(self.weights)
        else:
            self.weights = torch.ones(len(self.mapping_config))
        print(self.weights)
        if self.use_focal_loss:
            self.loss = FocalBCEWithLogitsLoss(
                reduction="none", gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha
            )
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.weights)

        self.all_predictions = []
        self.all_targets = []
        self.all_losses = []

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        source = batch["source_id_sequence"]
        target = batch["target_vec"]
        parents = batch["parents"]

        self.image = image
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image
        image_embedding = self.encoder(image)
        if len(image_embedding[0].shape) ==2:
            image_embedding = [torch.unsqueeze(image_embedding[0], 1)]

        image_embedding = torch.cat(image_embedding, dim=1)

        # hidden = self.decoder.reset_state(image.shape[0]).to(image.device.index)
        hidden = self.decoder.init_hidden_state(image_embedding)

        # Feed <START> to the model in the first layer 1==<START>
        decoder_inp = torch.ones([image.shape[0]], dtype=torch.int64).to(image.device.index)

        loss = 0
        # predictions_list = []
        parents_lvl = [None] * image.shape[0]
        flat_prediction = torch.zeros(image.shape[0], len(self.mapping_config), dtype=image_embedding.dtype).to(
            image.device.index
        )
        flat_target = torch.zeros(image.shape[0], len(self.mapping_config), dtype=target[0].dtype).to(
            image.device.index
        )
        for i_lev in range(len(target)):
            predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)
            # print(f"Pre: {torch.min(predictions)} {torch.max(predictions)} {torch.mean(predictions)}")
            # print(
            #     f"Post: {torch.min(torch.sigmoid(predictions))} {torch.max(torch.sigmoid(predictions))} {torch.mean(torch.sigmoid(predictions))}"
            # )
            # predictions_list.append(torch.sigmoid(predictions))

            source_indexes, target_indexes = self.map_level_prediction(parents_lvl)
            parents_lvl = [x[i_lev] for x in parents]
            flat_prediction[target_indexes] = torch.sigmoid(predictions)[source_indexes]
            flat_target[target_indexes] = target[i_lev][source_indexes]

            # loss += torch.mean(self.loss(predictions, target[i_lev]))
            decoder_inp = source[i_lev]
            # decoder_inp = torch.tensor(target[i_lev]).to(torch.int64).to(image.device.index)

        if self.use_label_smoothing:
            targets_smooth = flat_target.float() * (1 - self.LABEL_SMOOTHING_factor) + 0.5 * self.LABEL_SMOOTHING_factor
        else:
            targets_smooth = flat_target
        lloss = self.loss(flat_prediction, targets_smooth)
        # total_loss = loss / len(target)

        # loss = total_loss
        return {"loss": lloss}

    def training_step_end(self, outputs):
        self.log("train/loss", outputs["loss"].mean(), prog_bar=True)
        # if (self.global_step % self.trainer.log_every_n_steps) == 0:
        #     for i, (pred, target) in enumerate(zip(outputs["predictions"], outputs["targets"])):
        #         self.logger.experiment.add_histogram(f"train/predict_{i}", pred, self.global_step)
        #         self.logger.experiment.add_histogram(f"train/target_{i}", target, self.global_step)

        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idx):

        image = batch["image"]
        source = batch["source_id_sequence"]
        target = batch["target_vec"]
        parents = batch["parents"]
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image
        image_embedding = self.encoder(image)
        print(image_embedding[0].shape)
        if len(image_embedding[0].shape) ==2:
            image_embedding = [torch.unsqueeze(image_embedding[0], 1)]


        image_embedding = torch.cat(image_embedding, dim=1)

        # hidden = self.decoder.reset_state(image.shape[0]).to(image.device.index)

        hidden = self.decoder.init_hidden_state(image_embedding)

        # Feed <START> to the model in the first layer 1==<START>
        decoder_inp = torch.ones([image.shape[0]], dtype=torch.int64).to(image.device.index)

        loss = 0
        # Check if batch contains all traces (target [BATCH_SIZE, MAX_SEQUENCE, LEVEL, MAX_CLASSIFIER])
        if "mask" in batch:
            for i_lev in range(target.shape[2]):
                decoder_inp = source[i_lev]
                predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)

                prediction_size = len(self.classifier_config[i_lev]["tokenizer"])

                target_lev = target[:, 0, i_lev, :prediction_size]  # loop over the max_seq
                loss += torch.mean(self.loss(predictions, target_lev))
                decoder_inp = torch.unsqueeze(source[:, 0, i_lev], dim=1)

        else:

            flat_prediction = torch.zeros(image.shape[0], len(self.mapping_config), dtype=image_embedding.dtype).to(
                image.device.index
            )

            flat_prediction_norm = torch.zeros(
                image.shape[0], len(self.mapping_config), dtype=image_embedding.dtype
            ).to(image.device.index)
            flat_target = torch.zeros(image.shape[0], len(self.mapping_config), dtype=target[0].dtype).to(
                image.device.index
            )
            parents_lvl = [None] * image.shape[0]
            for i_lev in range(len(target)):
                predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)
                # loss += torch.mean(self.loss(predictions, target[i_lev]))
                decoder_inp = source[i_lev]

                source_indexes, target_indexes = self.map_level_prediction(parents_lvl)

                flat_prediction[target_indexes] = torch.sigmoid(predictions)[source_indexes]

                # Compute normalized version (maxprediction == 1.)

                flat_prediction_norm[target_indexes] = torch.sigmoid(predictions)[source_indexes] / torch.max(
                    torch.sigmoid(predictions)
                )

                flat_target[target_indexes] = target[i_lev][source_indexes]

                # parents_lvl = parents[i_lev]
                parents_lvl = [x[i_lev] for x in parents]

            # total_loss = loss / len(target)

            total_loss = torch.mean(self.loss(flat_prediction, flat_target))

            tt = flat_target.detach().cpu().numpy()
            pp = flat_prediction.detach().cpu().numpy()
            # pp = flat_prediction_norm.detach().cpu().numpy()
            ll = total_loss.detach().cpu().numpy()

            self.all_targets.append(tt)
            self.all_predictions.append(pp)
            self.all_losses.append(ll)

        return {
            "loss": total_loss,
        }

    def map_level_prediction(self, parents):
        target_indexes = []
        source_indexes = []
        # print(parents)
        for batch_id, parent in enumerate(parents):
            if parent not in self.mapping_lut:
                continue
            p = self.mapping_lut[parent]
            for source_id, target_id in p.items():
                target_indexes.append([batch_id, target_id])
                source_indexes.append([batch_id, source_id])

        source_indexes = np.asarray(source_indexes)
        target_indexes = np.asarray(target_indexes)
        if len(source_indexes.shape) < 2:
            return np.zeros([2, 0]), np.zeros([2, 0])

        return np.swapaxes(source_indexes, 0, 1), np.swapaxes(target_indexes, 0, 1)

    def on_validation_epoch_start(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_losses = []

    def validation_epoch_end(self, outputs):

        if self.filter_label_by_count is not None and self.filter_label_by_count > 0:
            mask = torch.zeros(len(self.mapping_config), dtype=torch.bool)
            for x in self.mapping_config:
                mask[x["index"]] = True if x["count"] > self.filter_label_by_count else False
        else:
            mask = torch.ones(len(self.mapping_config), dtype=torch.bool)

        self.log("val/number_of_labels", torch.sum(mask), prog_bar=True)  # TODO False

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1

        # # f1score prediction
        # f1_score = self.f1_val.compute().cpu().detach()
        # self.log("val/f1", np.nanmean(f1_score), prog_bar=True)
        # f1_score[~mask] = np.nan
        # self.log("val/f1_mask", np.nanmean(f1_score), prog_bar=True)

        # level_results = {}
        # for i, x in enumerate(self.mapping_config):
        #     if len(x["parents"]) not in level_results:
        #         level_results[len(x["parents"])] = []
        #     level_results[len(x["parents"])].append(f1_score[x["index"]])
        # for depth, x in sorted(level_results.items(), key=lambda x: x[0]):
        #     self.log(f"val/f1_{depth}", np.nanmean(torch.stack(x, dim=0)), prog_bar=False)

        # # f1score each layer normalized prediction
        # f1_norm_score = self.f1_val_norm.compute().cpu().detach()
        # self.log("val/f1_norm", np.nanmean(f1_norm_score), prog_bar=True)
        # f1_norm_score[~mask] = np.nan
        # self.log("val/f1_norm_mask", np.nanmean(f1_norm_score), prog_bar=True)

        # level_results = {}
        # for i, x in enumerate(self.mapping_config):
        #     if len(x["parents"]) not in level_results:
        #         level_results[len(x["parents"])] = []
        #     level_results[len(x["parents"])].append(f1_norm_score[x["index"]])
        # for depth, x in sorted(level_results.items(), key=lambda x: x[0]):
        #     self.log(f"val/f1_norm_{depth}", np.nanmean(torch.stack(x, dim=0)), prog_bar=False)

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

        nonfiltered_lbs = np.where(~mask.numpy())
        self.all_predictions = np.delete(self.all_predictions, nonfiltered_lbs, axis=1)
        self.all_targets = np.delete(self.all_targets, nonfiltered_lbs, axis=1)
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

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        source = batch["source_id_sequence"]
        target = batch["target_vec"]
        parents = batch["parents"]
        # TODO add threshold for each label and scale factor for levels prediction
        # forward image
        image_embedding = self.encoder(image)
        image_embedding = torch.cat(image_embedding, dim=1)

        # hidden = self.decoder.reset_state(image.shape[0]).to(image.device.index)
        # print(hidden.device)
        hidden = self.decoder.init_hidden_state(image_embedding)

        # Feed <START> to the model in the first layer 1==<START>
        decoder_inp = torch.ones([image.shape[0]], dtype=torch.int64).to(image.device.index)

        loss = 0
        # Check if batch contains all traces (target [BATCH_SIZE, MAX_SEQUENCE, LEVEL, MAX_CLASSIFIER])
        if "mask" in batch:
            for i_lev in range(target.shape[2]):
                decoder_inp = source[i_lev]
                predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)

                prediction_size = len(self.classifier_config[i_lev]["tokenizer"])

                target_lev = target[:, 0, i_lev, :prediction_size]  # loop over the max_seq
                loss += torch.mean(self.loss(predictions, target_lev))
                decoder_inp = torch.unsqueeze(source[:, 0, i_lev], dim=1)

        else:

            flat_prediction = torch.zeros(image.shape[0], len(self.mapping_config), dtype=image_embedding.dtype).to(
                image.device.index
            )

            flat_prediction_norm = torch.zeros(
                image.shape[0], len(self.mapping_config), dtype=image_embedding.dtype
            ).to(image.device.index)
            flat_target = torch.zeros(image.shape[0], len(self.mapping_config), dtype=target[0].dtype).to(
                image.device.index
            )
            parents_lvl = [None] * image.shape[0]
            for i_lev in range(len(target)):
                predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)
                loss += torch.mean(self.loss(predictions, target[i_lev]))
                decoder_inp = source[i_lev]

                source_indexes, target_indexes = self.map_level_prediction(parents_lvl)

                flat_prediction[target_indexes] = torch.sigmoid(predictions)[source_indexes]

                # Compute normalized version (maxprediction == 1.)

                flat_prediction_norm[target_indexes] = torch.sigmoid(predictions)[source_indexes] / torch.max(
                    torch.sigmoid(predictions)
                )

                flat_target[target_indexes] = target[i_lev][source_indexes]

                # parents_lvl = parents[i_lev]
                parents_lvl = [x[i_lev] for x in parents]

            total_loss = loss / len(target)

            tt = flat_target.detach().cpu().numpy()
            pp = flat_prediction.detach().cpu().numpy()
            ll = total_loss.detach().cpu().numpy()

            self.all_targets.append(tt)
            self.all_predictions.append(pp)
            self.all_losses.append(ll)

        return {
            "test/loss": total_loss,
        }

    def map_level_prediction(self, parents):
        target_indexes = []
        source_indexes = []
        # print(parents)
        for batch_id, parent in enumerate(parents):
            if parent not in self.mapping_lut:
                continue
            p = self.mapping_lut[parent]
            for source_id, target_id in p.items():
                target_indexes.append([batch_id, target_id])
                source_indexes.append([batch_id, source_id])

        source_indexes = np.asarray(source_indexes)
        target_indexes = np.asarray(target_indexes)
        if len(source_indexes.shape) < 2:
            return np.zeros([2, 0]), np.zeros([2, 0])

        return np.swapaxes(source_indexes, 0, 1), np.swapaxes(target_indexes, 0, 1)

    def on_test_epoch_start(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_losses = []
        self.f_values = pd.DataFrame()

    def test_epoch_end(self, outputs):

        if self.filter_label_by_count is not None and self.filter_label_by_count > 0:
            mask = torch.zeros(len(self.mapping_config), dtype=torch.bool)
            for x in self.mapping_config:
                mask[x["index"]] = True if x["count"] > self.filter_label_by_count else False
        else:
            mask = torch.ones(len(self.mapping_config), dtype=torch.bool)

        self.log("test/number_of_labels", torch.sum(mask), prog_bar=True)  # TODO False

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["test/loss"]
            count += 1

        # # f1score prediction
        # f1_score = self.f1_val.compute().cpu().detach()
        # self.log("test/f1", np.nanmean(f1_score), prog_bar=True)
        # f1_score[~mask] = np.nan
        # self.log("val/f1_mask", np.nanmean(f1_score), prog_bar=True)

        # level_results = {}
        # for i, x in enumerate(self.mapping_config):
        #     if len(x["parents"]) not in level_results:
        #         level_results[len(x["parents"])] = []
        #     level_results[len(x["parents"])].append(f1_score[x["index"]])
        # for depth, x in sorted(level_results.items(), key=lambda x: x[0]):
        #     self.log(f"val/f1_{depth}", np.nanmean(torch.stack(x, dim=0)), prog_bar=False)

        # # f1score each layer normalized prediction
        # f1_norm_score = self.f1_val_norm.compute().cpu().detach()
        # self.log("val/f1_norm", np.nanmean(f1_norm_score), prog_bar=True)
        # f1_norm_score[~mask] = np.nan
        # self.log("val/f1_norm_mask", np.nanmean(f1_norm_score), prog_bar=True)

        # level_results = {}
        # for i, x in enumerate(self.mapping_config):
        #     if len(x["parents"]) not in level_results:
        #         level_results[len(x["parents"])] = []
        #     level_results[len(x["parents"])].append(f1_norm_score[x["index"]])
        # for depth, x in sorted(level_results.items(), key=lambda x: x[0]):
        #     self.log(f"val/f1_norm_{depth}", np.nanmean(torch.stack(x, dim=0)), prog_bar=False)

        self.log("test/loss", loss / count, prog_bar=True)

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
            return fbeta_score(all_targets, y_pred, beta=2, average=None)

        self.all_predictions = np.concatenate(self.all_predictions, axis=0)
        self.all_targets = np.concatenate(self.all_targets, axis=0)
        self.all_losses = np.stack(self.all_losses)

        nonfiltered_lbs = np.where(~mask.numpy())
        self.all_predictions = np.delete(self.all_predictions, nonfiltered_lbs, axis=1)
        self.all_targets = np.delete(self.all_targets, nonfiltered_lbs, axis=1)
        metrics = {}
        arg_sorted = self.all_predictions.argsort(axis=1)
        for threshold in self.best_threshold:
            sc = get_score(binarize_prediction(self.all_predictions, threshold, arg_sorted), self.all_targets)
            metrics[f"test_f2_th_{threshold:.2f}"] = sc
            self.f_values[f"test_f2_th_{threshold:.2f}"] = sc

        metrics["test_loss"] = np.mean(self.all_losses)
        self.f_values["num_lbs"] = np.sum(self.all_targets, axis=0)

        print("saving the F-scores for each label")
        self.f_values.to_csv(self.outpath_f2_per_lbs)

    @auto_move_data
    def infer_step(self, batch, k=10):
        # lll = dict((str(l), 0) for l in range(0,11))
        # if self.filter_label_by_count is not None and self.filter_label_by_count > 0:
        #     mask = torch.zeros(len(self.mapping_config), dtype=torch.bool)
        #     for x in self.mapping_config:
        #         mask[x["index"]] = True if x["count"] > self.filter_label_by_count else False
        #         if x["count"] > self.filter_label_by_count and x['level_id'] in lll:
        #             # print(x['level_id'])
        #             lll[x['level_id']] += 1
        image = batch["image"]
        image_embedding = self.encoder(image)
        image_embedding = image_embedding[0]

        seqs = self.old_beam_search(image_embedding)
        if self.use_diverse_beam_search:
            seqs += self.diverse_beam_search(
                image_embedding, num_groups=self.div_beam_s_group, diversity_strength=-0.2, beam_size=10
            )

        seqs = list(dict.fromkeys(seqs))
        lb_seqs = []
        for i in seqs:
            if i in self.label_mapping:
                lb_seqs.append({"id": i, "txt": self.label_mapping[i]})

        return lb_seqs

    def diverse_beam_search(
        self, image_embedding, num_groups=5, diversity_strength=-0.2, beam_size=10, convert_outputs=True
    ):

        seqs = torch.ones([beam_size, 1], dtype=torch.int64).to(image_embedding.device.index)  # (k, 1)
        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        encoder_dim = image_embedding.size(2)
        # Flatten encoding
        num_pixels = image_embedding.size(1)
        # We'll treat the problem as having a batch size of k
        image_embedding = image_embedding.expand(beam_size, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        decoder_inp = torch.ones([beam_size, 1], dtype=torch.int64).to(image_embedding.device.index)
        hidden = self.decoder.reset_state(beam_size).to(image_embedding.device.index)

        final_scores = torch.zeros(beam_size, 1).to(image_embedding.device.index)
        final_beams = torch.ones([beam_size, 1], dtype=torch.int64).to(image_embedding.device.index)
        final_indices = []

        for i_lev in range(len(self.classifier_config)):
            lev_dictionary = self.classifier_config[i_lev]["tokenizer"]
            predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)

            predictions_prob = torch.sigmoid(predictions)
            lprobs = torch.log(predictions_prob)
            scores = final_scores
            # diverse beam search
            beam_size, vocab_size = lprobs.size()

            # initialize diversity penalty
            diversity_buf = torch.zeros(lprobs[0, :].size()).to(lprobs)
            scores_G, indices_G, beams_G = [], [], []
            for g in range(num_groups):
                lprobs_g = lprobs[g::num_groups, :]
                scores_g = scores[g::num_groups]

                # apply diversity penalty
                lprobs_g = torch.add(
                    lprobs_g,
                    other=diversity_buf.unsqueeze(0),
                    alpha=diversity_strength,
                )

                scores_buf, indices_buf, beams_buf = self.simple_beam_search(i_lev, lprobs_g, scores_g)

                beams_buf.mul_(num_groups).add_(g)

                scores_G.append(scores_buf.clone())
                indices_G.append(indices_buf.clone())
                beams_G.append(beams_buf.clone())

                # update diversity penalty
                diversity_buf.scatter_add_(0, indices_buf, torch.ones(indices_buf.size()).to(diversity_buf))
            prev_word_inds = torch.stack(beams_G, dim=0).view(beam_size, -1).squeeze(1)
            # print(prev_word_inds)
            final_scores = torch.stack(scores_G, dim=0).view(beam_size, -1)
            next_word_inds = torch.stack(indices_G, dim=0).view(beam_size, -1).squeeze(1)
            # print(next_word_inds)
            final_beams = torch.cat([final_beams[prev_word_inds], next_word_inds.unsqueeze(1)], dim=-1)

            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
            )  # (s, step+1) #TODO skip the complete sequences
            # print(seqs)
            # Which sequences are incomplete (didn't reach <end>)?
            # incomplete_inds = [
            #     ind for ind, next_word in enumerate(next_word_inds) if next_word != lev_dictionary.index("#PAD")
            # ]
            # complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            # if len(complete_inds) > 0:
            #     complete_seqs.extend(seqs[complete_inds].tolist())
            #     complete_seqs_scores.extend(final_scores[complete_inds])
            # beam_size -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            # if beam_size == 0:
            #     break
            # seqs = seqs[incomplete_inds]
            hidden = hidden[prev_word_inds]
            image_embedding = image_embedding[prev_word_inds]
            decoder_inp = next_word_inds.unsqueeze(1)
        print("#############")
        print(final_beams)
        print(final_scores)
        if convert_outputs:
            final_beams = list(map(lambda l: l[1:], final_beams))
            seqs = self.seq2str(final_beams)
            seqs = sorted(seqs, key=len)
            return [j for i, j in enumerate(seqs) if all(j not in k for k in seqs[i + 1 :])]

        return final_beams, final_scores

    def simple_beam_search(self, step, lprobs, scores):
        beam_size, vocab_size = lprobs.size()
        lprobs = lprobs + scores
        top_prediction = torch.topk(lprobs.view(-1), beam_size, 0)
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf

    def old_beam_search(self, image_embedding, k=10, convert_outputs=True):

        #
        encoder_dim = image_embedding.size(2)

        # Flatten encoding
        num_pixels = image_embedding.size(1)

        # We'll treat the problem as having a batch size of k
        image_embedding = image_embedding.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        decoder_inp = torch.ones([k, 1], dtype=torch.int64).to(image_embedding.device.index)
        # Tensor to store top k previous words at each step; now they're just <start>
        # decoder_inp = torch.LongTensor([[self.dictionary["<start>"]]] * k).to(image_embedding.device.index)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = torch.ones([k, 1], dtype=torch.int64).to(image_embedding.device.index)  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(image_embedding.device.index)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        # step = 1
        # h, c = self.decoder.init_hidden_state(image_embedding)

        hidden = self.decoder.reset_state(k).to(image_embedding.device.index)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        # while True:

        for i_lev in range(len(self.classifier_config)):
            lev_dictionary = self.classifier_config[i_lev]["tokenizer"]
            predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)

            predictions_prob = torch.sigmoid(predictions)

            predictions_prob = top_k_scores.expand_as(predictions_prob) + predictions_prob
            # Get the top_k predictions
            if i_lev == 0:
                top_k_scores, top_k_words = predictions_prob[0].topk(k, 0, True, True)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = predictions_prob.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // len(lev_dictionary)  # vocab_size  # (s)
            next_word_inds = top_k_words % len(lev_dictionary)  # vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # print(seqs)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind for ind, next_word in enumerate(next_word_inds) if next_word != lev_dictionary.index("#PAD")
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds])
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            hidden = hidden[prev_word_inds[incomplete_inds]]
            # c = c[prev_word_inds[incomplete_inds]]
            image_embedding = image_embedding[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            decoder_inp = next_word_inds[incomplete_inds].unsqueeze(1)

        if len(complete_seqs_scores) == 0:
            seq = seqs[0]
        else:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

        if convert_outputs:
            seqs = list(map(lambda l: l[1:], seqs))
            return self.seq2str(seqs)
        return seq

    def seq2str(self, seqs):
        # reverse the indexes in dictionary to the string labels
        final_lbs = list()
        for seq in seqs:
            label_hirarchy = list()
            for i_lev in range(len(seq)):
                if seq[i_lev] != 0:
                    label_hirarchy.append(self.classifier_config[i_lev]["tokenizer"][seq[i_lev]])

            # print(''.join(label_hirarchy))
            final_lbs.append("".join(label_hirarchy))
        return final_lbs

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parent_parser = Encoder.add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--label_mapping_path", type=str)
        parser.add_argument("--outpath_f2_per_lbs", type=str)

        parser.add_argument("--use_diverse_beam_search", action="store_true", default=False)
        parser.add_argument("--div_beam_s_group", type=int, default=2)

        parser.add_argument("--use_label_smoothing", action="store_true", default=False)
        parser.add_argument("--using_weights", action="store_true", default=False)
        parser.add_argument("--use_focal_loss", action="store_true", default=False)
        parser.add_argument("--focal_loss_gamma", type=float, default=2)
        parser.add_argument("--focal_loss_alpha", type=float, default=0.25)

        parser.add_argument("--filter_label_by_count", type=int, default=0)

        parser.add_argument("--best_threshold", nargs="+", type=float, default=[0.2])
        return parser


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, attention_dim):
        """
        encoder_dim: feature size of encoded images
        decoder_dim: size of decoder's RNN
        """
        super(BahdanauAttention, self).__init__()
        self.w_enc = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.w_dec = nn.Linear(attention_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        encoder_out: encoded images, a tensor of dimension (batch_size, num_channels, encoder_dim)
        """
        decoder_hidden_with_time_axis = torch.unsqueeze(decoder_hidden, 1)  # (batch_size, 1, decoder_dim)

        attention_hidden_layer = self.tanh(self.w_enc(encoder_out) + self.w_dec(decoder_hidden_with_time_axis))
        attention_weights = self.softmax(self.full_att(attention_hidden_layer).squeeze(2))  # (batch_size, channels)

        attention_weighted_encoding = (encoder_out * attention_weights.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, attention_dim, encoder_dim, max_vocab_size, tf=False):
        super(Decoder, self).__init__()
        self.attention_dim = attention_dim
        self.embedding = nn.ModuleList(
            [nn.Embedding(max_vocab_size, embedding_dim) for x in range(len(vocabulary_size))]
        )
        self.gru = nn.GRUCell(embedding_dim + encoder_dim, attention_dim, bias=True)
        self.init_h = nn.Linear(embedding_dim, attention_dim)
        # Todo add more linear layers Todo add kernel regulizer
        self.fc1 = nn.Linear(attention_dim, attention_dim)

        self.fc3 = nn.ModuleList(
            [
                nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(attention_dim, vocabulary_size[i]))
                for i in range(len(vocabulary_size))
            ]
        )
        self.attention = BahdanauAttention(encoder_dim, attention_dim)
        # Todo add drop out layers

    def forward(self, x, encoder_out, hidden, level):
        """
        Forward propagation.
        encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        decoder_inp: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        #caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        context_vec, attention_weights = self.attention(encoder_out, hidden)

        if level < 9:
            x = self.embedding[level](x)

        x = torch.cat([context_vec, torch.squeeze(x, 1)], dim=1)

        output = self.gru(x, hidden)

        x = self.fc1(output)

        if level < 9:
            x = self.fc3[level](x)

        return x, output, attention_weights

    def reset_state(self, batch_size):
        return torch.zeros((batch_size, self.attention_dim))

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        return h
