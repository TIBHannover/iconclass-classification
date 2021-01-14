import re
import argparse

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

import pytorch_lightning as pl
from torchvision.models import resnet50, resnet152, densenet161, inception_v3
from models.models import ModelsManager

from models.resnet import ResNet50

from models.base_model import BaseModel
from datasets.utils import read_jsonl

from models.loss import FocalBCEWithLogitsLoss

from pytorch_lightning.core.decorators import auto_move_data


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

        self.use_focal_loss = dict_args.get("use_focal_loss", None)

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

        self.max_level = len(self.classifier_config)

        self.vocabulary_size = [len(x["tokenizer"]) for x in self.classifier_config]  # get from tockenizer
        self.max_vocab_size = max(self.vocabulary_size)
        self.embedding_dim = 128
        self.attention_dim = 64
        # self.max_vocab_size = max(self.vocabulary_size)
        self.encoder = Encoder(network=self.encoder_model, embedding_dim=self.embedding_dim, pretrained=True)
        self.encoder_dim = self.encoder.dim
        self.decoder = Decoder(
            self.vocabulary_size, self.embedding_dim, self.attention_dim, self.embedding_dim, self.max_vocab_size
        )

        if self.use_focal_loss:
            self.loss = FocalBCEWithLogitsLoss(reduction="none")
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.byol_embedding_path = dict_args.get("byol_embedding_path", None)

        if self.byol_embedding_path is not None:
            self.load_pretrained_byol(self.byol_embedding_path)

        self.f1_val = pl.metrics.classification.F1(num_classes=len(self.mapping_config), multilabel=True, average=None)

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        source = batch["source_id_sequence"]
        target = batch["target_vec"]
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # print(image.shape)
        # forward image
        image_embedding = self.encoder(image)
        # print('*********')
        # print(image_embedding.shape)
        # return loss
        hidden = self.decoder.reset_state(image.shape[0]).to(image.device.index)
        # print(hidden.device)

        # Feed <START> to the model in the first layer 1==<START>
        decoder_inp = torch.ones([image.shape[0], 1], dtype=torch.int64).to(image.device.index)
        # print('#########################')
        # print(decoder_inp)
        loss = 0
        predictions_list = []

        for i_lev in range(len(target)):
            predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)
            # print(f"Pre: {torch.min(predictions)} {torch.max(predictions)} {torch.mean(predictions)}")
            # print(
            #     f"Post: {torch.min(torch.sigmoid(predictions))} {torch.max(torch.sigmoid(predictions))} {torch.mean(torch.sigmoid(predictions))}"
            # )
            predictions_list.append(torch.sigmoid(predictions))
            # print('#########################')
            # print(predictions.shape)
            # print('#########################')
            # print(target[i_lev].shape)
            loss += torch.mean(self.loss(predictions, target[i_lev]))
            decoder_inp = torch.unsqueeze(source[i_lev], dim=1)
            # print(decoder_inp.shape)
            # decoder_inp = torch.tensor(target[i_lev]).to(torch.int64).to(image.device.index)
        # print('#########################')
        # print(predictions.shape)
        total_loss = loss / len(target)

        # print(torch.mean(loss))
        loss = total_loss
        return {"loss": torch.mean(loss), "predictions": predictions_list, "targets": target}

    def training_step_end(self, outputs):
        self.log("train/loss", outputs["loss"].mean(), prog_bar=True)
        if (self.global_step % self.trainer.log_every_n_steps) == 0:
            for i, (pred, target) in enumerate(zip(outputs["predictions"], outputs["targets"])):
                self.logger.experiment.add_histogram(f"predict_{i}", pred, self.global_step)
                self.logger.experiment.add_histogram(f"target_{i}", target, self.global_step)

        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idx):
        # print(batch.keys())
        # print(batch["parents"])
        image = batch["image"]
        source = batch["source_id_sequence"]
        target = batch["target_vec"]
        parents = batch["parents"]
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # print(image.shape)
        # forward image
        image_embedding = self.encoder(image)
        # print('*********')
        # print(image_embedding.shape)
        # return loss
        hidden = self.decoder.reset_state(image.shape[0]).to(image.device.index)
        # print(hidden.device)

        # Feed <START> to the model in the first layer 1==<START>
        decoder_inp = torch.ones([image.shape[0], 1], dtype=torch.int64).to(image.device.index)
        # print('#########################')
        # print(decoder_inp)
        loss = 0
        # Check if batch contains all traces (target [BATCH_SIZE, MAX_SEQUENCE, LEVEL, MAX_CLASSIFIER])
        if "mask" in batch:
            for i_lev in range(target.shape[2]):
                predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)

                prediction_size = len(self.classifier_config[i_lev]["tokenizer"])

                target_lev = target[:, 0, i_lev, :prediction_size]
                loss += torch.mean(self.loss(predictions, target_lev))
                decoder_inp = torch.unsqueeze(source[:, 0, i_lev], dim=1)

        else:

            flat_prediction = torch.zeros(image.shape[0], len(self.mapping_config), dtype=image_embedding.dtype).to(
                image.device.index
            )
            flat_target = torch.zeros(image.shape[0], len(self.mapping_config), dtype=target[0].dtype).to(
                image.device.index
            )
            parents_lvl = [None] * image.shape[0]
            for i_lev in range(len(target)):
                predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)

                loss += torch.mean(self.loss(predictions, target[i_lev]))
                decoder_inp = torch.unsqueeze(source[i_lev], dim=1)

                source_indexes, target_indexes = self.map_level_prediction(parents_lvl)

                flat_prediction[target_indexes] = torch.sigmoid(predictions)[source_indexes]
                flat_target[target_indexes] = target[i_lev][source_indexes]

                parents_lvl = parents[i_lev]

            self.f1_val(flat_prediction, flat_target)

        return {
            "loss": loss,
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
        # print(source_indexes.shape)
        # print(target_indexes.shape)
        # print(self.mapping_lut[None])
        return np.swapaxes(source_indexes, 0, 1), np.swapaxes(target_indexes, 0, 1)

    def validation_epoch_end(self, outputs):

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1

        f1_score = self.f1_val.compute().cpu().detach()

        self.log("val/f1", np.nanmean(f1_score), prog_bar=True)

        f1_score = self.f1_val.compute().cpu().detach()

        self.log("val/f1", np.nanmean(f1_score), prog_bar=True)

        level_results = {}
        for i, x in enumerate(self.mapping_config):
            if len(x["parents"]) not in level_results:
                level_results[len(x["parents"])] = []
            level_results[len(x["parents"])].append(f1_score[x["index"]])
        for depth, x in sorted(level_results.items(), key=lambda x: x[0]):
            self.log(f"val/f1_{depth}", np.nanmean(torch.stack(x, dim=0)), prog_bar=True)

        self.log("val/loss", loss / count, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    @auto_move_data
    def infer_step(self, batch, k=10):
        image = batch["image"]
        image_embedding = self.encoder(image)

        self.beam_search(image_embedding)

    def beam_search(self, image_embedding, k=10):

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
        seqs = decoder_inp  # (k, 1)

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
            # print(predictions_prob.shape)
            # print(predictions_prob)

            predictions_prob = top_k_scores.expand_as(predictions_prob) + predictions_prob  # (s, vocab_size)
            # print(predictions_prob.shape)
            # print(predictions_prob)
            # Get the top_k predictions
            if i_lev == 0:
                top_k_scores, top_k_words = predictions_prob[0].topk(k, 0, True, True)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = predictions_prob.view(-1).topk(k, 0, True, True)  # (s)

            # print(top_k_scores)

            # print(top_k_words)

            # embeddings = self.decoder.embedding(decoder_inp).squeeze(1)  # (s, embed_dim)

            # awe, _ = self.decoder.attention(image_embedding, h)  # (s, encoder_dim), (s, num_pixels)

            # gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            # awe = gate * awe

            # h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            # scores = self.decoder.fc(h)  # (s, vocab_size)
            # scores = F.log_softmax(scores, dim=1)
            # print(scores)
            # Add
            # scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # # For the first step, all k points will have the same scores (since same k previous words, h, c)
            # if step == 1:
            #     top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            # else:
            #     # Unroll and find top scores, and their unrolled indices
            #     top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // len(lev_dictionary)  # vocab_size  # (s)
            next_word_inds = top_k_words % len(lev_dictionary)  # vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind for ind, next_word in enumerate(next_word_inds) if next_word != lev_dictionary.index("#PAD")
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
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
            # print(top_k_scores)
            # Break if things have been going on too long
            # if step > 500:
            #     break
            # step += 1

        if len(complete_seqs_scores) == 0:
            seq = seqs[0].tolist()
        else:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

        print(seqs)
        print()

        # # References
        # img_caps = allcaps[0].tolist()
        # img_captions = list(
        #     map(lambda c: [w for w in c if w not in {self.dictionary['<start>'], self.dictionary['<end>'], self.dictionary['<pad>']}],
        #         img_caps))  # remove <start> and pads
        # references.append(img_captions)

        # Hypotheses

        result_idx = [
            w for w in seq if w not in {self.dictionary["<start>"], self.dictionary["<end>"], self.dictionary["<pad>"]}
        ]
        result_str = [self.inv_dictionary[w] for w in result_idx]

        gt = batch["sequence"].squeeze(0).cpu().numpy().tolist()

        gt_idx = [
            w for w in gt if w not in {self.dictionary["<start>"], self.dictionary["<end>"], self.dictionary["<pad>"]}
        ]
        gt_str = [self.inv_dictionary[w] for w in gt_idx]

        # print("########")
        # print(gt_str)
        # print(result_str)
        # if self.params.test.prediction_output_path is not None:
        #     image_out = os.path.join(self.params.test.prediction_output_path, "img")
        #     gt_out = os.path.join(self.params.test.prediction_output_path, "gt")
        #     res_out = os.path.join(self.params.test.prediction_output_path, "res")

        #     os.makedirs(image_out, exist_ok=True)
        #     os.makedirs(gt_out, exist_ok=True)
        #     os.makedirs(res_out, exist_ok=True)

        #     filename = os.path.splitext(os.path.basename(batch["path"][0]))[0]

        #     with open(os.path.join(gt_out, f"{filename}.tex"), "w") as f:
        #         f.write("$" + " ".join(gt_str) + "$\n")

        #     with open(os.path.join(res_out, f"{filename}.tex"), "w") as f:
        #         f.write("$" + " ".join(result_str) + "$\n")

        #     imageio.imwrite(
        #         os.path.join(image_out, f"{filename}.jpg"), batch["image"].squeeze(0).squeeze(0).cpu().numpy()
        #     )

        return {"loss": loss, "perplexity": perplexity, "gt_str": gt_str, "pred_str": result_str}

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument(
            "--encoder_model", choices=("resnet152", "densenet161", "resnet50", "inceptionv3"), default="resnet50"
        )
        parser.add_argument("--byol_embedding_path", type=str)
        parser.add_argument("--use_focal_loss", action="store_true")

        return parser

    def load_pretrained_byol(self, path_checkpoint):
        assert self.encoder_model == "resnet50", "BYOL currently working with renset50"
        data = torch.load(path_checkpoint)["state_dict"]

        load_dict = {}
        for name, var in data.items():
            if "model.target_net.0._features" in name:
                new_name = re.sub("^model.target_net.0._features.", "", name)
                load_dict[new_name] = var
        # TODO move this to the encoder
        self.encoder.net.load_state_dict(load_dict)


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
        # print('decoder_hidden_with_time_axis.shape {}'.format(decoder_hidden_with_time_axis.shape))
        # print('encoder_out.shape{}'.format(encoder_out.shape))
        # print('hidden {}'.format(decoder_hidden.shape))

        attention_hidden_layer = self.tanh(
            self.w_enc(encoder_out) + self.w_dec(decoder_hidden_with_time_axis)
        )  # (batch_size,channel, attention_dim)
        # print('score {}'.format(attention_hidden_layer.shape))
        attention_weights = self.softmax(self.full_att(attention_hidden_layer).squeeze(2))  # (batch_size, channels)

        attention_weighted_encoding = (encoder_out * attention_weights.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, encoder_dim)
        # print('context vector {}'.format(attention_weighted_encoding.shape))
        # print('attntion weights {}'.format(attention_weights.shape))
        return attention_weighted_encoding, attention_weights


class Encoder(nn.Module):
    def __init__(self, network="resnet152", embedding_dim=128, pretrained=True):
        super(Encoder, self).__init__()
        self.network = network
        if network == "resnet152":
            self.net = resnet152(pretrained=pretrained, progress=False)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == "densenet161":
            self.net = densenet161(pretrained=pretrained, progress=False)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-2])
            self.dim = 1920
        elif network == "resnet50":
            self.net = resnet50(pretrained=pretrained, progress=False)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == "inceptionv3":  # TODO:: fix the input dimension of images
            self.net = inception_v3(pretrained=pretrained, progress=False)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        self.embedding_dim = embedding_dim
        self._fc = nn.Linear(self.dim, self.embedding_dim)  # Todo add layers
        self._conv1 = torch.nn.Conv2d(self.dim, self.embedding_dim, kernel_size=[1, 1])

    def forward(self, x):
        x = self.net(x)
        x = self._conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, attention_dim, encoder_dim, max_vocab_size, tf=False):
        super(Decoder, self).__init__()
        self.attention_dim = attention_dim
        self.embedding = nn.ModuleList(
            [nn.Embedding(max_vocab_size, embedding_dim) for x in range(len(vocabulary_size))]
        )
        self.gru = nn.GRU(embedding_dim + encoder_dim, attention_dim, bias=True)
        # Todo add more linear layers Todo add kernel regulizer
        self.fc1 = nn.Linear(attention_dim, attention_dim)
        self.fc2 = nn.ModuleList([nn.Linear(attention_dim, attention_dim) for i in range(len(vocabulary_size))])
        self.fc3 = nn.ModuleList([nn.Linear(attention_dim, vocabulary_size[i]) for i in range(len(vocabulary_size))])
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

        # print(level)
        # print(x.shape)
        if level < 9:
            x = self.embedding[level](x)
        # else:??
        #    x = self.embedding[7](decoder_inp)
        # print(x.shape)
        # print(context_vec.shape)
        x = torch.cat([context_vec.unsqueeze(1), x], dim=2)
        # print('before gru {}'.format(x.shape))
        output, state = self.gru(x.permute(1, 0, 2))
        # print('output after gru {}'.format(output.shape))

        output = output.permute(1, 0, 2)
        state = state.squeeze(0)
        # print('state after gru {}'.format(state.shape))
        x = self.fc1(output)
        x = torch.reshape(x, (-1, x.size()[2]))
        # print('after gru reshape {}'.format(x.shape))
        if level < 9:
            x = self.fc2[level](x)
            x = self.fc3[level](x)
        # else:???
        #    x = self.fc2[7](x)
        #    x = self.fc3[7](x)
        # print('output of decoder{}'.format(x.shape))
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return torch.zeros((batch_size, self.attention_dim))
