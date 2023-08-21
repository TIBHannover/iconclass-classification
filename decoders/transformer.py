import logging
import re
import torch
import torch.nn.functional as F
import torchvision
import argparse
from torch import nn
from typing import Dict, List

from decoders.decoders import DecodersManager

from models import utils


@DecodersManager.export("transformer_level_wise")
class TransformerLevelWiseDecoder(nn.Module):
    def __init__(self, in_features, embedding_size, vocabulary_sizes, args=None, **kwargs):
        super(TransformerLevelWiseDecoder, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.d_model = dict_args.get("transformer_d_model")
        self.nhead = dict_args.get("transformer_nhead")
        self.num_encoder_layers = dict_args.get("transformer_num_encoder_layers")
        self.num_decoder_layers = dict_args.get("transformer_num_decoder_layers")
        self.dim_feedforward = dict_args.get("transformer_dim_feedforward")
        self.dropout = dict_args.get("transformer_dropout")

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        total_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"Number of parameters transformer: {total_params}", flush=True)

        self.hidden_dim = self.transformer.d_model

        # self.embeddings = nn.ModuleList(
        #     [nn.Embedding(embedding_size, self.hidden_dim) for x in range(len(vocabulary_sizes))]
        # )

        self.embeddings = nn.Embedding(embedding_size, self.hidden_dim)

        total_params = sum(p.numel() for p in self.embeddings.parameters())
        print(f"Number of parameters embeddings: {total_params}", flush=True)

        self.classifiers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, vocabulary_sizes[i]) for i in range(len(vocabulary_sizes))]
        )

        total_params = sum(p.numel() for p in self.classifiers.parameters())
        print(f"Number of parameters classifiers: {total_params}", flush=True)

        # Todo add drop out layers

    def build_src(self, inputs):
        assert "ontology_indexes" in inputs, ""

        src = inputs["ontology_indexes"]
        # add pad and start
        src_level_with_tokens = utils.add_sequence_tokens_to_index(src, add_start=True)[:, :, :-1]

        # flat traces to batch
        src_level_with_tokens = src_level_with_tokens.reshape(-1, src_level_with_tokens.shape[-1])

        return {"src": src_level_with_tokens}

    def forward(self, inputs):  # image_embedding, mask, query, pos_embedding):
        assert "image_embedding" in inputs, ""
        num_traces = 1
        if "ontology_indexes" in inputs:
            src = self.build_src(inputs)["src"]
            # src is padded input [BS*NUM_TRACES,NUM_LEVEL+1]
            # if "ontology_indexes" in inputs:
            num_traces = inputs["ontology_indexes"].shape[1]
        image_embedding = torch.repeat_interleave(inputs["image_embedding"], num_traces, dim=0)

        # print(level)
        # print(x.shape)
        # for x in
        query_embedding = []
        for level in range(src.shape[1]):
            # print("################", level, src[:, level], flush=True)
            # query_embedding.append(self.embeddings[level](src[:, level]))

            query_embedding.append(self.embeddings(src[:, level]))
        query_embedding = torch.stack(query_embedding)

        image_embedding = image_embedding.permute(1, 0, 2)

        tgt_mask = self.transformer.generate_square_subsequent_mask(query_embedding.shape[0]).to(image_embedding.device)

        # image_embedding [H*W/16, BS, FEATURE_SIZE]
        # query_embedding [NUM_LEVEL+1, BS, FEATURE_SIZE]
        decoder_output = self.transformer(
            src=image_embedding,
            # src_key_padding_mask=~mask,
            tgt=query_embedding,
            tgt_mask=tgt_mask,
        )

        # print(decoder_output.shape)
        decoder_output = decoder_output.permute(1, 0, 2)
        # print(decoder_output.shape)
        classfier_results = []
        for level in range(src.shape[1]):
            x = self.classifiers[level](decoder_output[:, level, :]) * (src[:, level] != 0).unsqueeze(1)
            # print(x.shape)
            classfier_results.append(x)

        decoder_without_tokens = utils.del_sequence_tokens_from_level_ontology(classfier_results)

        # flat output (similar to yolo)
        flat_prediction = utils.map_to_flat_ontology(decoder_without_tokens, inputs["ontology_levels"])

        return {"classifier": classfier_results, "prediction": flat_prediction}

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("Add FlatDecoder args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--transformer_d_model", type=int, default=512)
        parser.add_argument("--transformer_nhead", type=int, default=8)
        parser.add_argument("--transformer_num_encoder_layers", type=int, default=0)
        parser.add_argument("--transformer_num_decoder_layers", type=int, default=3)
        parser.add_argument("--transformer_dim_feedforward", type=int, default=2048)
        parser.add_argument("--transformer_dropout", type=float, default=0.1)
        return parser
