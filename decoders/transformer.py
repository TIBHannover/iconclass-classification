import logging
import re
import torch
import torch.nn.functional as F
import torchvision
import argparse
from torch import nn
from typing import Dict, List

from decoders.decoders import DecodersManager


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

        self.hidden_dim = self.transformer.d_model

        self.embeddings = nn.ModuleList(
            [nn.Embedding(embedding_size, self.hidden_dim) for x in range(len(vocabulary_sizes))]
        )

        self.classifiers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, vocabulary_sizes[i]) for i in range(len(vocabulary_sizes))]
        )

        # Todo add drop out layers

    def forward(
        self,
        context_vec,
        src,
    ):  # image_embedding, mask, query, pos_embedding):

        # print(level)
        # print(x.shape)
        # for x in
        query_embedding = []
        for level in range(src.shape[1]):
            query_embedding.append(self.embeddings[level](src[:, level]))
        query_embedding = torch.stack(query_embedding)

        # print("SHAPE 1")
        # print(image_embedding.shape)
        # print(query_embedding.shape)
        # print(pos_embedding.shape)
        # print(mask.shape)
        image_embedding = context_vec.permute(1, 0, 2)
        # mask = mask.flatten(1)

        # print("SHAPE 2")
        # print(image_embedding.shape)
        # print(pos_embedding.shape)
        # print(mask.shape)

        tgt_mask = self.transformer.generate_square_subsequent_mask(query_embedding.shape[0]).to(image_embedding.device)
        # , pos_embedding
        # image_embedding = image_embedding + pos_embedding  # TODO detr only apply this on q and k
        # image_mask has to be inverted (zero -> content)

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
            x = self.classifiers[level](decoder_output[:, level, :])
            # print(x.shape)
            classfier_results.append(x)
        return classfier_results

    def reset_state(self, batch_size):
        return torch.zeros((batch_size, self.attention_dim))

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("Add FlatDecoder args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--transformer_d_model", type=int, default=512)
        parser.add_argument("--transformer_nhead", type=int, default=8)
        parser.add_argument("--transformer_num_encoder_layers", type=int, default=6)
        parser.add_argument("--transformer_num_decoder_layers", type=int, default=6)
        parser.add_argument("--transformer_dim_feedforward", type=int, default=2048)
        parser.add_argument("--transformer_dropout", type=float, default=0.1)
        return parser