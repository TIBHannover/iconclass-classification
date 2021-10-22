import logging
import re
import torch
import torch.nn.functional as F
import torchvision
import argparse
from torch import nn
from typing import Dict, List

from decoders.decoders import DecodersManager

from tools.beam_search import BeamSearchScorer
from models.utils import mapping_local_global_idx, check_seq_path


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


class AttnRNNLevelWise(nn.Module):
    def __init__(self, embedding_size, embedding_dim, attention_dim, encoder_dim, vocabulary_sizes):
        """
        encoder_dim: feature size of encoded images
        decoder_dim: size of decoder's RNN
        """
        super(AttnRNNLevelWise, self).__init__()
        self.attention_dim = attention_dim
        self.embedding = nn.ModuleList(
            [nn.Embedding(embedding_size, embedding_dim) for x in range(len(vocabulary_sizes))]
        )
        self.gru = nn.GRUCell(embedding_dim + encoder_dim, attention_dim, bias=True)
        self.init_h = nn.Linear(embedding_dim, attention_dim)
        # Todo add more linear layers Todo add kernel regulizer
        self.fc1 = nn.Linear(attention_dim, attention_dim)

        self.fc3 = nn.ModuleList(
            [
                nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(attention_dim, vocabulary_sizes[i]))
                for i in range(len(vocabulary_sizes))
            ]
        )
        self.attention = BahdanauAttention(encoder_dim, attention_dim)

    def forward(self, x, encoder_out, hidden, level):
        """
        Forward propagation.
        encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        decoder_inp: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        #caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        context_vec, attention_weights = self.attention(encoder_out, hidden)

        x = self.embedding[level](x)
        # print(f"{context_vec.shape} {level} {x.shape}")
        x = torch.cat([context_vec, torch.squeeze(x, 1)], dim=1)

        output = self.gru(x, hidden)

        x = self.fc1(output)

        x = self.fc3[level](x)

        return x, output, attention_weights
    
    
    def reset_state(self, batch_size):
        return torch.zeros((batch_size, self.attention_dim))

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        return h


@DecodersManager.export("attn_rnn_level_wise")
class AttnRNNLevelWiseDecoder(nn.Module):
    def __init__(self, in_features, embedding_size, vocabulary_sizes, mapping_global_indexes, args=None, **kwargs):
        super(AttnRNNLevelWiseDecoder, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.decoder_dropout = dict_args.get("decoder_dropout", 0.5)
        self.decoder_embedding_dim = dict_args.get("decoder_embedding_dim", 256)
        self.decoder_attention_dim = dict_args.get("decoder_attention_dim", 128)

        self.in_features = in_features
        self.embedding_size = embedding_size
        self.vocabulary_sizes = vocabulary_sizes
        
        self.mapping_global_indexes = mapping_global_indexes
        
        self.model = AttnRNNLevelWise(
            self.embedding_size,
            self.decoder_embedding_dim,
            self.decoder_attention_dim,
            self.in_features,
            self.vocabulary_sizes,
        )
        
    def forward(self, context_vec, src):
        hidden = self.model.init_hidden_state(context_vec)
        print(src)
        outputs = []
        for i_lev in range(src.shape[1]):
            pred, hidden, _ = self.model(src[:, i_lev], context_vec, hidden, i_lev)

            outputs.append(pred)
            
        return outputs
    
    def test_final(self, context_vec, src, k, ontology):
        batch_size = context_vec.shape[0]
        #expand to beam size
        context_vec = torch.repeat_interleave(context_vec, k, dim=0)
        src = torch.repeat_interleave(src, k, dim=0)
        hidden = self.model.init_hidden_state(context_vec)
        bsearch = BeamSearchScorer(batch_size, k, len(ontology), 
                                   context_vec.device.index, src, 
                                   self.mapping_global_indexes
                                   )

        for i_lev in range(len(ontology)):
            lb_level_dict = ontology[i_lev]["tokenizer"]
            pred, hidden, _ = self.model(src[:,i_lev], context_vec, hidden, i_lev)
            src, beam_id = bsearch.process(src, pred, lb_level_dict, i_lev)
            print(src)
            
        


    
    # def test(self, context_vec, src, k, ontology):
    #     batch_size = context_vec.shape[0]
    #     #expand to beam size
    #     context_vec = torch.repeat_interleave(context_vec, k, dim=0)
    #     src = torch.repeat_interleave(src, k, dim=0)
        
    #     hidden = self.model.init_hidden_state(context_vec)
        
    #     # Tensor to store top k sequences; now they're just <start>
    #     seqs = torch.ones([context_vec.shape[0], 1], dtype=torch.int64).to(context_vec.device.index)  # (batch_size*k, 1)
    #     scores = torch.zeros([context_vec.shape[0], 1], dtype=torch.int64).to(context_vec.device.index)
    #     pos_ind = (torch.tensor(range(batch_size)) * k).view(-1, 1).to(context_vec.device.index)
    #     print(seqs.shape)
        
    #     # Lists to store completed sequences and scores
    #     complete_seqs = list()
    #     complete_seqs_scores = list()
        
    #     outputs_pred = []
    #     for i_lev in range(len(ontology)):
    #         lb_level_dict = ontology[i_lev]["tokenizer"]
    #         pred, hidden, _ = self.model(src, context_vec, hidden, i_lev)
    #         outputs_pred.append(pred)
            
    #         sig_pred = torch.sigmoid(pred)
    #         sig_pred = scores.repeat_interleave(sig_pred.shape[1], dim = 1) + sig_pred
    #         # Get the top_k predictions
    #         # if i_lev == 0:
    #         #     top_k_scores, top_k_words = sig_pred[0].topk(k, 0, True, True)
    #         # else:
    #             # Unroll and find top scores, and their unrolled indices
    #         topk_scores, topk_words = sig_pred.view(batch_size, -1).topk(k, 1, True, True)
    #         next_word_inds = (topk_words % sig_pred.shape[1]).view(batch_size*k,1)
    #         prev_word_inds = (topk_words // sig_pred.shape[1] + pos_ind.expand_as(topk_words)).view(batch_size*k,1)

    #         seqs= torch.cat([seqs[prev_word_inds.squeeze(1)], next_word_inds], dim =1)


            
    #         incomplete_inds = [
    #             ind for ind, v in enumerate(next_word_inds) if v != lb_level_dict.index("#PAD")
    #         ]
    #         complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            
    #         # Set aside complete sequences
    #         if len(complete_inds) > 0:
    #             complete_seqs.extend(seqs[complete_inds])
    #             complete_seqs_scores.extend(scores[complete_inds])
    #         k -= len(complete_inds)  # reduce beam length accordingly
            
    #         if k == 0: 
    #             break
            
    #         scores += topk_scores.view(batch_size*k , 1)
            
    #         seqs = seqs[incomplete_inds]
    #         hidden = hidden[prev_word_inds[incomplete_inds]]
    #         # c = c[prev_word_inds[incomplete_inds]]
    #         context_vec = context_vec[prev_word_inds[incomplete_inds]]
    #         top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
    #         decoder_inp = next_word_inds[incomplete_inds].unsqueeze(1)

    #     if len(complete_seqs_scores) == 0:
    #         seq = seqs[0]
    #     else:
    #         i = complete_seqs_scores.index(max(complete_seqs_scores))
    #         seq = complete_seqs[i]

    #     if convert_outputs:
    #         seqs = list(map(lambda l: l[1:], seqs))
    #         return self.seq2str(seqs)
            
            
    #         # print(scores)
    #         # print(topk_words)
    #         exit()
    #         # sig_pred = torch.tensor(sig_pred>0.15, dtype=torch.int32)
            
    #         # sig_pred.view(args.batch_size,args.ma_trace, -1) 
            
    #         # idx = (sig_pred == 1).nonzero()
            

            
        #     outputs.append(pred)
        # return outputs
    
    def beam_search(self):
        print('etwas')
    
    @classmethod
    def add_args(cls, parent_parser):
        logging.info("Add FlatDecoder args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--decoder_dropout", type=float, default=0.5)
        parser.add_argument("--decoder_embedding_dim", type=int, default=256)
        parser.add_argument("--decoder_attention_dim", type=int, default=128)

        return parser