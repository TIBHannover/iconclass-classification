#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:04:05 2021

@author: javad
"""
import torch
from models.utils import mapping_local_global_idx, check_seq_path
from typing import Tuple, Dict


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length):
        self.max_length = max_length-1 # ignoring bos_token
        self.num_beams = num_beams # beam size
        self.beams = [] # Store the optimal sequence and its accumulated log_prob score
        self.worst_score = 1e9 # Initialize worst_score to infinity.

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) # calculate the score after penalty
        if len(self) <self.num_beams or score> self.worst_score:
                # If the class is not filled with num_beams sequences
                # Or after it is full, but the score value of the sequence to be added is greater than the minimum value in the class
                # Then update the sequence into the class, and eliminate the worst sequence in the previous class
            self.beams.append((score, hyp))
            if len(self)> self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                                # If not full, only update worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
                # After decoding to a certain layer, the score of each node of this layer represents the sum of log_prob from the root node to here
                # Take the highest log_prob at this time, if the highest score of the candidate sequence at this time is lower than the lowest score in the class
                # Then there is no need to continue decoding. At this time, the decoding of the sentence is completed, and there are num_beams optimal sequences in the class.
        if len(self) <self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len 
            ret = self.worst_score >= cur_score
            return ret




class BeamSearchScorer():
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        max_length:int,
        # num_beam_hyps_to_keep: int = 1,
        device: torch.device,
        src: torch.LongTensor,
        mapping_global_indexes:Dict,
        # num_beam_groups: int = 1, ?
        **kwargs,
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        # self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        # self.num_beam_groups = num_beam_groups
        # self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                max_length=max_length,
                num_beams=self.num_beams
            )for _ in range(batch_size)]
        
        self.local_seq_src = torch.ones(src.shape,dtype = torch.int64).to(device)
        self.beam_scores = torch.zeros((batch_size, num_beams)).to(device)
        self.beam_scores[:,1:] = -1e9
        self.beam_scores = self.beam_scores.view(-1)
        self.done = [False for _ in range(batch_size)]
        self.mapping_global_indexes = mapping_global_indexes
        
    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        src: torch.LongTensor,
        # next_scores: torch.FloatTensor,
        next_token_logits: torch.LongTensor,
        # next_indices: torch.LongTensor,
        # pad_token_id: Optional[int] = None,
        # eos_token_id: Optional[int] = None,
        lb_level_dict:Dict,
        i_lev:int
    ) -> Tuple[torch.Tensor]:
        cur_len = src.shape[-1]
        batch_size = len(self._beam_hyps)
        # assert batch_size == (input_ids.shape[0] // self.group_size) This does not make sense! 
        
        next_token_scores = torch.sigmoid(next_token_logits)
        next_scores = next_token_scores + self.beam_scores[:,None].expand_as(next_token_scores)
        next_scores = next_scores.view(self.batch_size, self.num_beams*next_token_scores.shape[1])
        next_scores, next_tokens = torch.topk(next_scores, next_token_scores.shape[1]*self.num_beams, dim=1, largest=True, sorted=True) #should be top k, because of lack of correct paths, sort scores
        # if i_lev == 2:
        #     print('&&&&&&&&&&&&&&&&&&&&&&&&')
        #     print(next_scores)
        #     print(next_tokens)
        #     print('%%%%%%%%%%%%%%%check beams and tokens%%%%%%%%%%%%%%%%%%%55')
        #     check_beams =next_tokens//sig_pred.shape[1] + pos_ind
        #     print(check_beams)
        #     check_tokens = next_tokens%sig_pred.shape[1]
        #     print(check_tokens)
        #     print('&&&&&&&&&&&&&&&&&&&&&&&&')
        # (Score, token_id, beam_id)
        next_batch_beam = []
        for batch_idx in range(self.batch_size):
            if self.done[batch_idx]:
                # print("problem 1")
                next_batch_beam.extend([0,lb_level_dict.index("#PAD"), 0])
                continue
            next_sent_beam = [] #save triples(beam_token_scores)
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])):
                beam_id = torch.div(beam_token_id, next_token_scores.shape[1], rounding_mode='floor')
                token_id = beam_token_id % next_token_scores.shape[1]
                effective_beam_id = batch_idx * self.num_beams + beam_id
                # print(effective_beam_id)
                if token_id == lb_level_dict.index("#PAD"):
                    # print("problem 3")
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    self._beam_hyps[batch_idx].add(src[effective_beam_id], beam_token_score)
                elif check_seq_path(token_id, self.local_seq_src[effective_beam_id,1:], self.mapping_global_indexes):
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
    
                    if len(next_sent_beam) ==self.num_beams:
                        break
                    self.done[batch_idx] = self.done[batch_idx] or self._beam_hyps[batch_idx].is_done(
                                        next_scores[batch_idx].max().item(), i_lev+1) 
                    if all(self.done):
                        break
                else:
                    continue

            next_batch_beam.extend(next_sent_beam)
        beam_idx = src.new([x[2] for x in next_batch_beam])
        

        src = src[beam_idx, :]
        # print(src)
        
        self.local_seq_src= self.local_seq_src[beam_idx, :]
        next_local_src = torch.stack([x[1] for x in next_batch_beam]
                                                       ).unsqueeze(1).to(src.device.index)
        # print(local_seq_src)
        self.local_seq_src = torch.cat((self.local_seq_src, next_local_src), dim =1)
        next_global_src = mapping_local_global_idx(self.mapping_global_indexes,self.local_seq_src[:,1:])
        print('*************')
        print(next_global_src)

        src = torch.cat((src, torch.tensor(next_global_src, dtype =torch.int64
                                                       ).unsqueeze(1).to(src.device.index)), dim =1)
        # hidden = hidden[src[:,i_lev]] #TODO should we change the order of hidden state according to winning beams?
        print(src)


        
        self.beam_scores = self.beam_scores.new([x[0] for x in next_batch_beam])
            
        return src, beam_idx
        
        
        
        # device = input_ids.device
        # next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        # next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        # next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        # for batch_idx, beam_hyp in enumerate(self._beam_hyps):
        #     if self._done[batch_idx]:
        #         assert (
        #             len(beam_hyp) >= self.num_beams
        #         ), f"Batch can only be done if at least {self.num_beams} beams have been generated"
        #         assert (
        #             eos_token_id is not None and pad_token_id is not None
        #         ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
        #         # pad the batch
        #         next_beam_scores[batch_idx, :] = 0
        #         next_beam_tokens[batch_idx, :] = pad_token_id
        #         next_beam_indices[batch_idx, :] = 0
        #         continue

        #     # next tokens for this sentence
        #     beam_idx = 0
        #     for beam_token_rank, (next_token, next_score, next_index) in enumerate(
        #         zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
        #     ):
        #         batch_beam_idx = batch_idx * self.group_size + next_index
        #         # add to generated hypotheses if end of sentence
        #         if (eos_token_id is not None) and (next_token.item() == eos_token_id):
        #             # if beam_token does not belong to top num_beams tokens, it should not be added
        #             is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
        #             if is_beam_token_worse_than_top_num_beams:
        #                 continue
        #             beam_hyp.add(
        #                 input_ids[batch_beam_idx].clone(),
        #                 next_score.item(),
        #             )
        #         else:
        #             # add next predicted token since it is not eos_token
        #             next_beam_scores[batch_idx, beam_idx] = next_score
        #             next_beam_tokens[batch_idx, beam_idx] = next_token
        #             next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
        #             beam_idx += 1

        #         # once the beam for next step is full, don't add more tokens to it.
        #         if beam_idx == self.group_size:
        #             break

        #     if beam_idx < self.group_size:
        #         raise ValueError(
        #             f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
        #         )

        #     # Check if we are done so that we can save a pad step if all(done)
        #     self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
        #         next_scores[batch_idx].max().item(), cur_len
        #     )

        # return UserDict(
        #     {
        #         "next_beam_scores": next_beam_scores.view(-1),
        #         "next_beam_tokens": next_beam_tokens.view(-1),
        #         "next_beam_indices": next_beam_indices.view(-1),
        #     }
        # )