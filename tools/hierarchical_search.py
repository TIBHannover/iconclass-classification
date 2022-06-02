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
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams  # beam size
        self.beams = []  # Store the optimal sequence and its accumulated log_prob score
        self.worst_score = 1e9  # Initialize worst_score to infinity.

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp)  # calculate the score after penalty
        if len(self) < self.num_beams or score > self.worst_score:
            # If the class is not filled with num_beams sequences
            # Or after it is full, but the score value of the sequence to be added is greater than the minimum value in the class
            # Then update the sequence into the class, and eliminate the worst sequence in the previous class
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
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
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len
            ret = self.worst_score >= cur_score
            return ret


class HierarchicalScorer:
    def __init__(
        self,
        device: torch.device,
        max_iter: 100,
        threshold: 0.1,
        mapper,
        **kwargs,
    ):
        self.max_iter = max_iter
        self.threshold = threshold
        self.mapper = mapper

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
        i_lev: int,
    ) -> Tuple[torch.Tensor]:
        print(f"{i_lev} ############### \n###################\n###################")
        cur_len = src.shape[-1]
        batch_size = len(self._beam_hyps)
        # assert batch_size == (input_ids.shape[0] // self.group_size) This does not make sense!

        mask = self.mapper.classifier_mask_from_parent(classifiers=src[:, -1])
        print("################## MASK")
        print(mask)
        print(mask.shape)
        next_token_scores = torch.sigmoid(next_token_logits)

        print("################## PRED")
        print(next_token_scores)
        print(next_token_scores.shape)
        print("################## FLAT")
        next_token_scores_flat = self.mapper.to_flat(next_token_scores, i_lev, remove_tokens=True)
        next_token_scores_flat = next_token_scores_flat * mask.to(next_token_scores_flat.device)
        result_pred = next_token_scores_flat
        print(next_token_scores_flat.shape)
        print(next_token_scores_flat)
        print("################## FLAT PADDED")
        next_token_scores = torch.cat(
            [
                next_token_scores[..., :1],
                torch.zeros([next_token_scores.shape[0], 1], device=next_token_scores.device),
                next_token_scores_flat,
            ],
            dim=1,
        )
        print(next_token_scores.shape)
        print(next_token_scores)
        print(self.beam_scores[:, None])

        print("****************** probs")
        print(next_token_scores)
        next_scores = next_token_scores * self.beam_scores[:, None].expand_as(next_token_scores)
        print(next_scores)
        next_scores = next_scores.view(self.batch_size, self.num_beams * next_token_scores.shape[1])
        next_scores, next_tokens = torch.topk(
            next_scores, next_token_scores.shape[1] * self.num_beams, dim=1, largest=True, sorted=True
        )  # should be top k, because of lack of correct paths, sort scores
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
        print("############### SCORES")
        print(next_scores)
        print(next_tokens)
        next_batch_beam = []
        for batch_idx in range(self.batch_size):
            print("#######################+++++++++")
            print(batch_idx)
            if self.done[batch_idx]:
                print("problem 1")
                next_batch_beam.append(
                    (
                        torch.tensor(0, device=next_scores.device),
                        torch.tensor(self.mapper.pad_id(), dtype=torch.int64, device=next_scores.device),
                        torch.tensor(0, dtype=torch.int64, device=next_scores.device),
                    )
                )
                continue
            next_sent_beam = []  # save triples(beam_token_scores)
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                print(f"token {beam_token_id}")
                beam_id = torch.div(beam_token_id, next_token_scores.shape[1], rounding_mode="floor")
                token_id = beam_token_id % next_token_scores.shape[1]
                effective_beam_id = batch_idx * self.num_beams + beam_id
                # print(effective_beam_id)
                if token_id == self.mapper.pad_id():
                    print("problem 3")
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    self._beam_hyps[batch_idx].add(src[effective_beam_id], beam_token_score)
                else:
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.num_beams:
                        break
                    self.done[batch_idx] = self.done[batch_idx] or self._beam_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), i_lev + 1
                    )
                    if all(self.done):
                        break
                # else:
                #     continue

            next_batch_beam.extend(next_sent_beam)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(src)
        print(next_sent_beam)
        print(next_batch_beam)
        beam_idx = src.new([x[2] for x in next_batch_beam])

        src = src[beam_idx, :]
        # print(src)

        self.local_seq_src = self.local_seq_src[beam_idx, :]
        next_local_src = torch.stack([x[1] for x in next_batch_beam]).unsqueeze(1).to(src.device.index)
        # print(local_seq_src)
        self.local_seq_src = torch.cat((self.local_seq_src, next_local_src), dim=1)

        print("#####################")
        print(src.shape)
        print(torch.tensor(self.local_seq_src, dtype=torch.int64).to(src.device.index).shape)
        # src = torch.cat((src, torch.tensor(self.local_seq_src, dtype=torch.int64).to(src.device.index)), dim=1)
        # hidden = hidden[src[:,i_lev]] #TODO should we change the order of hidden state according to winning beams?
        src = self.local_seq_src
        self.beam_scores = self.beam_scores.new([x[0] for x in next_batch_beam])

        return src, beam_idx, result_pred
