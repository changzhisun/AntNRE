#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/01/05 23:00:16

@author: Changzhi Sun
"""
import sys
sys.path.append("..")
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from antNRE.src.word_encoder import WordCharEncoder
from antNRE.modules.seq2seq_encoders.seq2seq_bilstm import BiLSTMEncoder
from antNRE.src.seq_decoder import SeqSoftmaxDecoder
from antNRE.src.decoder import VanillaSoftmaxDecoder
from antNRE.lib.vocabulary import Vocabulary
from antNRE.lib.util import parse_tag
from antNRE.lib.util import start_of_chunk
from antNRE.lib.util import end_of_chunk
#  from src.rel_encoder import RelFeatureExtractor
from allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor import BidirectionalEndpointSpanExtractor
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor

class EntSpanGenerator:

    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    def forward(self,
                batch: Dict[str, Any],
                all_ent_span_pred: torch.LongTensor,
                training: bool) -> (List[List[Tuple]], List[List[int]]):
        all_ent_ids = []
        all_ent_ids_label = []
        for i in range(len(batch['tokens'])):
            ent_span_pred = all_ent_span_pred[i]
            seq_len = batch['seq_lens'][i]
            gold_ent_ids = batch['ent_ids'][i]
            gold_ent_ids_label = batch['ent_ids_labels'][i]

            gold_dict = {k:v for k, v in zip(gold_ent_ids, gold_ent_ids_label)}
            cur_ent_ids, cur_ent_ids_label = self.generate_span_from_tags(
                ent_span_pred[:seq_len], gold_dict, training)
            cur_ent_ids = [(e[0], e[-1]-1) for e in cur_ent_ids]

            all_ent_ids.append(cur_ent_ids)
            all_ent_ids_label.append(cur_ent_ids_label)
        return all_ent_ids, all_ent_ids_label

    def generate_span_from_tags(self,
                                ent_span_labels: torch.LongTensor,
                                gold_dict: Dict[Tuple, int],
                                training: bool) -> (List[Tuple], List[int]):
        cur_ent_ids = []
        cur_ent_ids_label = []
        y = [self.vocab.get_token_from_index(t.item(), "ent_span_labels")
             for t in ent_span_labels]
        y = [ yy if yy == "O" else yy + "-ENT" for yy in y]
        t_entity = self.get_entity(y)
        all_ent_set = set(t_entity['ENT'])
        #  if training:
            #  gold_ent_set = set(gold_dict.keys())
            #  all_ent_set = all_ent_set | gold_ent_set
        sort_all_ent_set = sorted(all_ent_set, key=lambda x: x[0])
        for e in sort_all_ent_set:
            cur_ent_ids.append(e)
            if e in gold_dict:
                cur_ent_ids_label.append(gold_dict[e])
            else:
                cur_ent_ids_label.append(self.vocab.get_token_index("None", "ent_ids_labels"))
        return cur_ent_ids, cur_ent_ids_label

    def get_entity(self, y: List[str]) -> Dict[str, list]:
        last_guessed = 'O'        # previously identified chunk tag
        last_guessed_type = ''    # type of previous chunk tag in corpus
        guessed_idx = []
        t_guessed_entity2idx = defaultdict(list)
        for i, tag in enumerate(y):
            guessed, guessed_type = parse_tag(tag)
            start_guessed = start_of_chunk(last_guessed, guessed,
                                           last_guessed_type, guessed_type)
            end_guessed = end_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)
            if start_guessed:
                if guessed_idx:
                    t_guessed_entity2idx[guessed_idx[0]].append((guessed_idx[1], guessed_idx[-1]+1))
                guessed_idx = [guessed_type, i]
            elif guessed_idx and not start_guessed and guessed_type == guessed_idx[0]:
                guessed_idx.append(i)

            last_guessed = guessed
            last_guessed_type = guessed_type
        if guessed_idx:
            t_guessed_entity2idx[guessed_idx[0]].append((guessed_idx[1], guessed_idx[-1]+1))
        return t_guessed_entity2idx
