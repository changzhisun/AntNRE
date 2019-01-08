#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/12 16:53:03

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
from src.ent_span_generator import EntSpanGenerator
from src.rel_feat_extractor import RelFeatExtractor
from src.ent_span_feat_extractor import EntSpanFeatExtractor


class JointModel(nn.Module):

    def __init__(self,
                 word_encoder: WordCharEncoder,
                 seq2seq_encoder: BiLSTMEncoder,
                 ent_span_decoder: SeqSoftmaxDecoder,
                 ent_span_feat_extractor: EntSpanFeatExtractor,
                 ent_ids_decoder: VanillaSoftmaxDecoder,
                 rel_feat_extractor: RelFeatExtractor,
                 rel_decoder: VanillaSoftmaxDecoder,
                 vocab: Vocabulary,
                 use_cuda: bool) -> None:
        super(JointModel, self).__init__()
        self.word_encoder = word_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.ent_span_decoder = ent_span_decoder
        self.ent_span_feat_extractor = ent_span_feat_extractor
        self.ent_ids_decoder = ent_ids_decoder
        self.rel_feat_extractor = rel_feat_extractor
        self.rel_decoder = rel_decoder
        self.vocab = vocab
        self.use_cuda = use_cuda
        self.ent_span_generator = EntSpanGenerator(vocab)

    def forward(self, batch: Dict[str, Any]) -> Dict:
        token_inputs = batch['tokens']
        char_inputs = batch['token_chars']
        seq_tags = batch['ent_span_labels']
        seq_lens = batch['seq_lens']

        encoder_inputs = self.word_encoder(token_inputs, char_inputs)
        seq_feats = self.seq2seq_encoder(encoder_inputs, seq_lens).contiguous()
        ent_span_outputs = self.ent_span_decoder(seq_feats, seq_tags)

        all_ent_ids, all_ent_ids_label = self.ent_span_generator.forward(
            batch, ent_span_outputs, self.training)
        batch['all_ent_ids'] = all_ent_ids  # batch_size, ent_span_num, 2
        batch['all_ent_ids_label'] = all_ent_ids_label # batch_size, ent_span_num, 1
        batch['seq_feats'] = seq_feats

        outputs = {}
        outputs['ent_span_loss'] = ent_span_outputs['loss']
        outputs['ent_span_pred'] = ent_span_outputs['predict']

        ent_ids_num = sum([len(e) for e in batch['all_ent_ids']])
        if ent_ids_num != 0:
            ent_ids_batch, ent_ids_span_feats = self.ent_span_feat_extractor(batch)

            ent_ids_outputs = self.ent_ids_decoder(ent_ids_span_feats, 
                                                   ent_ids_batch['all_ent_ids_label'])
            all_ent_pred = self.create_all_ent_pred(batch, ent_ids_outputs)
            #  print("ent_ids_num=", ent_ids_num)
            all_candi_rels, all_rel_labels = self.generate_all_candidates(
                batch, ent_ids_outputs['predict'])

            candi_rel_num = sum([len(e) for e in all_candi_rels])

            outputs['ent_ids_loss'] = ent_ids_outputs['loss']
            outputs['all_ent_pred'] = all_ent_pred
        
            if candi_rel_num != 0:
                #  print("candi_rel_num", candi_rel_num)
                batch['all_candi_rels'] = all_candi_rels
                batch['all_rel_labels'] = all_rel_labels
                rel_batch = self.rel_feat_extractor(batch, ent_ids_span_feats)

                rel_outputs = self.rel_decoder(rel_batch['inputs'],
                                               rel_batch['all_rel_labels'])
                outputs['rel_loss'] = rel_outputs['loss']

                all_rel_pred = self.create_all_rel_pred(all_candi_rels, rel_outputs['predict'])
                outputs['all_rel_pred'] = all_rel_pred
                outputs['all_candi_rels'] = []
                for cur_candi_rels in all_candi_rels:
                    t_candi_rels = [((e1[0], e1[1]+1), (e2[0], e2[1]+1))
                                    for e1, e2 in cur_candi_rels]
                    outputs['all_candi_rels'].append(t_candi_rels)
            else:
                zero_loss = torch.Tensor([0])
                zero_loss.requires_grad = True
                if self.use_cuda:
                    zero_loss = zero_loss.cuda(non_blocking=True)

                batch_size, _ = batch['tokens'].size()
                outputs['rel_loss'] = zero_loss
                outputs['all_rel_pred'] = [[] for _ in range(batch_size)]
                outputs['all_candi_rels'] = [[] for _ in range(batch_size)]

        else:
            zero_loss = torch.Tensor([0])
            zero_loss.requires_grad = True
            if self.use_cuda:
                zero_loss = zero_loss.cuda(non_blocking=True)
            batch_size, seq_size = batch['tokens'].size()
            batch_tag = [[self.vocab.get_token_index("O", "ent_labels") for _ in range(seq_size)]
                         for _ in range(batch_size)]
            outputs['ent_ids_loss'] = zero_loss
            outputs['all_ent_pred'] = batch_tag
            outputs['rel_loss'] = zero_loss
            outputs['all_rel_pred'] = [[] for _ in range(batch_size)]
            outputs['all_candi_rels'] = [[] for _ in range(batch_size)]
        return outputs

    def create_all_rel_pred(self, 
                            all_candi_rels: List[List[Tuple[Tuple, Tuple]]],
                            predict: torch.LongTensor) -> List[List[int]]:
        all_rel_pred = []
        j = 0
        for i in range(len(all_candi_rels)):
            candi_num = len(all_candi_rels[i])
            all_rel_pred.append(predict[j: j + candi_num].cpu().numpy())
            j += candi_num
        return all_rel_pred

    def create_all_ent_pred(self,
                            batch: Dict[str, Any],
                            ent_ids_outputs: Dict[str, Any]) -> List[List]:
        all_ent_pred = []
        j = 0
        for i in range(len(batch['all_ent_ids'])):
            seq_len = batch['seq_lens'][i]
            cur_ent_pred = [self.vocab.get_token_index("O", "ent_labels")
                            for _ in range(seq_len)]
            cur_ent_ids = batch['all_ent_ids'][i]
            cur_ent_ids_num = len(cur_ent_ids)
            cur_ent_ids_pred = ent_ids_outputs['predict'][j: j + cur_ent_ids_num].cpu().numpy()
            for (start, end), t in zip(cur_ent_ids, cur_ent_ids_pred):
                t_label_text = self.vocab.get_token_from_index(t, "ent_ids_labels")
                if t_label_text == "None":
                    continue
                if start == end:
                    start_label = self.vocab.get_token_index("U-%s" % t_label_text, "ent_labels")
                    cur_ent_pred[start] = start_label
                else:
                    start_label = self.vocab.get_token_index("B-%s" % t_label_text, "ent_labels")
                    end_label = self.vocab.get_token_index("E-%s" % t_label_text, "ent_labels")
                    cur_ent_pred[start] = start_label
                    cur_ent_pred[end] = end_label
                    for k in range(start + 1, end):
                        cur_ent_pred[k] = self.vocab.get_token_index("I-%s" % t_label_text, "ent_labels")
            j += cur_ent_ids_num
            all_ent_pred.append(cur_ent_pred)
        return all_ent_pred


    def generate_all_candidates(
            self,
            batch: Dict[str, Any],
            predict: torch.LongTensor) -> (List[List[Tuple[Tuple, Tuple]]], List[List[int]]):
        all_candi_rels = []
        all_rel_labels = []
        k = 0
        for i in range(len(batch['tokens'])):
            gold_candi_rels = batch['candi_rels'][i]
            gold_rel_labels = batch['rel_labels'][i]
            cur_ent_ids = batch['all_ent_ids'][i]

            gold_dict = {((k[0][0], k[0][1]-1), (k[1][0], k[1][1]-1)): v
                         for k, v in zip(gold_candi_rels, gold_rel_labels)}

            fiter_cur_ent_ids = []
            for j in range(len(cur_ent_ids)):
                cur_pred = predict[k].item()
                if self.vocab.get_token_from_index(cur_pred, "ent_ids_labels") != "None":
                    fiter_cur_ent_ids.append(cur_ent_ids[j])
                k += 1

            cur_candi_rels = []
            cur_rel_labels = []
            for e1 in fiter_cur_ent_ids:
                for e2 in fiter_cur_ent_ids:
                    if e1[0] >= e2[0]:
                        continue
                    cur_candi_rels.append((e1, e2))
                    if (e1, e2) in gold_dict:
                        cur_rel_labels.append(gold_dict[(e1, e2)])
                    else:
                        cur_rel_labels.append(self.vocab.get_token_index("None", "rel_labels"))
            all_candi_rels.append(cur_candi_rels)
            all_rel_labels.append(cur_rel_labels)
        return all_candi_rels, all_rel_labels
