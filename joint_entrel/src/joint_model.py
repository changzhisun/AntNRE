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
from antNRE.src.seq_encoder import BiLSTMEncoder
from antNRE.src.seq_decoder import SeqSoftmaxDecoder
from antNRE.src.decoder import VanillaSoftmaxDecoder
from antNRE.lib.vocabulary import Vocabulary
from antNRE.lib.util import parse_tag
from antNRE.lib.util import start_of_chunk
from antNRE.lib.util import end_of_chunk
from src.rel_encoder import RelFeatureExtractor

class JointModel(nn.Module):

    def __init__(self,
                 word_encoder: WordCharEncoder,
                 seq_encoder: BiLSTMEncoder,
                 seq_decoder: SeqSoftmaxDecoder,
                 rel_feat_extractor: RelFeatureExtractor,
                 rel_decoder: VanillaSoftmaxDecoder,
                 sch_k: float,
                 vocab: Vocabulary,
                 use_cuda: bool) -> None:
        super(JointModel, self).__init__()
        self.word_encoder = word_encoder
        self.seq_encoder = seq_encoder
        self.seq_decoder = seq_decoder
        self.rel_feat_extractor = rel_feat_extractor
        self.rel_decoder = rel_decoder
        self.sch_k = sch_k
        self.vocab = vocab
        self.use_cuda = use_cuda

    def forward(self, batch: Dict[str, Any]) -> Dict:
        batch_seq_input = batch['tokens']
        batch_seq_char_input = batch['token_chars']
        batch_seq_tags = batch['ent_labels']
        batch_seq_len = batch['seq_lens']

        batch_seq_encoder_input = self.word_encoder(batch_seq_input, batch_seq_char_input)
        batch_seq_feats = self.seq_encoder(batch_seq_encoder_input, batch_seq_len).contiguous()
        ent_outputs = self.seq_decoder(batch_seq_feats, batch_seq_tags)

        batch['ent_pred'] = ent_outputs['predict']
        batch['seq_encoder'] = batch_seq_feats
        self.create_rel_inputs(batch)
        rel_batch = self.create_rel_batch(batch)

        outputs = {}
        outputs['ent_loss'] = ent_outputs['loss']
        outputs['ent_pred'] = ent_outputs['predict']
        if len(rel_batch['all_candi_rels']) != 0:
            rel_feats = self.rel_feat_extractor(rel_batch)
            rel_outputs = self.rel_decoder(rel_feats, rel_batch['all_rel_labels'])

            rel_pred = rel_outputs['predict']
            all_rel_pred = [[] for _ in batch['seq_lens']]
            for i, pred in enumerate(rel_pred):
                all_rel_pred[batch['rel2ent_batch'][i]].append(pred.item())

            outputs['rel_loss'] = rel_outputs['loss']
            outputs['all_candi_rels'] = batch['all_candi_rels']
            outputs['all_rel_pred'] =  all_rel_pred
        else:
            zero_loss = torch.Tensor([0])
            zero_loss.requires_grad = True
            if self.use_cuda:
                zero_loss = zero_loss.cuda(non_blocking=True)
            outputs['rel_loss'] = zero_loss[0]
            outputs['all_rel_pred'] = [[] for _ in batch['seq_lens']]
            outputs['all_candi_rels'] = [[] for _ in batch['seq_lens']]
        return outputs

    def create_rel_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        rel_batch = defaultdict(list)
        for i in range(len(batch['tokens'])):
            rel_batch['all_candi_rels'].extend(batch['all_candi_rels'][i])
            rel_batch['all_rel_labels'].extend(batch['all_rel_labels'][i])
            candi_num = len(batch['all_candi_rels'][i])
            rel_batch['tokens'].extend([batch['tokens'][i] for _ in range(candi_num)])
            rel_batch['token_chars'].extend([batch['token_chars'][i] for _ in range(candi_num)])
            rel_batch['ent_labels'].extend([batch['ent_labels'][i] for _ in range(candi_num)])
            rel_batch['seq_encoder'].extend([batch['seq_encoder'][i] for _ in range(candi_num)])
            rel_batch['seq_lens'].extend([batch['seq_lens'][i] for _ in range(candi_num)])
            rel_batch['sch_ent_labels'].extend([batch['sch_ent_labels'][i] for _ in range(candi_num)])
        rel_batch['all_rel_labels'] = torch.LongTensor(rel_batch['all_rel_labels'])
        if self.use_cuda:
            rel_batch['all_rel_labels'] = rel_batch['all_rel_labels'].cuda(non_blocking=True)
        return rel_batch

    def create_rel_inputs(self,
                          batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        batch['rel2ent_batch'] = {}
        batch['all_candi_rels'] = []
        batch['all_rel_labels'] = []
        batch['sch_ent_labels'] = []
        if batch['i_epoch'] is not None:
            sch_p = self.sch_k / (self.sch_k + np.exp(batch['i_epoch'] / self.sch_k))        
        rel_idx = 0
        for i in range(len(batch['seq_lens'])):
            ent_pred = batch['ent_pred'][i]
            ent_gold = batch['ent_labels'][i]
            seq_len = batch['seq_lens'][i]
            candi_rels = batch['candi_rels'][i]
            rel_labels = batch['rel_labels'][i]
            if self.training and batch['i_epoch'] is not None:
                rd = np.random.random()                                         
                sch_ent_labels = ent_gold if rd <= sch_p else ent_pred
            else:
                sch_ent_labels = ent_pred
            candi2rel = {(tuple(k[0]), tuple(k[1])): v for k, v in zip(candi_rels, rel_labels)}

            all_candi_rels, all_rel_labels = self.generate_all_candi_rels(sch_ent_labels[:seq_len], candi2rel)

            assert len(all_candi_rels) == len(all_rel_labels)

            batch['all_candi_rels'].append(all_candi_rels)
            batch['all_rel_labels'].append(all_rel_labels)
            batch['sch_ent_labels'].append(sch_ent_labels)
            for each_candi in all_candi_rels:
                batch['rel2ent_batch'][rel_idx] = i
                rel_idx += 1

    def generate_all_candi_rels(
            self,
            ent_labels: torch.LongTensor,
            candi2rel: Dict[Tuple, int]) -> (List[Tuple], List[int]):
        all_candi_rels = []
        all_rel_labels = []
        y = [self.vocab.get_token_from_index(t.item(), "ent_labels") for t in ent_labels]
        t_entity = self.get_entity(y)
        entity_idx2type = self.get_entity_idx2type(t_entity)
        candi_set = self.generate_candi_rel(entity_idx2type)
        if self.training:
            self.add_gold_candidate(candi_set, candi2rel)

        for b, e in candi_set:
            if (b, e) in candi2rel:
                t = candi2rel[(b, e)]
            else:
                t = self.vocab.get_token_index("None", "rel_labels")
            all_candi_rels.append((b, e))
            all_rel_labels.append(t)
        return all_candi_rels, all_rel_labels

    def add_gold_candidate(self,
                           candi_set: Set[Tuple],
                           candi2rel: Dict[Tuple, int]) -> None:
        for b, e in candi2rel.keys():
            b = tuple(b)
            e = tuple(e)

            assert b[-1] <= e[0]

            candi_set.add((b, e))


    def generate_candi_rel(self, entity_idx2type: Dict[Tuple, str]) -> Set[Tuple]:
        candi_set = set()
        for ent1_idx in entity_idx2type.keys():
            for ent2_idx in entity_idx2type.keys():
                if ent1_idx[0] >= ent2_idx[0]:
                    continue
                candi_set.add((ent1_idx, ent2_idx))
        return candi_set

    def get_entity_idx2type(self, t_entity: Dict[str, list]) -> Dict[Tuple, str]:
        entity_idx2type = {}
        for k, v in t_entity.items():
            for e in v:
                entity_idx2type[e] = "Entity"
        return entity_idx2type

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
