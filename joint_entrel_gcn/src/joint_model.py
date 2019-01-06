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
from antNRE.lib.util import parse_tag
from antNRE.lib.util import start_of_chunk
from antNRE.lib.util import end_of_chunk
from src.ent_span_generator import EntSpanGenerator
from allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor import BidirectionalEndpointSpanExtractor
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor


class JointModel(nn.Module):

    def __init__(self,
                 word_encoder: WordCharEncoder,
                 seq2seq_encoder: BiLSTMEncoder,
                 ent_span_decoder: SeqSoftmaxDecoder,
                 ent_ids_span_extractor: EndpointSpanExtractor,
                 ent_ids_decoder: VanillaSoftmaxDecoder,
                 #  rel_feat_extractor: RelFeatureExtractor,
                 #  rel_decoder: VanillaSoftmaxDecoder,
                 sch_k: float,
                 vocab: Vocabulary,
                 use_cuda: bool) -> None:
        super(JointModel, self).__init__()
        self.word_encoder = word_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.ent_span_decoder = ent_span_decoder
        self.ent_ids_span_extractor = ent_ids_span_extractor
        self.ent_ids_decoder = ent_ids_decoder
        #  self.rel_feat_extractor = rel_feat_extractor
        #  self.rel_decoder = rel_decoder
        self.sch_k = sch_k
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

        outputs = {}
        outputs['ent_span_loss'] = ent_span_outputs['loss']
        outputs['ent_span_pred'] = ent_span_outputs['predict']

        ent_ids_num = sum([len(e) for e in batch['all_ent_ids']])
        if ent_ids_num != 0:
            ent_ids_batch = self.create_ent_ids_batch(batch, seq_feats)
            
            ent_ids_span_feats = self.ent_ids_span_extractor(ent_ids_batch['seq_feats'], 
                                                             ent_ids_batch['all_ent_ids'])
            ent_ids_outputs = self.ent_ids_decoder(ent_ids_span_feats.squeeze(1), 
                                                   ent_ids_batch['all_ent_ids_label'])
            all_ent_pred = self.create_all_ent_pred(batch, ent_ids_outputs)
            outputs['ent_ids_loss'] = ent_ids_outputs['loss']
            outputs['all_ent_pred'] = all_ent_pred
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
        return outputs

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

    def create_ent_ids_batch(self,
                             batch: Dict[str, Any],
                             batch_seq_feats: torch.FloatTensor) -> Dict[str, Any]:
        """
        Returns
        -------
        "seq_feats": torch.FloatTensor (new_batch_size, sequence_size, hidden_size)
            new_batch_size = the number of entity spans of batch
        "all_ent_ids": torch.LongTensor (new_batch_size, 1, 2)
            one span, [start, end]
        "all_ent_ids_label": torch.LongTensor (new_batch_size)
        """
        ent_ids_batch = defaultdict(list)
        for i in range(len(batch['tokens'])):
            ent_ids_batch['all_ent_ids'].extend(batch['all_ent_ids'][i])
            ent_ids_batch['all_ent_ids_label'].extend(batch['all_ent_ids_label'][i])
            ent_ids_num = len(batch['all_ent_ids_label'][i])
            ent_ids_batch['seq_feats'].extend([batch_seq_feats[i] for _ in range(ent_ids_num)])
        ent_ids_batch['all_ent_ids_label'] = torch.LongTensor(ent_ids_batch['all_ent_ids_label'])
        ent_ids_batch['all_ent_ids'] = torch.LongTensor(ent_ids_batch['all_ent_ids']).unsqueeze(1)
        ent_ids_batch['seq_feats'] = torch.stack(ent_ids_batch['seq_feats'], 0)
        if self.use_cuda:
            ent_ids_batch['all_ent_ids_label'] = ent_ids_batch['all_ent_ids_label'].cuda(non_blocking=True)
            ent_ids_batch['all_ent_ids'] = ent_ids_batch['all_ent_ids'].cuda(non_blocking=True)
        return ent_ids_batch

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
