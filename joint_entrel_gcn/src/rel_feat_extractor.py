#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/01/07 00:06:53

@author: Changzhi Sun
"""
import sys
sys.path.append("..")
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor import BidirectionalEndpointSpanExtractor

class RelFeatExtractor(nn.Module):

    def __init__(self,
                 hidden_size,
                 context_span_extractor: BidirectionalEndpointSpanExtractor,
                 dropout: float,
                 use_cuda: bool) -> None:
        super(RelFeatExtractor, self).__init__()
        self.context_span_extractor  = context_span_extractor
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.rel2hidden = nn.Sequential(nn.Linear(self.context_span_extractor.get_output_dim(),
                                                  self.hidden_size),
                                        nn.ReLU(),
                                        self.dropout)
        self.rel_feat_size = self.hidden_size * 5 
        self.mlp = nn.Sequential(nn.Linear(self.rel_feat_size,
                                           self.hidden_size),
                                 nn.ReLU(),
                                 self.dropout)
        zero_feat = torch.zeros(self.context_span_extractor.get_output_dim())
        if use_cuda:
            zero_feat = zero_feat.cuda(non_blocking=True)
        self.zero_feat = zero_feat

    def forward(self,
                batch: Dict[str, Any],
                ent_ids_span_feats: torch.FloatTensor) -> Dict[str, Any]:
        cache_span_feats = self.cache_all_span_feats(batch, ent_ids_span_feats)
        rel_batch = self.create_rel_batch(batch, cache_span_feats)
        return rel_batch
        
    def get_output_dim(self) -> int:
        return self.hidden_size

    def cache_all_span_feats(self,
                             batch: Dict[str, Any],
                             ent_ids_span_feats: torch.FloatTensor) -> List[Dict[Tuple, torch.FloatTensor]]:
        cache_ent_span_feats = self.cache_span_feats(batch['all_ent_ids'], ent_ids_span_feats)
        context_spans = self.get_context_spans(batch['all_candi_rels'], batch['seq_lens'])

        all_context_spans = []
        all_seq_feats = []
        for i, span in enumerate(context_spans):
            all_context_spans.extend(span)
            num_span = len(span)
            all_seq_feats.extend([batch['seq_feats'][i] for _ in range(num_span)])
        if len(all_seq_feats) == 0:
            return cache_ent_span_feats

        all_seq_feats = torch.stack(all_seq_feats)
        all_context_span_tensor = torch.LongTensor(all_context_spans).unsqueeze(1)
        if self.use_cuda:
            all_context_span_tensor = all_context_span_tensor.cuda(non_blocking=True)

        all_context_span_feat = self.context_span_extractor(
            all_seq_feats, all_context_span_tensor).squeeze(1)

        cache_context_span_feat = self.cache_span_feats(context_spans, all_context_span_feat)

        cache_span_feats = []
        for ent_span_dict, context_span_dict in zip(cache_ent_span_feats, cache_context_span_feat):
            new_dict = {}
            new_dict.update(ent_span_dict)
            new_dict.update(context_span_dict)
            cache_span_feats.append(new_dict)
        return cache_span_feats

    def cache_span_feats(self, 
                         all_spans: List[List[Tuple]],
                         span_feats: torch.FloatTensor) -> List[Dict[Tuple, torch.FloatTensor]]:
        cache_span_feats = []
        i = 0
        for cur_spans in all_spans:
            span2feat = {}
            for e in cur_spans:
                span2feat[e] = span_feats[i]
                i += 1
            cache_span_feats.append(span2feat)
        return cache_span_feats

    def create_rel_batch(self,
                         batch: Dict[str, Any],
                         cache_span_feats: List[Dict[Tuple, torch.FloatTensor]]) -> Dict[str, Any]:

        rel_batch = defaultdict(list)
        for i in range(len(batch['tokens'])):
            rel_batch['all_rel_labels'].extend(batch['all_rel_labels'][i])
            span2feat = cache_span_feats[i]
            seq_len = batch['seq_lens'][i]
            for e1, e2 in batch['all_candi_rels'][i]:
                L = (0, e1[0]-1)
                M = (e1[1]+1, e2[0]-1)
                R = (e2[1]+1, seq_len-1)

                if L[0] > L[1]:
                    rel_batch['L'].append(self.zero_feat)
                else:
                    rel_batch['L'].append(span2feat[L])

                if M[0] > M[1]:
                    rel_batch['M'].append(self.zero_feat)
                else:
                    rel_batch['M'].append(span2feat[M])

                if R[0] > R[1]:
                    rel_batch['R'].append(self.zero_feat)
                else:
                    rel_batch['R'].append(span2feat[R])

                rel_batch['E1'].append(span2feat[e1])
                rel_batch['E2'].append(span2feat[e2])
        rel_batch['E1'] = torch.stack(rel_batch['E1'])
        rel_batch['E2'] = torch.stack(rel_batch['E2'])

        rel_batch['L'] = self.rel2hidden(torch.stack(rel_batch['L']))
        rel_batch['M'] = self.rel2hidden(torch.stack(rel_batch['M']))
        rel_batch['R'] = self.rel2hidden(torch.stack(rel_batch['R']))

        rel_batch['all_rel_labels'] = torch.LongTensor(rel_batch['all_rel_labels'])
        concat_list = [rel_batch['L'],
                       rel_batch['E1'],
                       rel_batch['M'],
                       rel_batch['E2'],
                       rel_batch['R']]
        rel_batch['inputs'] = self.mlp(torch.cat(concat_list, 1))
        if self.use_cuda:
            rel_batch['all_rel_labels'] = rel_batch['all_rel_labels'].cuda(non_blocking=True)
        return rel_batch

    def get_context_spans(self,
                          all_candi_rels: List[List[Tuple[Tuple, Tuple]]],
                          seq_lens: List[int]) -> List[List[Tuple]]:
        context_spans = []
        for i in range(len(seq_lens)):
            seq_len = seq_lens[i]
            cur_context_spans = []
            for e1, e2 in all_candi_rels[i]:
                #  print(e1, e2, seq_len)
                L = (0, e1[0]-1)
                M = (e1[1]+1, e2[0]-1)
                R = (e2[1]+1, seq_len-1)
                for e in [L, M, R]:
                    if e[0] > e[1]:
                        continue
                    cur_context_spans.append(e)
            #  print(cur_context_spans)
            #  print("=====")
            context_spans.append(cur_context_spans)
        return context_spans
