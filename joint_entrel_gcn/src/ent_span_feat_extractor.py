#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/01/07 00:40:06

@author: Changzhi Sun
"""
import sys
sys.path.append("..")
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor

class EntSpanFeatExtractor(nn.Module):

    def __init__(self,
                 hidden_size,
                 ent_ids_span_extractor: EndpointSpanExtractor,
                 dropout: float,
                 use_cuda: bool) -> None:
        super(EntSpanFeatExtractor, self).__init__()
        self.ent_ids_span_extractor = ent_ids_span_extractor
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.ent2hidden = nn.Sequential(nn.Linear(self.ent_ids_span_extractor.get_output_dim(),
                                                  self.hidden_size),
                                        nn.ReLU(),
                                        self.dropout)
        self.use_cuda = use_cuda

    def forward(self, batch: Dict[str, Any]) -> (Dict[str, Any], torch.FloatTensor):
        ent_ids_batch = self.create_ent_ids_batch(batch)
        ent_ids_span_feats = self.ent_ids_span_extractor(
            ent_ids_batch['seq_feats'],
            ent_ids_batch['all_ent_ids'])
        ent_ids_span_feats = self.ent2hidden(ent_ids_span_feats.squeeze(1))
        return ent_ids_batch, ent_ids_span_feats

    def create_ent_ids_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns
        -------
        "seq_feats": torch.FloatTensor (new_batch_size, sequence_size, hidden_size)
            new_batch_size = the number of entity spans of batch
        "all_ent_ids": torch.LongTensor (new_batch_size, 1, 2)
            one span, [start, end]
        "all_ent_ids_label": torch.LongTensor (new_batch_size)
        """
        batch_seq_feats = batch['seq_feats']
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
