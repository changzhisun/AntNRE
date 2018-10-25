#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/10 21:49:23

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from antNRE.src.word_encoder import WordCharEncoder
from antNRE.src.seq_encoder import BiLSTMEncoder
from antNRE.src.seq_decoder import SeqSoftmaxDecoder

class EntModel(nn.Module):

    def __init__(self,
                 word_encoder: WordCharEncoder,
                 seq_encoder: BiLSTMEncoder,
                 seq_decoder: SeqSoftmaxDecoder) -> None:
        super(EntModel, self).__init__()
        self.word_encoder = word_encoder
        self.seq_encoder = seq_encoder
        self.seq_decoder = seq_decoder

    def forward(self,
                batch_seq_input: torch.LongTensor,
                batch_seq_char_input: Optional[torch.LongTensor],
                batch_seq_tags: torch.LongTensor) -> Dict:
        batch_seq_encoder_input = self.word_encoder(batch_seq_input, batch_seq_char_input)
        batch_seq_len = (batch_seq_tags >= 0).int().sum(dim=1)
        batch_seq_feats = self.seq_encoder(batch_seq_encoder_input, batch_seq_len).contiguous()
        outputs = {}
        res = self.seq_decoder(batch_seq_feats, batch_seq_tags)
        outputs['ent_loss'] = res['loss']
        outputs['ent_pred'] = res['predict']
        return outputs
