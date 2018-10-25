#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/10 21:30:45

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from antNRE.src.word_encoder import WordCharEncoder

class BiLSTMEncoder(nn.Module):

    def __init__(self,
                 word_encoder_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.5) -> None:
        super(BiLSTMEncoder, self).__init__()
        self.word_encoder_size = word_encoder_size
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(word_encoder_size,
                              hidden_size // 2,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True,
                              dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                batch_seq_encoder_input: torch.Tensor,
                batch_seq_len: List) -> torch.Tensor:
        batch_size, seq_size, word_encoder_size  = batch_seq_encoder_input.size()

        assert word_encoder_size == self.word_encoder_size

        batch_seq_encoder_input_pack = nn.utils.rnn.pack_padded_sequence(
            batch_seq_encoder_input,
            batch_seq_len,
            batch_first=True)
        batch_seq_encoder_output, _ = self.bilstm(batch_seq_encoder_input_pack) 
        batch_seq_encoder_output, _ = nn.utils.rnn.pad_packed_sequence(
            batch_seq_encoder_output, batch_first=True)
        return self.dropout(batch_seq_encoder_output)
