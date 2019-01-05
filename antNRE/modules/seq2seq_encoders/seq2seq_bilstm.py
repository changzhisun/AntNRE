#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/01/04 23:18:46

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
from typing import List
from overrides import overrides

from antNRE.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

class BiLSTMEncoder(Seq2SeqEncoder):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.5) -> None:
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_size,
                              hidden_size // 2,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True,
                              dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    @overrides
    def get_input_dim(self) -> int:
        return  self.input_size

    @overrides
    def get_output_dim(self) -> int:
        return  self.hidden_size

    def forward(self,
                inputs: torch.Tensor,
                seq_lens: List[int]) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: shape
            (batch_size, sequence_size, input_size)

        seq_lens: shape
            (batch_size)
        """
        batch_size, sequence_size, input_size = inputs.size()

        assert input_size == self.input_size
        assert batch_size == len(seq_lens)

        pack_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, seq_lens, batch_first=True)
        pack_outputs, _ = self.bilstm(pack_inputs) 
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            pack_outputs, batch_first=True)
        return self.dropout(outputs)
