#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/09 19:46:54

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CharCNN(nn.Module):

    def __init__(self,
                 char_vocab_size: int,
                 char_dims: int,
                 out_channels: int,
                 kernel_sizes: List,
                 padding_idx: int = 0,
                 dropout: float = 0.5) -> None:
        super(CharCNN, self).__init__()
        self.char_embeddings = nn.Embedding(char_vocab_size,
                                            char_dims, 
                                            padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, (K, char_dims), padding=(K-1, 0))
                                    for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_sent_char_input : torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        batch_sent_char_input : ``torch.LongTensor``, required
            (batch_size, sent_size, char_seq_size)

        Returns
        -------
        batch_sent_char_vecs : ``torch.Tensor``, required
            (batch_size, sent_size, -1)
        """
        batch_size, sent_size, char_seq_size = batch_sent_char_input.size()

        batchsent_char_input = batch_sent_char_input.view(-1, char_seq_size)    # (batch_size x sent_size, char_seq_size)
        batchsent_char_emb = self.char_embeddings(batchsent_char_input)         # (batch_size x sent_size, char_seq_size, char_dims)
        batchsent_char_emb = self.dropout(batchsent_char_emb)
        batchsent_char_emb = batchsent_char_emb.unsqueeze(1)                    # (batch_size x sent_size, 1, char_seq_size, char_dims)
        
        batchsent_char_conv = [F.relu(conv(batchsent_char_emb)).squeeze(3)
                               for conv in self.convs]
        batchsent_pool = [F.max_pool1d(char_conv, char_conv.size(2)).squeeze(2) 
                          for char_conv in batchsent_char_conv]
        batchsent_pool = torch.cat(batchsent_pool, 1)
        batchsent_pool = self.dropout(batchsent_pool)
        batch_sent_char_vecs = batchsent_pool.view(batch_size, sent_size, -1)
        return batch_sent_char_vecs
