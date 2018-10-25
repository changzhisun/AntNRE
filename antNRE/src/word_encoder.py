#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/09 20:39:10

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from antNRE.src.char_cnn import CharCNN

class WordCharEncoder(nn.Module):
    
    def __init__(self,
                 word_vocab_size: int,
                 word_dims: int,
                 char_emb_kwargs: Dict[str, Any],
                 dropout: float = 0.5,
                 padding_idx: int = 0,
                 aux_word_dims: Optional[int] = None,
                 char_batch_size: int = 10) -> None:
        super(WordCharEncoder, self).__init__()
        self.char_embeddings = CharCNN(**char_emb_kwargs)
        self.word_embeddings = nn.Embedding(word_vocab_size,
                                            word_dims,
                                            padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.char_batch_size = char_batch_size
        if aux_word_dims is not None:
            self.aux_word_embeddings = nn.Embedding(word_vocab_size,
                                                    aux_word_dims,
                                                    padding_idx=padding_idx)

    def forward(self,
                batch_sent_input: torch.LongTensor,
                batch_sent_char_input: Optional[torch.LongTensor] = None) -> Dict[str, Any]:
        """
        Parameters
        ----------
        batch_sent_input : ``torch.LongTensor``, required
            (batch_size, sent_size)

        batch_sent_char_input : ``torch.LongTensor``, required
            (batch_size, sent_size, char_seq_size)

        Returns
        -------
        batch_sent_vecs : ``torch.Tensor``, required
            (batch_size, sent_size, -1)
        """
        batch_sent_vecs = self.word_embeddings(batch_sent_input)
        batch_sent_vecs = self.dropout(batch_sent_vecs)
        if batch_sent_char_input is not None:
            batch_sent_char_vecs = []
            for i in range(0, len(batch_sent_char_input), self.char_batch_size):
                char_batch = batch_sent_char_input[i: i + self.char_batch_size]
                batch_sent_char_vecs.append(self.char_embeddings(char_batch))
            batch_sent_char_vecs = torch.cat(batch_sent_char_vecs, 0)
            embedding_list = [batch_sent_char_vecs, batch_sent_vecs]
            if hasattr(self, 'aux_word_embeddings'):
                aux_batch_sent_vecs = self.aux_word_embeddings(batch_sent_input)
                embedding_list.append(aux_batch_sent_vecs)
            batch_sent_vecs = torch.cat(embedding_list, 2)
        return batch_sent_vecs
