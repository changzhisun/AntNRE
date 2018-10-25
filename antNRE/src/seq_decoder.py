#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/10 16:08:57

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

class SeqSoftmaxDecoder(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 tag_size: int,
                 bias: bool = True) -> None:
        super(SeqSoftmaxDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.tag_size = tag_size

        self.hidden2tag = nn.Linear(hidden_size, self.tag_size, bias=bias)

    def forward(self,
                batch_seq_feats: torch.Tensor,
                batch_seq_tags: torch.LongTensor) -> Dict[str, Any]:

        batch_size, seq_size, hidden_size = batch_seq_feats.size()

        assert hidden_size == self.hidden_size
        
        batchseq_feats = batch_seq_feats.view(-1, hidden_size)
        batchseq_feats = self.hidden2tag(batchseq_feats)           
        batchseq_log_probs = F.log_softmax(batchseq_feats, dim=1)  # (batch_size x seq_size, tag_size)
        
        no_pad_ave_loss = self.loss_fn(batchseq_log_probs, batch_seq_tags)
        #  batch_seq_pred_tags = self.get_seq_pred_tags(batchseq_feats,
                                                     #  batch_seq_tags)
        batch_seq_pred_tags = batchseq_feats.argmax(dim=1).view(batch_size, seq_size)

        outputs = {}
        outputs['loss'] = no_pad_ave_loss
        outputs['predict'] = batch_seq_pred_tags
        return outputs

    def loss_fn(self, log_probs: torch.Tensor, tags: torch.LongTensor) -> torch.Tensor:
        tags = tags.view(-1)
        mask = (tags >= 0).float()
        num_tokens = int(torch.sum(mask).item())
        log_probs = log_probs[range(log_probs.size(0)), tags] * mask
        return -torch.sum(log_probs) / num_tokens
    
    def get_seq_pred_tags(self,
                          batchseq_feats: torch.Tensor,
                          batch_seq_tags: torch.LongTensor) -> List[torch.Tensor]:
        batch_size, seq_size = batch_seq_tags.size()
        batch_seq_pred_tags = batchseq_feats.argmax(dim=1).view(batch_size, seq_size)
        mask = (batch_seq_tags >= 0).int()
        batch_seq_pred_tags = [seq_pred_tags[:cur_seq_len]
                               for seq_pred_tags, cur_seq_len in zip(batch_seq_pred_tags, mask.sum(dim=1))]
        return batch_seq_pred_tags
