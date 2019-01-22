#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/16 13:46:14

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

class VanillaSoftmaxDecoder(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 tag_size: int,
                 bias: bool = True) -> None:
        super(VanillaSoftmaxDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.tag_size = tag_size

        self.hidden2tag = nn.Linear(hidden_size, self.tag_size, bias=bias)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self,
                batch_feats: torch.Tensor,
                batch_tags: torch.LongTensor) -> Dict[str, Any]:
        batch_size, hidden_size = batch_feats.size()

        assert hidden_size == self.hidden_size
        
        batch_feats = self.hidden2tag(batch_feats)           
        ave_loss = self.loss_function(batch_feats, batch_tags)
        
        batch_pred_tags = batch_feats.argmax(dim=1)

        outputs = {}
        outputs['loss'] = ave_loss
        outputs['predict'] = batch_pred_tags
        outputs['softmax_probs'] = F.softmax(batch_feats, dim=1)
        return outputs
