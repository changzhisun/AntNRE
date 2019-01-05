#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/01/04 23:06:51

@author: Changzhi Sun
"""
import torch.nn as nn

class Seq2SeqEncoder(nn.Module):

    """
    A ``Seq2SeqEncoder`` is a ``Module`` that takes as input a sequence of vectors and returns a
    modified sequence of vectors.  Input shape: ``(batch_size, sequence_length, input_dim)``; output
    shape: ``(batch_size, sequence_length, output_dim)``.
    """

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError
