#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/01/07 23:19:21

@author: Changzhi Sun
"""
import torch
from overrides import overrides
from typing import Tuple

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.nn import util
from antNRE.modules.seq2vec_encoders.cnn_encoder import CnnEncoder

class CnnSpanExtractor(SpanExtractor):
    """
    Computes span representations by running CnnEncoder for each word in the document. 

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.
    Returns
    -------
    sum_text_embeddings : ``torch.FloatTensor``.
        A tensor of shape (batch_size, num_spans, output_dim), which each span representation
        is formed by running CnnEncoder over the span.
    """
    def __init__(self,
                 input_dim: int,
                 num_filters: int,
                 ngram_filter_sizes: Tuple[int]) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = num_filters * len(ngram_filter_sizes)
        self.cnn = CnnEncoder(input_dim, num_filters, tuple(ngram_filter_sizes))

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sequence_tensor)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        #  text_embeddings = span_embeddings * span_mask.unsqueeze(-1)
        batch_size, num_spans, max_batch_span_width, _ = span_embeddings.size()

        view_text_embeddings = span_embeddings.view(batch_size * num_spans,
                                                    max_batch_span_width,
                                                    -1)
        span_mask = span_mask.view(batch_size * num_spans, max_batch_span_width)
        cnn_text_embeddings = self.cnn(view_text_embeddings, span_mask)
        cnn_text_embeddings = cnn_text_embeddings.view(batch_size, num_spans, self._output_dim)
        return cnn_text_embeddings

#  torch.manual_seed(1)
#  sequence_tensor = torch.randn(2, 1, 5).cuda()
#  span_indices = torch.LongTensor([[[0, 0]], [[0, 0]]]).cuda()
#  extractor = CnnSpanExtractor(5, 
                             #  5,
                             #  (2, 3)).cuda()
#  print(extractor(sequence_tensor, span_indices))
#  print(extractor(sequence_tensor, span_indices).size())
#  print("====")
#  print((sequence_tensor[0][0] + sequence_tensor[0][1]) )

#  print((sequence_tensor[1][1] + sequence_tensor[1][2] + sequence_tensor[1][3]) )

