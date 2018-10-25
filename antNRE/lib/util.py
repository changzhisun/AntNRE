#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/09/21 15:26:50

@author: Changzhi Sun
"""
import sys, os
import json
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from antNRE.lib.vocabulary import Vocabulary


def load_word_vectors(vector_file: str,
                      ndims: int,
                      vocab: Vocabulary, 
                      namespace: str = 'tokens') -> List[List]:
    token_vocab_size = vocab.get_vocab_size(namespace)
    oov_idx = vocab.get_token_index(vocab._oov_token, namespace)
    padding_idx = vocab.get_token_index(vocab._padding_token, namespace)
    W = np.random.uniform(-0.25, 0.25, (token_vocab_size, ndims))
    W[padding_idx, :] = 0.0
    total, found = 0, 0
    with open(vector_file) as fp:
        for i, line in enumerate(fp):
            line = line.rstrip().split()
            if line:
                total += 1
                try:
                    assert len(line) == ndims+1,(
                        "Line[{}] {} vector dims {} doesn't match ndims={}".format(i, line[0], len(line)-1, ndims)
                    )
                except AssertionError as e:
                    print(e)
                    continue
                word = line[0]
                idx = vocab.get_token_index(word, namespace)
                if idx != oov_idx:
                    found += 1
                    vecs = np.array(list(map(float, line[1:])))
                    W[idx, :] = vecs
    print("Found {} [{:.2f}%] vectors from {} vectors in {} with ndims={}".format(
        found, found * 100/token_vocab_size, total, vector_file, ndims))
    #  norm_W = np.sqrt((W*W).sum(axis=1, keepdims=True))
    #  valid_idx = norm_W.squeeze() != 0
    #  W[valid_idx, :] /= norm_W[valid_idx]
    return W

def parse_tag(t: str) -> Tuple:
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def end_of_chunk(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'U': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'U': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'U': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'U': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'U' and tag == 'E': chunk_start = True
    if prev_tag == 'U' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start


def assign_embeddings(embedding_module: nn.Embedding,
                      pretrained_embeddings: np.array,
                      fix_embedding: bool = False) -> None:
    embedding_module.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
    if fix_embedding:
        embedding_module.weight.requires_grad = False
