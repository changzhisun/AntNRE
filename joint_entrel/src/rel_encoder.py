#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/12 16:24:46

@author: Changzhi Sun
"""
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from antNRE.src.word_encoder import WordCharEncoder
from antNRE.src.seq_encoder import BiLSTMEncoder
from antNRE.lib.vocabulary import Vocabulary
from antNRE.lib.util import assign_embeddings

class RelFeatureExtractor(nn.Module):

    def __init__(self,
                 word_encoder: WordCharEncoder,
                 seq_encoder: BiLSTMEncoder,
                 vocab : Vocabulary,
                 out_channels: int,
                 kernel_sizes: List,
                 max_sent_len: int,
                 dropout: float,
                 use_cuda: bool) -> None:
        super(RelFeatureExtractor, self).__init__()
        self.word_encoder = word_encoder
        self.seq_encoder = seq_encoder
        self.vocab = vocab
        self.use_cuda = use_cuda
        self.ent_vocab_size = vocab.get_vocab_size("ent_labels")
        self.conv_input_size = self.seq_encoder.hidden_size + self.ent_vocab_size
        self.max_sent_len = max_sent_len
        self.ent_emb_oh = torch.Tensor(np.eye(self.ent_vocab_size))
        if use_cuda:
            self.ent_emb_oh = self.ent_emb_oh.cuda(non_blocking=True)

        self.e1_convs = nn.ModuleList([nn.Conv2d(1,
                                                 out_channels,
                                                 (K, self.conv_input_size),
                                                 padding=(K-1, 0))
                                       for K in kernel_sizes])
        self.m_convs = nn.ModuleList([nn.Conv2d(1,
                                                out_channels,
                                                (K, self.conv_input_size),
                                                padding=(K-1, 0))
                                      for K in kernel_sizes])
        self.e2_convs = nn.ModuleList([nn.Conv2d(1,
                                                out_channels,
                                                (K, self.conv_input_size),
                                                padding=(K-1, 0))
                                       for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.seq_encoder.hidden_size
        self.all_feat_dim = len(kernel_sizes) * out_channels * 3 + 2 * self.hidden_size + max_sent_len
        self.mlp = nn.Sequential(self.dropout,
                                 nn.Linear(self.all_feat_dim, self.hidden_size),
                                 nn.ReLU(),
                                 self.dropout)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        if batch['seq_encoder'] is None:
            batch_seq_encoder_input = self.word_encoder(
                batch['tokens'], batch['token_chars'])
            batch_input['seq_encoder'] = self.seq_encoder(
                batch_seq_encoder_input, batch['seq_len']).contiguous()

        batch_word_repr = self.generate_word_reprsentation(
            batch['seq_encoder'],
            batch['seq_lens'], 
            batch['sch_ent_labels'])

        E1 = [e1 for e1, e2 in batch['all_candi_rels']]
        E2 = [e2 for e1, e2 in batch['all_candi_rels']]
        M = [(e1[-1], e2[0]) for e1, e2 in batch['all_candi_rels']]
        L = [(0, e1[0]) for e1, e2 in batch['all_candi_rels']]
        R = [(e2[-1], cur_len) for (e1, e2), cur_len in zip(batch['all_candi_rels'], batch['seq_lens'])]

        M_conv_vecs = self.get_conv_vecs(batch_word_repr, M, self.m_convs)
        E1_conv_vecs = self.get_conv_vecs(batch_word_repr, E1, self.e1_convs)
        E2_conv_vecs = self.get_conv_vecs(batch_word_repr, E2, self.e2_convs)

        L_vecs = self.get_biLSTM_Minus(batch['seq_encoder'], L, batch['seq_lens'])
        R_vecs = self.get_biLSTM_Minus(batch['seq_encoder'], R, batch['seq_lens'])

        dist_vecs = self.get_dist_vecs(M)

        final_vecs = torch.cat([M_conv_vecs, E1_conv_vecs, E2_conv_vecs, L_vecs, R_vecs, dist_vecs], 1)
        return self.mlp(final_vecs)

    def get_dist_vecs(self, ent_idxs: List) -> torch.Tensor:
        dist_vecs = []
        for s, e in ent_idxs:
            assert s <= e

            vecs = np.eye(self.max_sent_len)[e - s]
            vecs = torch.Tensor(vecs)
            
            if self.use_cuda:
                vecs = vecs.cuda(non_blocking=True)
            dist_vecs.append(vecs)
        dist_vecs = torch.stack(dist_vecs)
        return dist_vecs
       
    def get_biLSTM_Minus(self,
                         seq_encoder: List[torch.Tensor], 
                         span_list: List,
                         seq_lens: List[int]) -> torch.Tensor:
        hidden_size = self.seq_encoder.hidden_size
        vecs = []
        for i in range(len(seq_encoder)):
            s, e = span_list[i]
            rnn_outputs = seq_encoder[i][:seq_lens[i]]
            fward_rnn_output, bward_rnn_output = rnn_outputs.split(hidden_size // 2, 1)
            fward_right_vec = self.get_forward_segment(fward_rnn_output, s, e)
            bward_right_vec = self.get_forward_segment(bward_rnn_output, s, e)
            vec = torch.cat([fward_right_vec, bward_right_vec], 0).unsqueeze(0)
            vecs.append(vec)
        vecs = torch.cat(vecs, 0)
        return vecs

    def get_forward_segment(self,
                            fward_rnn_output: torch.Tensor, 
                            s: int,
                            e: int) -> torch.Tensor:
        max_len, h_size = fward_rnn_output.size()
        if s >= e:
            zero_vec = torch.zeros(h_size)
            if self.use_cuda:
                zero_vec = zero_vec.cuda(non_blocking=True)
            return zero_vec
        if s == 0:
            return fward_rnn_output[e-1]
        return fward_rnn_output[e-1] - fward_rnn_output[s-1]

    def get_backward_segment(self, bward_rnn_output, s, e):
        max_len, h_size = bward_rnn_output.size()
        if s >= e:
            zero_vec = torch.zeros(h_size)
            if self.use_cuda:
                zero_vec = zero_vec.cuda(non_blocking=True)
            return zero_vec
        if e == max_len:
            return bward_rnn_output[b]
        return bward_rnn_output[b] - bward_rnn_output[e]

    def get_conv_vecs(self,
                      batch_word_repr: List[torch.Tensor],
                      span_list: List,
                      convs: nn.Module) -> torch.tensor:
        vecs = []
        for word_repr, (s, e) in zip(batch_word_repr, span_list):
            if s == e:
                vecs.append([])
                continue
            vecs.append(list(word_repr[s:e].split(1)))
        vecs = self.pad_feature(vecs)
        vecs = self.apply_conv(vecs, convs)
        return vecs
    
    def apply_conv(self, h: torch.Tensor, convs: nn.Module) -> torch.Tensor:
        h = self.dropout(h)
        h = h.unsqueeze(1) # batch_size x 1 x seq_size x conv_input_size
        h = [F.relu(conv(h)).squeeze(3) for conv in convs] #[(N,Co,W), ...]*len(Ks)
        h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h] #[(N,Co), ...]*len(Ks)
        h = torch.cat(h, 1)
        return h

    def pad_feature(self, features: List[torch.Tensor]) -> torch.Tensor:
        max_len = max([len(e) for e in features])

        def init_pad_h():
            pad_h = torch.zeros(1, self.conv_input_size)
            if self.use_cuda:
                pad_h = pad_h.cuda(non_blocking=True)
            return pad_h

        if max_len == 0:
            return torch.cat([init_pad_h() for _ in features], 0).unsqueeze(1)

        f = []
        for feature in features:
            feature = feature + [init_pad_h() for e in range(max_len - len(feature))]
            feature = torch.cat(feature, 0) # seq_size x conv_input_size
            f.append(feature.unsqueeze(0)) # 1 x seq_size x conv_input_size
        return torch.cat(f, 0) # batch_size x seq_size x conv_input_size


    def generate_word_reprsentation(
            self, 
            seq_encoder: torch.Tensor, 
            seq_lens: List[int],
            ent_labels: List[torch.LongTensor]) -> List[torch.Tensor]:
        seq_encoder = torch.stack(seq_encoder, 0) 
        ent_labels = torch.stack(ent_labels, 0)
        ent_embs = self.ent_emb_oh[ent_labels]
        batch_word_repr = torch.cat([seq_encoder, ent_embs], 2)
        return batch_word_repr
