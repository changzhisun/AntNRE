#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/01/12 10:40:34

@author: Changzhi Sun
"""
import sys
sys.path.append("..")
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from src.graph_cnn_encoder import GCN

class GCNExtractor(nn.Module):

    def __init__(self,
                 gcn: GCN,
                 use_cuda: bool) -> None:
        super(GCNExtractor, self).__init__()
        self.gcn = gcn
        self.use_cuda = use_cuda

    def forward(self,
                batch: Dict[str, Any],
                ent_feats: torch.FloatTensor,
                rel_batch: Dict[str, Any],
                bin_rel_pred: Optional[torch.LongTensor]) -> (torch.FloatTensor, Optional[torch.FloatTensor]):
        all_ent_gcn_feats = []
        all_rel_gcn_feats = []
        rel_feats = rel_batch['inputs'] if rel_batch is not None else None
        i_ent = 0
        i_rel = 0
        #  print(rel_feats.size())
        for i in range(len(batch['tokens'])):
            cur_candi_rels = batch['all_candi_rels'][i]
            cur_ent_ids = batch['all_ent_ids'][i]
            cur_candi_rels_len = len(cur_candi_rels)
            cur_ent_ids_len = len(cur_ent_ids)

            if cur_ent_ids_len == 0:
                assert cur_candi_rels_len == 0
                continue
            ent_inputs =  ent_feats[i_ent: i_ent + cur_ent_ids_len]
            if cur_candi_rels_len == 0:
                graph_inputs = ent_inputs
                #  adj = np.ones((cur_ent_ids_len, cur_ent_ids_len))
                #  adj[range(cur_ent_ids_len), range(cur_ent_ids_len)] = 0
                adj = np.zeros((cur_ent_ids_len, cur_ent_ids_len))
                adj = sp.coo_matrix(adj)
            else:
                rel_inputs = rel_feats[i_rel: i_rel + cur_candi_rels_len]
                graph_inputs = torch.cat([ent_inputs, rel_inputs], 0)

                ent2id = {k: v for v, k in enumerate(cur_ent_ids)}
                #  print(ent2id)
                adj = np.zeros((cur_candi_rels_len + cur_ent_ids_len, cur_candi_rels_len + cur_ent_ids_len))
                #  adj[:cur_ent_ids_len, : cur_ent_ids_len] = 1.0
                #  adj[range(cur_ent_ids_len), range(cur_ent_ids_len)] = 0
                for j in range(cur_candi_rels_len):
                    if bin_rel_pred is not None and bin_rel_pred[i_rel + j].item() == 0:
                        continue
                    e1, e2 = cur_candi_rels[j]
                    adj[ent2id[e1], cur_ent_ids_len + j] = 1.0
                    adj[ent2id[e2], cur_ent_ids_len + j] = 1.0
                    adj[cur_ent_ids_len + j, ent2id[e1]] = 1.0
                    adj[cur_ent_ids_len + j, ent2id[e2]] = 1.0
                    #  print(cur_ent_ids_len + j, e1, e2)
                adj = sp.coo_matrix(adj)
                #  print(adj)
                #  print("====")

            adj = self.normalize_adj(adj + sp.eye(adj.shape[0]))
            #  adj = self.normalize(adj + sp.eye(adj.shape[0]))
            adj = self.scipy2torch_sparse(adj)

            #  adj = torch.eye(cur_ent_ids_len + cur_candi_rels_len).cuda(non_blocking=True)

            cur_graph_feats = self.gcn(graph_inputs, adj)
            cur_ent_gcn_feats = cur_graph_feats[:cur_ent_ids_len]
            all_ent_gcn_feats.append(cur_ent_gcn_feats)
            if cur_candi_rels_len != 0:
                cur_rel_gcn_feats = cur_graph_feats[cur_ent_ids_len: cur_ent_ids_len + cur_candi_rels_len]
                all_rel_gcn_feats.append(cur_rel_gcn_feats)
            i_ent = i_ent + cur_ent_ids_len
            i_rel = i_rel + cur_candi_rels_len
        all_ent_gcn_feats = torch.cat(all_ent_gcn_feats, 0)
        if rel_batch is not None:
            all_rel_gcn_feats = torch.cat(all_rel_gcn_feats, 0)
        else:
            all_rel_gcn_feats = None
        return all_ent_gcn_feats, all_rel_gcn_feats
    
    def get_input_dim(self) -> int:
        return self.gcn.get_input_dim()
    
    def get_output_dim(self) -> int:
        return self.gcn.get_output_dim()

    def normalize_adj(self, mx: sp.coo_matrix) -> sp.coo_matrix:
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

    def normalize(self, mx: sp.coo_matrix) -> sp.coo_matrix:
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).tocoo()
        return mx

    def scipy2torch_sparse(self, mx: sp.coo_matrix) -> torch.sparse.FloatTensor:
        values = mx.data
        indices = np.vstack((mx.row, mx.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = mx.shape
        mx = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        if self.use_cuda:
            mx = mx.cuda(non_blocking=True)
        return mx
