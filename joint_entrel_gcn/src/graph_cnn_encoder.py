#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/01/12 08:18:27

@author: Changzhi Sun
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvoluation(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bias: bool = True) -> None:
        super(GraphConvoluation, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,
                inputs: torch.FloatTensor,
                adj: torch.sparse.FloatTensor) -> torch.FloatTensor:
        support = torch.mm(inputs, self.weight)
        #  print(support)
        output = torch.spmm(adj, support)
        #  print(output)
        #  print("======")
        if self.bias is not None:
            return output + self.bias
        return output


    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


    def __repr__(self):
        return (self.__class__.__name__ + ' ('
                + str(self.input_dim) + ' -> '
                + str(self.output_dim) + ')')


class GCN(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 beta: float = 0.8,
                 dropout: float = 0.5) -> None:
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.beta = beta

        self.gc = nn.ModuleList([GraphConvoluation(input_dim if i == 0 else hidden_dim,
                                               hidden_dim) 
                             for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                inputs: torch.FloatTensor,
                adj: torch.sparse.FloatTensor) -> torch.FloatTensor:
        for i in range(self.num_layers):
            inputs = inputs * self.beta + (1 - self.beta) * F.relu(self.gc[i](inputs, adj))
            #  inputs = F.relu(self.gc[i](inputs, adj))
            inputs = self.dropout(inputs)
        return inputs


    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.hidden_dim

#  g = GCN(10, 10, 1)
#  inputs = torch.randn(2, 10)
#  i = torch.LongTensor([[0, 1],
                      #  [1, 0]])
#  v = torch.FloatTensor([1, 1])
#  adj = torch.sparse.FloatTensor(i, v, torch.Size([2, 2]))
#  print(adj)
#  print(torch.spmm(adj, inputs))
#  print(adj.to_dense())
#  print(torch.spmm(adj.to_dense(), inputs))
#  print(inputs)
#  print(g(inputs, adj))
