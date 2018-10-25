#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/11 16:32:52

@author: Changzhi Sun
"""
from collections import Counter
from typing import List
import numpy as np

from antNRE.lib.k_means import KMeans

class DataLoader:

   def __init__(self,
		datasets: List,
		n_bkts: int) -> None:
       len_counter = Counter()
       for instance in datasets: 
           len_counter[len(instance['tokens'])] += 1
       self._bucket_sizes = KMeans(n_bkts, len_counter).splits
       self._buckets = [[] for i in range(n_bkts)]
       len2bkt = {}
       prev_size = -1
       for bkt_idx, size in enumerate(self._bucket_sizes):
           len2bkt.update(zip(range(prev_size+1, size+1), [bkt_idx] * (size - prev_size)))
           prev_size = size

       self._record = []
       for instance in datasets: 
           bkt_idx = len2bkt[len(instance['tokens'])]
           self._buckets[bkt_idx].append(instance)
           idx = len(self._buckets[bkt_idx]) - 1
           self._record.append((bkt_idx, idx))

   def get_batches(self, batch_size: int, shuffle: bool = True) -> List:
       batches = []
       for bkt_idx, bucket in enumerate(self._buckets):
           bucket_len = len(bucket)
           print(bucket_len, self._bucket_sizes[bkt_idx])
           n_tokens = bucket_len * self._bucket_sizes[bkt_idx]
           n_splits = max(n_tokens // batch_size, 1)
           range_func = np.random.permutation if shuffle else np.arange
           for bkt_batch in np.array_split(range_func(bucket_len), n_splits):
               batches.append((bkt_idx, bkt_batch))

       if shuffle:
           np.random.shuffle(batches)
       return batches

   def get_batch_instance(self, batch) -> List:
       bkt_idx, bkt_instance_idxes = batch
       return [self._buckets[bkt_idx][bkt_instance_idx] 
               for bkt_instance_idx in bkt_instance_idxes]

   def get_datasets(self) -> List:
       return [self._buckets[bkt_idx][bkt_instance_idx] for bkt_idx, bkt_instance_idx in self._record]
