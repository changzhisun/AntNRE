#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/25 15:15:27

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


def get_idx2ent_and_tags(entity_mentions : List[Dict],
                         n : int,
                         entity_schema: str) -> Tuple:
    tags = ['O'] * n
    idx2ent = {}
    for men in entity_mentions:
        idx2ent[men['emId']] = (men['offset'], men['label'], men['text'])
        s, e = men['offset']
        if entity_schema == "BIO":
            tags[s] = 'B-' + men['label']
            for j in range(s+1, e):
                tags[j] = 'I-' + men['label']
        else:
            if e - s == 1:
                tags[s] = "U-" + men['label']
            elif e - s == 2:
                tags[s] = 'B-' + men['label']
                tags[s+1] = 'E-' + men['label']
            else:
                tags[s] = 'B-' + men['label']
                tags[e - 1] = 'E-' + men['label']
                for j in range(s+1, e - 1):
                    tags[j] = 'I-' + men['label']
    return idx2ent, tags


def format_json_file(json_file: str, save_file: str, entity_schema: str) -> None:
    with open(json_file, "r", encoding="utf8") as fj:
        with open(save_file, "w", encoding="utf8") as fs:
            for line in fj:
                sent = json.loads(line)
                tokens = sent['sentText'].split(' ')

                idx2ent, tags = get_idx2ent_and_tags(sent['entityMentions'],
                                                     len(tokens),
                                                     entity_schema)
                for w, t in zip(tokens, tags):
                    print("{0}\t{1}\t{2}".format(w, t[0], t), file=fs)

                for men in sent['entityMentions']:
                    print("{0}\t{1}".format(tuple(men['offset']), men['label']), file=fs)

                for men in sent['relationMentions']:
                    em1_idx, em1_t, em1_x = idx2ent[men['em1Id']]
                    em2_idx, em2_t, em2_x = idx2ent[men['em2Id']]
                    em1_text = men['em1Text']
                    em2_text = men['em2Text']

                    assert em1_text == em1_x
                    assert em2_text == em2_x

                    direction = "-->"
                    if em1_idx[0] > em2_idx[0]:
                        direction = "<--"
                        em1_idx, em2_idx = em2_idx, em1_idx
                        em1_text, em2_text = em2_text, em1_text
                        em1_t, em2_t = em2_t, em1_t

                    if em1_idx[1] > em2_idx[0]:
                        continue

                    label = men['label'] + direction
                    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(
                           tuple(em1_idx), em1_text, em1_t,
                           tuple(em2_idx), em2_text, em2_t,
                           label), file=fs)
                print(file=fs)


def load_corpus_from_json_file(json_file: str,
                               save_dir: str,
                               entity_schema: str) -> List[Dict]:
    basename = os.path.basename(json_file)
    save_file = os.path.join(save_dir, basename.split('.')[0] + ".format")

    format_json_file(json_file, save_file, entity_schema)
    
    with open(save_file, "r", encoding="utf8") as f:
        instances = []
        instance = defaultdict(list)
        for line in f:
            line = line.rstrip()
            if line:
                line = line.split('\t')
                if len(line) == 3:
                    instance['tokens'].append(line[0])
                    instance['ent_span_labels'].append(line[1])
                    instance['ent_labels'].append(line[2])
                elif len(line) == 2:
                    instance['ent_ids'].append(eval(line[0]))
                    instance['ent_ids_labels'].append(line[1])
                elif len(line) == 7:
                    candi_rel = (eval(line[0]), eval(line[3]))
                    instance['candi_rels'].append(candi_rel)
                    instance['rel_labels'].append(line[-1])
            else:
                if instance:
                    assert len(instance['tokens']) == len(instance['ent_labels'])
                    assert len(instance['tokens']) == len(instance['ent_span_labels'])
                    assert len(instance['candi_rels']) == len(instance['rel_labels'])
                    assert len(instance['ent_ids']) == len(instance['ent_ids_labels'])
                    instances.append(instance)
                instance = defaultdict(list)
        if instance:
            assert len(instance['tokens']) == len(instance['ent_labels'])
            assert len(instance['tokens']) == len(instance['ent_span_labels'])
            assert len(instance['candi_rels']) == len(instance['rel_labels'])
            assert len(instance['ent_ids']) == len(instance['ent_ids_labels'])
            instances.append(instance)
    return instances 


def count_vocab_items(instance: Dict,
                      namespace: str,
                      namespace_counter: Dict[str, Dict[str, int]],
                      lower_case: bool) -> None:
    if namespace == "token_chars":
        for token in instance['tokens']:
            for ch in token:
                ch = ch.lower() if lower_case else ch
                namespace_counter[namespace][ch] += 1
    else:
        for item in instance[namespace]:
            item = item.lower() if lower_case else item
            namespace_counter[namespace][item] += 1

def create_counter(instances: List[Dict]) -> Dict[str, Dict[str, int]]:
    namespace_counter: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for instance in instances:
        count_vocab_items(instance, "tokens", namespace_counter, True)
        count_vocab_items(instance, "token_chars", namespace_counter, False)
        count_vocab_items(instance, "ent_labels", namespace_counter, False)
        count_vocab_items(instance, "ent_span_labels", namespace_counter, False)
        count_vocab_items(instance, "ent_ids_labels", namespace_counter, False)
        count_vocab_items(instance, "rel_labels", namespace_counter, False)
    return namespace_counter

def data2number(corpus: List[Dict], vocab: Vocabulary) -> List[Dict]:
    instances = []
    oov_idx = vocab.get_token_index(vocab._oov_token, 'tokens')
    for e in corpus:
        instance = {}
        instance['tokens'] = seq2number(e, vocab, 'tokens', True)
        instance['token_chars'] = seqchar2number(e, vocab, False)
        instance['ent_labels'] = seq2number(e, vocab, 'ent_labels', False)
        instance['ent_span_labels'] = seq2number(e, vocab, 'ent_span_labels', False)
        instance['ent_ids_labels'] = seq2number(e, vocab, 'ent_ids_labels', False)
        instance['rel_labels'] = seq2number(e, vocab, 'rel_labels', False)
        instance['candi_rels'] = e['candi_rels']
        instance['ent_ids'] = e['ent_ids']
        
        assert all([oov_idx != n for n in instance['tokens']])
        assert all([oov_idx != m for n in instance['token_chars'] for m in n])

        instances.append(instance)
    return instances

def seq2number(instance: Dict,
               vocab: Vocabulary, 
               namespace: str,
               lower_case: bool) -> List:
    return [vocab.get_token_index(item.lower() if lower_case else item, namespace)
            for item in instance[namespace]]

def seqchar2number(instance: Dict,
                   vocab: Vocabulary, 
                   lower_case: bool) -> List[List]:
    nums = []
    for token in instance['tokens']:
        nums.append([vocab.get_token_index(item.lower() if lower_case else item, 'token_chars')
                     for item in token])
    return nums

def get_minibatch(batch: List[Dict], vocab: Vocabulary, use_cuda: bool) -> Dict[str, Any]:
    batch = sorted(batch, key=lambda x : len(x['tokens']), reverse=True)
    batch_seq_len = [len(instance['tokens']) for instance in batch]
    max_seq_len = max(batch_seq_len)
    max_char_seq_len = max([len(tok) for instance in batch for tok in instance['token_chars']])

    outputs = defaultdict(list)
    token_padding_idx = vocab.get_token_index(vocab._padding_token, 'tokens')
    char_padding_idx = vocab.get_token_index(vocab._padding_token, 'token_chars')
    label_padding_idx = -1
    for instance in batch:
        cur_seq_len = len(instance['tokens'])

        outputs['tokens'].append(instance['tokens'] + [token_padding_idx] * (max_seq_len - cur_seq_len))
        outputs['ent_labels'].append(instance['ent_labels'] + [label_padding_idx] * (max_seq_len - cur_seq_len))
        outputs['ent_span_labels'].append(instance['ent_span_labels'] + [label_padding_idx] * (max_seq_len - cur_seq_len))
        outputs['candi_rels'].append(instance['candi_rels'])
        outputs['ent_ids'].append(instance['ent_ids'])
        outputs['ent_ids_labels'].append(instance['ent_ids_labels'])
        outputs['rel_labels'].append(instance['rel_labels'])
        char_pad = []
        for char_seq in instance['token_chars']:
            char_pad.append(char_seq + [char_padding_idx] * (max_char_seq_len - len(char_seq)))
        char_pad = char_pad + [[char_padding_idx] * max_char_seq_len] * (max_seq_len - cur_seq_len)
        outputs['token_chars'].append(char_pad)
    outputs['tokens'] = torch.LongTensor(outputs['tokens'])
    outputs['token_chars'] = torch.LongTensor(outputs['token_chars'])
    outputs['ent_labels'] = torch.LongTensor(outputs['ent_labels'])
    outputs['ent_span_labels'] = torch.LongTensor(outputs['ent_span_labels'])
    outputs['seq_lens'] = batch_seq_len
    if use_cuda:
        outputs['tokens'] = outputs['tokens'].cuda(non_blocking=True)
        outputs['token_chars'] = outputs['token_chars'].cuda(non_blocking=True)
        outputs['ent_labels'] = outputs['ent_labels'].cuda(non_blocking=True)
        outputs['ent_span_labels'] = outputs['ent_span_labels'].cuda(non_blocking=True)
    return outputs

def print_predictions(datasets: List,
                      filename: str,
                      vocab: Vocabulary) -> None:
    with open(filename, "w", encoding="utf8") as f:
        for instance in datasets:
            seq_len = int((instance['ent_span_labels'] >= 0).sum())
            for idx, true_label, pred_label in zip(instance['tokens'][:seq_len],
                                                   instance['ent_labels'][:seq_len],
                                                   instance['all_ent_pred'][:seq_len]):
                token = vocab.get_token_from_index(idx, "tokens")
                true_label = vocab.get_token_from_index(true_label, "ent_labels")
                pred_label = vocab.get_token_from_index(pred_label, "ent_labels")
                #  if true_label != "O":
                    #  true_label = true_label + "-ENT"
                #  if pred_label != "O":
                    #  pred_label = pred_label + "-ENT"
                print("{}\t{}\t{}".format(token, true_label, pred_label), file=f)
            
            for (s, e), r in zip(instance['candi_rels'], instance['rel_labels']):
                r = vocab.get_token_from_index(r, "rel_labels")

                assert r != "None"

                if r[-3:] == "<--":
                    s, e = e, s
                r = r[:-3]
                print("Rel-True\t{}\t{}\t{}".format(s, e, r), file=f)

            for (s, e), r in zip(instance['all_candi_rels'], instance['all_rel_pred']):
                r = vocab.get_token_from_index(r, "rel_labels")
                if r == "None":
                    continue
                if r[-3:] == "<--":
                    s, e = e, s
                r = r[:-3]
                print("Rel-Pred\t{}\t{}\t{}".format(s, e, r), file=f)
            print(file=f)

def print_ent_span_predictions(datasets: List,
                               filename: str,
                               vocab: Vocabulary) -> None:
    with open(filename, "w", encoding="utf8") as f:
        for instance in datasets:
            seq_len = int((instance['ent_span_labels'] >= 0).sum())
            for idx, true_label, pred_label in zip(instance['tokens'][:seq_len],
                                                   instance['ent_span_labels'][:seq_len],
                                                   instance['ent_span_pred'][:seq_len]):
                token = vocab.get_token_from_index(idx, "tokens")
                true_label = vocab.get_token_from_index(true_label, "ent_span_labels")
                pred_label = vocab.get_token_from_index(pred_label, "ent_span_labels")
                if true_label != "O":
                    true_label = true_label + "-ENT"
                if pred_label != "O":
                    pred_label = pred_label + "-ENT"
                print("{}\t{}\t{}".format(token, true_label, pred_label), file=f)
            print(file=f)
