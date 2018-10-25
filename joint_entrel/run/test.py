#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/09/21 15:48:17

@author: Changzhi Sun
"""
import os
import sys
sys.path.append('..')
import argparse
import json
from typing import Dict, List, Any
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from config import Configurable
from antNRE.lib import vocabulary, util
from antNRE.src.seq_encoder import BiLSTMEncoder
from antNRE.src.seq_decoder import SeqSoftmaxDecoder
from antNRE.src.decoder import VanillaSoftmaxDecoder
from antNRE.src.word_encoder import WordCharEncoder
from entrel_eval import eval_file
from src.ent_model import EntModel
from src.rel_encoder import RelFeatureExtractor
from src.joint_model import JointModel
import lib.util as myutil

torch.manual_seed(5216) # CPU random seed
np.random.seed(5216)

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_file', default='../configs/default.cfg')
args, extra_args = argparser.parse_known_args()
config = Configurable(args.config_file, extra_args)

use_cuda = config.use_cuda
# GPU and CPU using different random seed
if use_cuda:
    torch.cuda.manual_seed(5216)

train_corpus = myutil.load_corpus_from_json_file(config.train_file,
                                               config.save_dir,
                                               config.entity_schema)
dev_corpus = myutil.load_corpus_from_json_file(config.dev_file,
                                             config.save_dir,
                                             config.entity_schema)
test_corpus = myutil.load_corpus_from_json_file(config.test_file,
                                              config.save_dir,
                                              config.entity_schema)
max_sent_len = max([len(e['tokens']) for e in train_corpus + dev_corpus + test_corpus])
max_sent_len = min(max_sent_len, config.max_sent_len)
train_corpus = [e for e in train_corpus if len(e['tokens']) <= max_sent_len]
dev_corpus = [e for e in dev_corpus if len(e['tokens']) <= max_sent_len]
test_corpus = [e for e in test_corpus if len(e['tokens']) <= max_sent_len]
print("Total items in train corpus: %s" % len(train_corpus))
print("Total items in dev corpus: %s" % len(dev_corpus))
print("Total items in test corpus: %s" % len(test_corpus))
print("Max sentence length: %s" % max_sent_len)

namespace_counter = myutil.create_counter(train_corpus + dev_corpus + test_corpus)
for namespace in namespace_counter.keys():
    print(namespace, len(namespace_counter[namespace]))
tokens_to_add = {'rel_labels': ["None"]}
vocab = vocabulary.Vocabulary(namespace_counter, tokens_to_add=tokens_to_add)
print(vocab)
train_corpus = myutil.data2number(train_corpus, vocab)
dev_corpus = myutil.data2number(dev_corpus, vocab)
test_corpus = myutil.data2number(test_corpus, vocab)
word_encoder_size = config.word_dims + config.char_output_channels * len(config.char_kernel_sizes)
char_emb_kwargs = {
    'char_vocab_size': vocab.get_vocab_size('token_chars'),
    'char_dims': config.char_dims,
    'out_channels': config.char_output_channels,
    'kernel_sizes': config.char_kernel_sizes,
    'padding_idx': vocab.get_token_index(vocab._padding_token, 'token_chars'),
    'dropout': config.dropout,
}
word_encoder_kwargs = {
    'word_vocab_size': vocab.get_vocab_size('tokens'),
    'word_dims': config.word_dims,
    'char_emb_kwargs': char_emb_kwargs,
    'dropout': config.dropout,
    'padding_idx': vocab.get_token_index(vocab._padding_token, 'tokens'),
}
word_encoder = WordCharEncoder(**word_encoder_kwargs)

seq_encoder_kwargs = {
    'word_encoder_size': word_encoder_size,
    'hidden_size': config.lstm_hiddens,
    'num_layers': config.lstm_layers,
    'bidirectional': True,
    'dropout': config.dropout,
}
seq_encoder = BiLSTMEncoder(**seq_encoder_kwargs)
seq_decoder = SeqSoftmaxDecoder(hidden_size=config.lstm_hiddens,
                                tag_size=vocab.get_vocab_size("ent_labels"))
rel_feat_kwargs = {
    "word_encoder": word_encoder,
    "seq_encoder": seq_encoder,
    "vocab": vocab,
    "out_channels": config.rel_output_channels,
    "kernel_sizes": config.rel_kernel_sizes,
    "max_sent_len": max_sent_len,
    "dropout": config.dropout,
    "use_cuda": config.use_cuda
}
rel_decoder = VanillaSoftmaxDecoder(hidden_size=config.lstm_hiddens,
                                    tag_size=vocab.get_vocab_size("rel_labels"))
rel_feat_extractor = RelFeatureExtractor(**rel_feat_kwargs)
mymodel = JointModel(word_encoder,
                     seq_encoder,
                     seq_decoder, 
                     rel_feat_extractor,
                     rel_decoder,
                     config.schedule_k,
                     vocab,
                     config.use_cuda)

if config.use_cuda:
    mymodel.cuda()

if os.path.exists(config.load_model_path):
    state_dict = torch.load(
        open(config.load_model_path, "rb"),
        map_location=lambda storage, loc: storage)
    mymodel.load_state_dict(state_dict)
    print("Loading previous model successful [%s]" % config.load_model_path)

def create_batch_list(sort_batch_tensor: Dict[str, Any],
                      outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    new_batch = []
    for k in range(len(outputs['ent_pred'])):
        instance = {}
        instance['tokens'] = sort_batch_tensor['tokens'][k].cpu().numpy()
        instance['ent_labels'] = sort_batch_tensor['ent_labels'][k].cpu().numpy()
        instance['candi_rels'] = sort_batch_tensor['candi_rels'][k]
        instance['rel_labels'] = sort_batch_tensor['rel_labels'][k]
        instance['ent_pred'] = outputs['ent_pred'][k].cpu().numpy()
        instance['all_candi_rels'] = outputs['all_candi_rels'][k]
        instance['all_rel_pred'] = outputs['all_rel_pred'][k]
        new_batch.append(instance)
    return new_batch

def step(batch: List[Dict]) -> (List[Dict], Dict):
    sort_batch_tensor = myutil.get_minibatch(batch, vocab, config.use_cuda)
    sort_batch_tensor['i_epoch'] = None
    outputs = mymodel(sort_batch_tensor)
    new_batch = create_batch_list(sort_batch_tensor, outputs)
    batch_outputs = {}
    batch_outputs['ent_loss'] = outputs['ent_loss']
    batch_outputs['rel_loss'] = outputs['rel_loss']
    return new_batch, batch_outputs

def predict_all(corpus) -> None: 
    mymodel.eval()
    new_corpus = []
    for k in range(0,len(corpus), batch_size):
        print("[ %d / %d ]" % (len(corpus), min(len(corpus), k + batch_size)))
        batch = corpus[k: k + batch_size]
        new_batch, outputs = step(batch)
        new_corpus.extend(new_batch)
    return new_corpus

batch_size = config.batch_size
for title, corpus in zip( ["train", "dev", "test"], [train_corpus, dev_corpus, test_corpus]):
    if title == "train": continue
    print("\nEvaluating %s" % title)
    new_corpus = predict_all(corpus)
    eval_path = os.path.join(config.save_dir, "final.%s.output" % title)
    myutil.print_predictions(new_corpus, eval_path, vocab)
    eval_file(eval_path)
