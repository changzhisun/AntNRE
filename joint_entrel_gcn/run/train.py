#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/09/21 15:48:17

@author: Changzhi Sun
"""
import os
import sys
sys.path.append("..")
import argparse
import json
from typing import Dict, List, Any
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor import BidirectionalEndpointSpanExtractor
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor

from config import Configurable
from antNRE.lib import vocabulary, util
from antNRE.modules.seq2seq_encoders.seq2seq_bilstm import BiLSTMEncoder
from antNRE.src.seq_decoder import SeqSoftmaxDecoder
from antNRE.src.decoder import VanillaSoftmaxDecoder
from antNRE.src.word_encoder import WordCharEncoder
from antNRE.modules.span_extractors.sum_span_extractor import SumSpanExtractor
from antNRE.modules.span_extractors.cnn_span_extractor import CnnSpanExtractor
from entrel_eval import eval_file
from entrel_eval import Metrics
from src.joint_model import JointModel
from src.ent_span_feat_extractor import EntSpanFeatExtractor
from src.rel_feat_extractor import RelFeatExtractor
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

train_corpus = myutil.load_corpus_from_json_file(
    config.train_file, config.save_dir, config.entity_schema)
dev_corpus = myutil.load_corpus_from_json_file(
    config.dev_file, config.save_dir, config.entity_schema)
test_corpus = myutil.load_corpus_from_json_file(
    config.test_file, config.save_dir, config.entity_schema)

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

#  tokens_to_add = {'rel_labels': ["None"], 'ent_ids_labels': ["None"]}
tokens_to_add = {'rel_labels': ["None"], "ent_ids_labels": ["None"]}
vocab = vocabulary.Vocabulary(namespace_counter, tokens_to_add=tokens_to_add)
print(vocab)

train_corpus = myutil.data2number(train_corpus, vocab)
dev_corpus = myutil.data2number(dev_corpus, vocab)
test_corpus = myutil.data2number(test_corpus, vocab)

pretrained_embeddings = util.load_word_vectors(config.pretrained_embeddings_file,
                                               config.word_dims,
                                               vocab)
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

seq2seq_encoder_kwargs = {
    'input_size': word_encoder_size,
    'hidden_size': config.lstm_hiddens,
    'num_layers': config.lstm_layers,
    'bidirectional': True,
    'dropout': config.dropout,
}
seq2seq_encoder = BiLSTMEncoder(**seq2seq_encoder_kwargs)
ent_span_decoder = SeqSoftmaxDecoder(hidden_size=seq2seq_encoder.get_output_dim(),
                                     tag_size=vocab.get_vocab_size("ent_span_labels"))
#  ent_ids_span_extractor = EndpointSpanExtractor(
    #  input_dim = config.lstm_hiddens)
#  ent_ids_span_extractor = SumSpanExtractor(
    #  input_dim = config.lstm_hiddens)
ent_ids_span_extractor = CnnSpanExtractor(
    config.lstm_hiddens,
    config.rel_output_channels,
    config.rel_kernel_sizes)
print(ent_ids_span_extractor)
ent_span_feat_extractor = EntSpanFeatExtractor(
    config.lstm_hiddens,
    ent_ids_span_extractor,
    config.dropout,
    config.use_cuda)

#  context_span_extractor = BidirectionalEndpointSpanExtractor(
    #  input_dim = config.lstm_hiddens)
#  context_span_extractor = SumSpanExtractor(
    #  input_dim = config.lstm_hiddens)
context_span_extractor = CnnSpanExtractor(
    config.lstm_hiddens,
    config.rel_output_channels,
    config.rel_kernel_sizes)
print(context_span_extractor)
rel_feat_extractor = RelFeatExtractor(
    config.lstm_hiddens,
    context_span_extractor,
    config.dropout,
    config.use_cuda)


ent_ids_decoder = VanillaSoftmaxDecoder(hidden_size=config.lstm_hiddens,
                                        tag_size=vocab.get_vocab_size("ent_ids_labels"))
rel_decoder = VanillaSoftmaxDecoder(hidden_size=config.lstm_hiddens,
                                    tag_size=vocab.get_vocab_size("rel_labels"))
mymodel = JointModel(word_encoder,
                     seq2seq_encoder,
                     ent_span_decoder, 
                     ent_span_feat_extractor,
                     ent_ids_decoder,
                     rel_feat_extractor,
                     rel_decoder,
                     vocab,
                     config.use_cuda)

util.assign_embeddings(word_encoder.word_embeddings, pretrained_embeddings)
if config.use_cuda:
    mymodel.cuda()

if os.path.exists(config.load_model_path):
    state_dict = torch.load(
        open(config.load_model_path, "rb"),
        map_location=lambda storage, loc: storage)
    mymodel.load_state_dict(state_dict)
    print("Loading previous model successful [%s]" % config.load_model_path)

parameters = [p for p in mymodel.parameters() if p.requires_grad]
optimizer = optim.Adadelta(parameters)

def create_batch_list(sort_batch_tensor: Dict[str, Any],
                      outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    new_batch = []
    for k in range(len(outputs['ent_span_pred'])):
        instance = {}
        instance['tokens'] = sort_batch_tensor['tokens'][k].cpu().numpy()
        instance['ent_labels'] = sort_batch_tensor['ent_labels'][k].cpu().numpy()
        instance['ent_span_labels'] = sort_batch_tensor['ent_span_labels'][k].cpu().numpy()

        instance['candi_rels'] = sort_batch_tensor['candi_rels'][k]
        instance['rel_labels'] = sort_batch_tensor['rel_labels'][k]
        instance['ent_span_pred'] = outputs['ent_span_pred'][k].cpu().numpy()
        instance['all_ent_pred'] = outputs['all_ent_pred'][k]
        instance['all_candi_rels'] = outputs['all_candi_rels'][k]
        instance['all_rel_pred'] = outputs['all_rel_pred'][k]
        #  print(instance['all_rel_pred'])
        assert len(instance['all_candi_rels']) == len(instance['all_rel_pred'])
        new_batch.append(instance)
    return new_batch

def step(batch: List[Dict]) -> (List[Dict], Dict):
    sort_batch_tensor = myutil.get_minibatch(batch, vocab, config.use_cuda)
    outputs = mymodel(sort_batch_tensor)
    new_batch = create_batch_list(sort_batch_tensor, outputs)
    batch_outputs = {}
    batch_outputs['ent_span_loss'] = outputs['ent_span_loss']
    batch_outputs['ent_ids_loss'] = outputs['ent_ids_loss']
    batch_outputs['rel_loss'] = outputs['rel_loss']
    return new_batch, batch_outputs

def train_step(batch: List[Dict]) -> None:
    optimizer.zero_grad()
    mymodel.train()
    _, outputs = step(batch)
    loss = outputs['ent_span_loss'] + outputs['ent_ids_loss'] + outputs['rel_loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, config.clip_c)
    optimizer.step()
    print("Epoch : %d Minibatch : %d Loss : %.5f(%.5f, %.5f, %.5f)" % (
          i, j, loss.item(),
          outputs['ent_span_loss'].item(),
          outputs['ent_ids_loss'].item(),
          outputs['rel_loss'].item()))
    writer.add_scalar("Train/Loss", loss.item(), num_iter)
    writer.add_scalar("Train/EntSpanLoss", outputs['ent_span_loss'].item(), num_iter)
    writer.add_scalar("Train/EntLoss", outputs['ent_ids_loss'].item(), num_iter)
    writer.add_scalar("Train/RelLoss", outputs['rel_loss'].item(), num_iter)


def dev_step() -> float: 
    optimizer.zero_grad()
    mymodel.eval()
    new_corpus = []
    ent_span_losses = []
    ent_ids_losses = []
    rel_losses = []
    for k in range(0, len(dev_corpus), batch_size):
        batch = dev_corpus[k: k + batch_size]
        new_batch, outputs = step(batch)
        new_corpus.extend(new_batch)
        ent_span_losses.append(outputs['ent_span_loss'].item())
        ent_ids_losses.append(outputs['ent_ids_loss'].item())
        rel_losses.append(outputs['rel_loss'].item())
    avg_ent_span_loss = np.mean(ent_span_losses)
    avg_ent_ids_loss = np.mean(ent_ids_losses)
    avg_rel_loss = np.mean(rel_losses)
    loss = avg_ent_span_loss + avg_ent_ids_loss + avg_rel_loss

    print("Epoch : %d Avg Loss : %.5f(%.5f, %.5f, %.5f)" % (
          i, loss,
          avg_ent_span_loss, 
          avg_ent_ids_loss,
          avg_rel_loss))
    writer.add_scalar("Dev/Loss", loss, num_iter)
    writer.add_scalar("Dev/EntSpanLoss", avg_ent_span_loss, num_iter)
    writer.add_scalar("Dev/EntLoss", avg_ent_ids_loss, num_iter)
    writer.add_scalar("Dev/RelLoss", avg_rel_loss, num_iter)

    eval_path = os.path.join(config.save_dir, "validate.dev.output")
    eval_ent_span_path = os.path.join(config.save_dir, "validate.dev.output.ent_span")
    myutil.print_predictions(new_corpus, eval_path, vocab)
    #  myutil.print_ent_span_predictions(new_corpus, eval_ent_span_path, vocab)
    entity, relation= eval_file(eval_path)
    #  eval_file(eval_ent_span_path)


    writer.add_scalar("Dev/EntPrecision", entity.prec, num_iter)
    writer.add_scalar("Dev/EntRecall", entity.rec, num_iter)
    writer.add_scalar("Dev/EntFscore", entity.fscore, num_iter)
    writer.add_scalar("Dev/RelPrecision", relation.prec, num_iter)
    writer.add_scalar("Dev/RelRecall", relation.rec, num_iter)
    writer.add_scalar("Dev/RelFscore", relation.fscore, num_iter)
    return relation.fscore


batch_size = config.batch_size
best_f1 = 0.0
cur_patience = 0
num_iter = 0
writer = SummaryWriter(os.path.join(config.save_dir, "runs"))
for i in range(config.train_iters):
    np.random.shuffle(train_corpus)

    for j in range(0, len(train_corpus), batch_size):

        batch = train_corpus[j: j + batch_size]

        train_step(batch)
        num_iter += 1

    print("Evaluating Model on dev set ...")

    dev_f1 = dev_step()
    cur_patience += 1
    if dev_f1 > best_f1:
        cur_patience = 0
        best_f1 = dev_f1
        print("Saving model ...")
        torch.save(mymodel.state_dict(),
                    open(os.path.join(config.save_dir, "minibatch", "epoch__%d_model" % i), "wb"))
        torch.save(mymodel.state_dict(), open(config.save_model_path, "wb"))
    if cur_patience > config.patience:
        break
