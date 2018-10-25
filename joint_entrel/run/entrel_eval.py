#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/16 11:12:20

@author: Changzhi Sun
"""
import sys
import re

from collections import defaultdict, namedtuple


ANY_SPACE = '<SPACE>'

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

class EvalCounts(object):

    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def evaluate(iterable, options=None):
    if options is None:
        options = parse_args([])    # use defaults
    entity_counts = EvalCounts()
    relation_counts = EvalCounts()

    sents = []
    sent = [[], [], []]
    for line in iterable:
        line = line.rstrip('\r\n')
        if line == "":
            if len(sent[0]) != 0:
                sents.append(sent)
                sent = [[], [], []]
        else:
            features = line.split('\t')
            if len(features) == 3:
                sent[0].append(features)
            elif len(features) == 4:
                if features[0] == "Rel-Pred":
                    sent[1].append([eval(features[1]), eval(features[2]), features[3]])
                elif features[0] == "Rel-True":
                    sent[2].append([eval(features[1]), eval(features[2]), features[3]])
    if len(sent[0]) != 0:
        sents.append(sent)
    for i, sent in enumerate(sents):
        eval_instance(sent, entity_counts, relation_counts)
    return entity_counts, relation_counts


def eval_instance(sent, entity_counts, relation_counts):

    t_correct_entity2idx = defaultdict(list)
    t_guessed_entity2idx= defaultdict(list)
    sent_len = len(sent[0])

    get_instance_entity(t_correct_entity2idx, t_guessed_entity2idx, sent)

    all_keys = set(t_guessed_entity2idx.keys()) | set(t_correct_entity2idx.keys())
    for key in all_keys:
        entity_counts.found_correct += len(t_correct_entity2idx[key])
        entity_counts.found_guessed += len(t_guessed_entity2idx[key])
        entity_counts.t_found_correct[key] += len(t_correct_entity2idx[key])
        entity_counts.t_found_guessed[key] += len(t_guessed_entity2idx[key])
        correct_chunk_set = set(t_correct_entity2idx[key]) & set(t_guessed_entity2idx[key])
        entity_counts.correct_chunk += len(correct_chunk_set)
        entity_counts.t_correct_chunk[key] += len(correct_chunk_set)

    idx2t_correct_entity = { e : k for k, v in t_correct_entity2idx.items() for e in v}
    idx2t_guessed_entity = { e : k for k, v in t_guessed_entity2idx.items() for e in v}

    t_correct_relexact2idx = defaultdict(set)
    t_guessed_relexact2idx = defaultdict(set)
    #  print("====")
    #  print([e[0] for e in sent[0]])
    #  print(t_correct_entity2idx)
    for b, e, r in sent[2]:
        #  if b[-1] < e[0] or e[-1] < b[0]:
        b = tuple(range(b[0], b[-1]))
        e = tuple(range(e[0], e[-1]))
        # NYT data have overlap relation
        if set(b) & set(e):
            continue
        # Only for NYT
        if b not in idx2t_correct_entity or e not in idx2t_correct_entity:
            continue
        #  t_correct_relexact2idx[r].add((b, e,)) # doesn't consider entity type
        t_correct_relexact2idx[r].add((b, idx2t_correct_entity[b], e, idx2t_correct_entity[e])) # consider entity type
    #  print(t_correct_relexact2idx)

    for b, e, r in sent[1]:
        #  if b[-1] < e[0] or e[-1] < b[0]:
        b = tuple(range(b[0], b[-1]))
        e = tuple(range(e[0], e[-1]))
        #  if b not in t_guessed_relexact2idx or e not in t_guessed_relexact2idx:
            #  continue
        #  t_guessed_relexact2idx[r].add((b, e))  # doesn't consider entity type
        t_guessed_relexact2idx[r].add((b, idx2t_guessed_entity[b], e, idx2t_guessed_entity[e]))  # consider entity type

    all_keys = set(t_guessed_relexact2idx.keys()) | set(t_correct_relexact2idx.keys())
    for key in all_keys:
        relation_counts.found_correct += len(t_correct_relexact2idx[key])
        relation_counts.found_guessed += len(t_guessed_relexact2idx[key])
        relation_counts.t_found_correct[key] += len(t_correct_relexact2idx[key])
        relation_counts.t_found_guessed[key] += len(t_guessed_relexact2idx[key])
        correct_chunk_set = t_correct_relexact2idx[key] & t_guessed_relexact2idx[key]
        relation_counts.correct_chunk += len(correct_chunk_set)
        relation_counts.t_correct_chunk[key] += len(correct_chunk_set)
        #  print(key)
        #  print(t_correct_relhead2idx[key])
        #  print(t_guessed_relhead2idx[key])
        #  print(correct_chunk_set)
        #  print("####")
    return entity_counts, relation_counts

def get_instance_entity(t_correct_entity2idx, t_guessed_entity2idx, sent):
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    correct_idx = []
    guessed_idx = []
    for i, features in enumerate(sent[0]):
        #  print(i, features)
        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)
        if start_correct:
            if correct_idx:
                t_correct_entity2idx[correct_idx[0]].append(tuple(correct_idx[1:]))
            correct_idx = [correct_type, i]
        elif correct_idx and not start_correct and correct_type == correct_idx[0]:
            correct_idx.append(i)

        if start_guessed:
            if guessed_idx:
                t_guessed_entity2idx[guessed_idx[0]].append(tuple(guessed_idx[1:]))
            guessed_idx = [guessed_type, i]
        elif guessed_idx and not start_guessed and guessed_type == guessed_idx[0]:
            guessed_idx.append(i)
        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if correct_idx:
        t_correct_entity2idx[correct_idx[0]].append(tuple(correct_idx[1:]))
    if guessed_idx:
        t_guessed_entity2idx[guessed_idx[0]].append(tuple(guessed_idx[1:]))

def end_of_chunk(prev_tag, tag, prev_type, type_):
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

def start_of_chunk(prev_tag, tag, prev_type, type_):
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

def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('-t', '--no-types', action='store_const', const=True, default=False,
        help='evaluate without entity types')
    arg('file', nargs='?', default=None)
    arg('--outstream', default=None,
        help='output file for storing report')
    return parser.parse_args(argv)


def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


def report(entity_counts, relation_counts, out=None):
    if out is None:
        out = sys.stdout
    print("--------------------------------Entity---------------------------------\n")
    entity_score = report_count(entity_counts, out)
    print("\n-------------------------------Relation--------------------------------\n")
    relation_score = report_count(relation_counts, out)
    print("\n-----------------------------------------------------------------------\n")
    return entity_score, relation_score

def report_by_sample(entity_counts, relation_counts, out=None):
    if out is None:
        out = sys.stdout
    entity_score = report_count_by_sample(entity_counts, out)
    relation_score = report_count_by_sample(relation_counts, out)
    return (entity_score + relation_score) / 2
    #  return entity_score
    #  return relation_score

def report_count(counts, out):

    overall, by_type = metrics(counts)

    c = counts

    #  out.write('processed %d tokens with %d phrases; ' %
              #  (c.token_counter, c.found_correct))

    #  out.write('found: %d phrases; correct: %d.\n' %
              #  (c.found_guessed, c.correct_chunk))

    #  if c.token_counter > 0:
        #  out.write('accuracy: %6.2f%%; ' %
                  #  (100.*c.correct_tags/c.token_counter))
    out.write('precision: %6.2f%%; ' % (100.*overall.prec))
    out.write('recall: %6.2f%%; ' % (100.*overall.rec))
    out.write('FB1: %6.2f\n' % (100.*overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100.*m.prec))
        out.write('recall: %6.2f%%; ' % (100.*m.rec))
        out.write('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))
    return overall.fscore

def report_count_by_sample(counts, out):
    overall, by_type = metrics(counts)
    if overall.tp == 0 and overall.fp == 0 and overall.fn == 0:
        return 1.0
    return overall.fscore

def eval_file(filename):
    with open(filename) as f:
        entity_counts, relation_counts = evaluate(f)
    return report(entity_counts, relation_counts)

def eval_file_by_sample(filename):
    with open(filename) as f:
        entity_counts, relation_counts = evaluate(f)
    return report_by_sample(entity_counts, relation_counts)

if __name__ == '__main__':
    #  sys.exit(main(sys.argv))
    eval_file("../ckpt/default/dev.output")
