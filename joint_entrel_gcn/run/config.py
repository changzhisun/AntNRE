#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/09/21 15:42:09

@author: Changzhi Sun
"""
import os
import sys
import argparse

from configparser import SafeConfigParser

sys.path.append('..')

class Configurable:

    def __init__(self, config_file, extra_args):

        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = { k[2:] : v for k, v in zip(extra_args[0::2], extra_args[1::2]) }
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
                    print(section, k, v)
        self._config = config
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
            os.mkdir(os.path.join(self.save_dir, "minibatch"))
        config.write(open(self.config_file, "w", encoding="utf8"))
        print("Loaded config file successful.")
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    @property
    def pretrained_embeddings_file(self):
        return self._config.get('Data','pretrained_embeddings_file')

    @property
    def data_dir(self):
        return self._config.get('Data','data_dir')

    @property
    def entity_schema(self):
        return self._config.get('Data','entity_schema')

    @property
    def max_sent_len(self):
        return self._config.getint('Data','max_sent_len')

    @property
    def train_file(self):
        return self._config.get('Data','train_file')

    @property
    def dev_file(self):
        return self._config.get('Data','dev_file')

    @property
    def test_file(self):
        return self._config.get('Data','test_file')

    @property
    def save_dir(self):
        return self._config.get('Save','save_dir')

    @property
    def config_file(self):
        return self._config.get('Save','config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save','save_model_path')

    @property
    def load_dir(self):
        return self._config.get('Save','load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def lstm_layers(self):
        return self._config.getint('Network','lstm_layers')

    @property
    def gcn_layers(self):
        return self._config.getint('Network','gcn_layers')

    @property
    def gcn_beta(self):
        return self._config.getfloat('Network','gcn_beta')

    @property
    def word_dims(self):
        return self._config.getint('Network','word_dims')

    @property
    def char_dims(self):
        return self._config.getint('Network','char_dims')

    @property
    def lstm_hiddens(self):
        return self._config.getint('Network','lstm_hiddens')

    @property
    def dropout(self):
        return self._config.getfloat('Network','dropout')

    @property
    def char_kernel_sizes(self):
        return eval(self._config.get('Network','char_kernel_sizes'))

    @property
    def char_output_channels(self):
        return self._config.getint('Network','char_output_channels')

    @property
    def rel_kernel_sizes(self):
        return eval(self._config.get('Network','rel_kernel_sizes'))

    @property
    def rel_output_channels(self):
        return self._config.getint('Network','rel_output_channels')

    @property
    def use_cuda(self):
        return self._config.getboolean('Network','use_cuda')

    @property
    def schedule_k(self):
        return self._config.getfloat('Network','schedule_k')

    @property
    def clip_c(self):
        return self._config.getfloat('Optimizer','clip_c')

    @property
    def train_iters(self):
        return self._config.getint('Run','train_iters')

    @property
    def batch_size(self):
        return self._config.getint('Run','batch_size')

    @property
    def patience(self):
        return self._config.getint('Run','patience')

    @property
    def validate_every(self):
        return self._config.getint('Run','validate_every')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
