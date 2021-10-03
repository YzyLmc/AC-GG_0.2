#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 01:30:42 2021

@author: ziyi
"""


import torch
from torch import optim
import json

import random
import os
import os.path
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utils
from utils import read_vocab, Tokenizer, timeSince, try_cuda, vocab_pad_idx
from env import R2RBatch, ImageFeatures
import eval_speaker

from vocab import SUBTRAIN_VOCAB, TRAIN_VOCAB, TRAINVAL_VOCAB
from model import EncoderLSTM, SpeakerEncoderLSTM, dotSimilarity
from compatModel import compatModel

RESULT_DIR = 'tasks/R2R/compat/results/'
SNAPSHOT_DIR = 'tasks/R2R/compat/snapshots/'
PLOT_DIR = 'tasks/R2R/compat/plots/'

MAX_INSTRUCTION_LENGTH = 80

batch_size = 100
max_episode_len = 10
word_embedding_size = 300
glove_path = 'tasks/R2R/data/train_glove.npy'
action_embedding_size = 2048+128
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'sample'  # teacher or sample
learning_rate = 0.00001 #original learning rate 0.0001
#learning_rate = 0.005 #Bertscore LR
weight_decay = 0.0005
FEATURE_SIZE = 2048+128
n_iters = 2000
log_every = 100
save_every = 100

vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)

# load hard negatives
with open('tasks/R2R/hardNeg_train.json','r') as f:
    hardNeg_train = json.load(f)
for item in hardNeg_train:
    instr = item['instructions']
    item['instr_encoding'], item['instr_length'] = tok.encode_sentence(instr)
  
with open('tasks/R2R/hardNeg_val_seen.json','r') as f:
    hardNeg_val_seen = json.load(f)
with open('tasks/R2R/hardNeg_val_unseen.json','r') as f:
    hardNeg_val_unseen = json.load(f)
for item in hardNeg_val_seen:
    instr = item['instructions']
    item['instr_encoding'], item['instr_length'] = tok.encode_sentence(instr)
for item in hardNeg_val_unseen:
    instr = item['instructions']
    item['instr_encoding'], item['instr_length'] = tok.encode_sentence(instr)

def get_model_prefix(args, image_feature_list):
    image_feature_name = "+".join(
        [featurizer.get_name() for featurizer in image_feature_list])
    model_prefix = 'compat_{}_{}'.format(
        feedback_method, image_feature_name)
    if args.use_train_subset:
        model_prefix = 'trainsub_' + model_prefix
    return model_prefix

def eval_model(agent, results_path, use_dropout):
    agent.results_path = results_path
    agent.test(
        use_dropout=use_dropout)
    
def filter_param(param_list):
    return [p for p in param_list if p.requires_grad]

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
def make_env_and_models(args, train_vocab_path, train_splits, test_splits,
                        test_instruction_limit=None):
    setup()
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    train_env = R2RBatch(image_features_list, batch_size=batch_size,
                         splits=train_splits, tokenizer=tok)
    
    train_env.data.extend(hardNeg_train) # extend train data and shuffle
    random.shuffle(train_env.data)
    
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    glove = np.load(glove_path)
    feature_size = FEATURE_SIZE
    
    visEncoder = try_cuda(SpeakerEncoderLSTM(
        action_embedding_size, feature_size, enc_hidden_size, dropout_ratio,
        bidirectional=bidirectional))
    
    lanEncoder = try_cuda(EncoderLSTM(
        len(vocab), word_embedding_size, enc_hidden_size, vocab_pad_idx,
        dropout_ratio, bidirectional=args.bidirectional, glove=glove))
    
    dotSim = try_cuda(dotSimilarity(batch_size, enc_hidden_size))
    
    test_envs = {
    split: (R2RBatch(image_features_list, batch_size=batch_size,
                     splits=[split], tokenizer=tok,
                     instruction_limit=test_instruction_limit),
            eval_speaker.SpeakerEvaluation(
                [split], instructions_per_path=test_instruction_limit))
    for split in test_splits}
    
    test_envs['val_seen'][0].data.extend(hardNeg_val_seen)
    test_envs['val_unseen'][0].data.extend(hardNeg_val_unseen)

    return train_env, test_envs, visEncoder, lanEncoder, dotSim

def train_setup(args):
    train_splits = ['train_aug']
    # val_splits = ['train_subset', 'val_seen', 'val_unseen']
    val_splits = ['val_seen', 'val_unseen']
    vocab = TRAIN_VOCAB

    if args.use_train_subset:
        train_splits = ['sub_' + split for split in train_splits]
        val_splits = ['sub_' + split for split in val_splits]
        vocab = SUBTRAIN_VOCAB

    train_env, val_envs, visEncoder, lanEncoder, dotSim= make_env_and_models(
        args, vocab, train_splits, val_splits)
    agent = compatModel(
        train_env, "", visEncoder, lanEncoder, dotSim)
    
    return agent, train_env, val_envs

def train_val(args):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    agent, train_env, val_envs = train_setup(args)
    train(args, train_env, agent, val_envs=val_envs)
    

def train(args, train_env, agent, log_every=log_every, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''

    if val_envs is None:
        val_envs = {}

    print('Training with %s feedback' % feedback_method)
    visEncoder_optimizer = optim.Adam(
        filter_param(agent.visEncoder.parameters()), lr=learning_rate,
        weight_decay=weight_decay)
    lanEncoder_optimizer = optim.Adam(
        filter_param(agent.lanEncoder.parameters()), lr=learning_rate,
        weight_decay=weight_decay)
    dotSim_optimizer = optim.Adam(
        filter_param(agent.dotSim.parameters()), lr=learning_rate,
        weight_decay=weight_decay)

    data_log = defaultdict(list)
    start = time.time()

    split_string = "-".join(train_env.splits)

    def make_path(n_iter):
        return os.path.join(
            args.snapshot_dir, '%s_%s_iter_%d' % (
                get_model_prefix(args, train_env.image_features_list),
                split_string, n_iter))

    best_metrics = {}
    last_model_saved = {}
    for idx in range(0, args.n_iters, log_every):
        agent.env = train_env

        interval = min(log_every, args.n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(visEncoder_optimizer, lanEncoder_optimizer, dotSim_optimizer, interval
                    )
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        save_log = []
        
        print(('%s (%d %d%%) %s' % (
             timeSince(start, float(iter)/args.n_iters),
             iter, float(iter)/args.n_iters*100, loss_str)))
        if not args.no_save:
            if save_every and iter % save_every == 0:
                agent.save(make_path(iter))
                
def make_arg_parser():
    parser = argparse.ArgumentParser()
    ImageFeatures.add_args(parser)
    parser.add_argument(
        "--use_train_subset", action='store_true',
        help="use a subset of the original train data for validation")
    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--n_iters", type=int, default=2000)
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument("--result_dir", default=RESULT_DIR)
    parser.add_argument("--snapshot_dir", default=SNAPSHOT_DIR)
    parser.add_argument("--plot_dir", default=PLOT_DIR)
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), train_val)
    
    
    
    
    