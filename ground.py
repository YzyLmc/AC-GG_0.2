#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:43:36 2021

@author: ziyi
"""



from utils import try_cuda, read_vocab, Tokenizer, vocab_pad_idx


import numpy as np
import sys
sys.path.append('build')

import torch
from follower import Seq2SeqAgent
from model import EncoderLSTM, AttnDecoderLSTM
from vocab import SUBTRAIN_VOCAB, TRAIN_VOCAB, TRAINVAL_VOCAB

MAX_INPUT_LENGTH = 80
feature_size = 2048+128
max_episode_len = 10
word_embedding_size = 300
glove_path = 'tasks/R2R/data/train_glove.npy'
action_embedding_size = 2048+128
hidden_size = 512
dropout_ratio = 0.5
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)
glove = np.load(glove_path)

encoder = try_cuda(EncoderLSTM(
        len(vocab), word_embedding_size, hidden_size, vocab_pad_idx,
        dropout_ratio, glove=glove))
decoder = try_cuda(AttnDecoderLSTM(
    action_embedding_size, hidden_size, dropout_ratio,
    feature_size=feature_size))

agent = Seq2SeqAgent(
        None, "", encoder, decoder, max_episode_len,
        max_instruction_length=MAX_INPUT_LENGTH)

agent.load('tasks/R2R/snapshots/release/follower_final_release', map_location = 'cuda')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_end_pose(encoded_instructions, scanId, viewpointId):
    pose = agent.end_pose(encoded_instructions, scanId, viewpointId)   
    return pose

if __name__ == '__main__':
    #change following lines to ground your own instr
    #######################################################################################
    scanId = '5q7pvUzZiYa'
    viewpointId = '55e4436f528c4bf09e4550079c572f7b'
    encoded_instructions, _ = tok.encode_sentence('walk past the kitchen and the stove . go through the kitchen and turn left . walk past the kitchen and turn left . walk into the kitchen and wait by the sink .')
    #######################################################################################
    
    encoded_instructions = torch.tensor(encoded_instructions, device = device)
    traj = agent.generate(encoded_instructions, scanId, viewpointId)
    
    print(traj)



