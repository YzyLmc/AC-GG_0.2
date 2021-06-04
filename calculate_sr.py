#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:00:02 2021

@author: ziyi
"""


from utils import try_cuda, read_vocab, Tokenizer, vocab_pad_idx

import numpy as np
import sys
sys.path.append('build')

import MatterSim
import math
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.load('tasks/R2R/snapshots/release/follower_final_release', map_location = device)

import json


with open('tasks/R2R/data/R2R_val_seen.json') as f:
  data = json.load(f)
  
def get_end_pose(agent,encoded_instructions, scanId, viewpointId, heading = 0., elevation = 0.):
    pose = agent.end_pose(encoded_instructions, scanId, viewpointId, heading = heading, elevation = elevation)   
    return pose.point

def get_gt_end_pose(scan, viewpoint, heading = 0, elevation = 0):
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.newEpisode(scan, viewpoint, heading, elevation)
    state = sim.getState()
    return state.location.point
    
def dist(location1, location2):
            x1 = location1[0]
            y1 = location1[1]
            z1 = location1[2]
            x2 = location2[0]
            y2 = location2[1]
            z2 = location2[2]
            #return np.sqrt(np.square(x1-x2)+np.square(y1-y2)+np.square(z1-z2))
            return np.sqrt(np.square(x1-x2)+np.square(y1-y2))
        
itr = 0
success = 0

for i in range(len(data)):
    scan = data[i]['scan']
    viewpoint_st = data[i]['path'][0]
    viewpoint_end = data[i]['path'][-1]
    end_pose_gt = get_gt_end_pose(scan, viewpoint_end)
    ins = data[i]['instructions']
    for ins_i in ins:
        encoded_instructions, _ = tok.encode_sentence(ins_i)
        encoded_instructions = torch.tensor(encoded_instructions, device = device)
        end_pose_pred = get_end_pose(agent,encoded_instructions, scan, viewpoint_st)
        distance = dist(end_pose_pred, end_pose_gt)
        
        if distance < 3:
            success += 1
            
        itr += 1
    
sr = success/itr
print('sr = {}/{} = {}'.format(success,itr,sr))
    
        
    






