#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 00:34:15 2021

@author: ziyi
"""


###for practicing grounding logic only

import torch

import os
import time
import json
import random
import sys
import numpy as np
import math
from collections import defaultdict
from speaker import Speaker
from arbiter import Arbiter
import MatterSim

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch, SemiBatch, ArbiterBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import IPython


from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
IMAGENET_CANDIDATE_FEATURES = 'img_features/ResNet-152-candidate.tsv'

PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'
PLACE365_CANDIDATE_FEATURES = 'img_features/ResNet-152-places365-candidate.tsv'

if args.place365:
    features = PLACE365_FEATURES
    CANDIDATE_FEATURES = PLACE365_CANDIDATE_FEATURES
else:
    features = IMAGENET_FEATURES
    CANDIDATE_FEATURES = IMAGENET_CANDIDATE_FEATURES
    
feature_dict = read_img_features(features)
candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)
glove_path = 'tasks/R2R/data/train_glove.npy'
glove = np.load(glove_path)
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)


listner = Seq2SeqAgent(None, "", tok, feat=feature_dict, candidates = candidate_dict, episode_len = args.maxAction)
listner.load('snap/agent/state_dict/best_val_unseen')

if __name__ == '__main__':
    #change following lines to ground your own instr
    #######################################################################################
    scanId = 'vyrNrziPKCB'
    viewpointId = 'c8a5472a5ef243319ffa4f88d3ddb4bd'
    encoded_instructions = listner.tok.encode_sentence('Exit the room using the door on the left. Turn slightly left and go past the round table an chairs. Wait there. ')
    #######################################################################################
    
    encoded_instructions = torch.tensor(encoded_instructions, device = device)
    sim = MatterSim.Simulator()         #init mattersim
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
   
    angle_inc = np.pi / 6.
    traj = listner.ground(sim,encoded_instructions, scanId, viewpointId)
    
    print(traj)
