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
import networkx as nx
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
 
#load features and feature_candidates
feature_dict = read_img_features(features)
candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)
#load glove and vocab
glove_path = 'tasks/R2R/data/train_glove.npy'
glove = np.load(glove_path)
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)

#intantialize listener and load pre-trained model
listner = Seq2SeqAgent(None, "", tok, feat=feature_dict, candidates = candidate_dict, episode_len = args.maxAction)
listner.load('snap/long/ablation_cand_0208_accuGrad_envdrop_ty/state_dict/best_val_unseen')

# nav graph loader from env.py
def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def _load_nav_graphs(scans):
    ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
    print('Loading navigation graphs for %d scans' % len(scans))
    graphs = load_nav_graphs(scans)
    paths = {}
    for scan,G in graphs.items(): # compute all shortest paths
        paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    distances = {}
    for scan,G in graphs.items(): # compute all shortest paths
        distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        
    return distances

if __name__ == '__main__':
# =============================================================================
#     #change following lines to ground your own instr
#     #######################################################################################
#     scanId = 'vyrNrziPKCB'
#     viewpointId = 'c8a5472a5ef243319ffa4f88d3ddb4bd'
#     encoded_instructions = listner.tok.encode_sentence('Exit the room using the door on the left. Turn slightly left and go past the round table an chairs. Wait there. ')
#     #######################################################################################
#     
#     encoded_instructions = torch.tensor(encoded_instructions, device = device)
# =============================================================================
    sim = MatterSim.Simulator()         #init mattersim
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
   
    angle_inc = np.pi / 6.
# =============================================================================
#     traj = listner.ground(sim,encoded_instructions, scanId, viewpointId)
#     
#     print(traj)
# =============================================================================
    
    with open('tasks/R2R/data/R2R_val_unseen.json') as f:
        data = json.load(f)
        
    scans = []
    for traj in data:
        if traj['scan'] not in scans:
            scans.append(traj['scan'])
            
    distances = _load_nav_graphs(scans)
        
    itr = 0
    success = 0
    
    for i in range(len(data)):
        scan = data[i]['scan']
        viewpoint_st = data[i]['path'][0]
        viewpoint_end = data[i]['path'][-1]
        heading = data[i]['heading']        
        ins = data[i]['instructions']        
        for ins_i in ins:
            sim.newEpisode(scan, viewpoint_st, heading, 0.0)    
            encoded_instructions = listner.tok.encode_sentence(ins_i)
            encoded_instructions = torch.tensor(encoded_instructions, device = device)
            traj = listner.ground(sim, encoded_instructions, scan, viewpoint_st,heading=heading)
            end_pose_pred = traj[0]['traj'][-1]
            #end_pose_pred = get_end_pose(agent,encoded_instructions, scan, viewpoint_st)
            #distance = dist(end_pose_pred, end_pose_gt)
            distance = distances[scan][viewpoint_end][end_pose_pred]
            
            if distance < 3:
                success += 1
                
            itr += 1
        
    sr = success/itr
    print('sr = {}/{} = {}'.format(success,itr,sr))
    
    