#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:43:36 2021

@author: ziyi
"""



from utils import try_cuda, read_vocab, Tokenizer, vocab_pad_idx

import numpy as np
import json
import networkx as nx
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
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.load('tasks/R2R/snapshots/release/follower_final_release', map_location = device)




def get_end_pose(agent,encoded_instructions, scanId, viewpointId, heading = 0., elevation = 0.):
    pose = agent.end_pose(encoded_instructions, scanId, viewpointId, heading = heading, elevation = elevation)   
    return pose.point

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
#     encoded_instructions, _ = tok.encode_sentence('Exit the room using the door on the left. Turn slightly left and go past the round table an chairs. Wait there. ')
#     #######################################################################################
#     
#     encoded_instructions = torch.tensor(encoded_instructions, device = device)
#     traj = agent.generate(encoded_instructions, scanId, viewpointId)
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
        #end_pose_gt = get_gt_end_pose(scan, viewpoint_end)
        ins = data[i]['instructions']
        for ins_i in ins:
            encoded_instructions, _ = tok.encode_sentence(ins_i)
            encoded_instructions = torch.tensor(encoded_instructions, device = device)
            traj = agent.generate(encoded_instructions, scan, viewpoint_st)
            end_pose_pred = traj['trajectory'][-1][0]
            #end_pose_pred = get_end_pose(agent,encoded_instructions, scan, viewpoint_st)
            #distance = dist(end_pose_pred, end_pose_gt)
            distance = distances[scan][viewpoint_end][end_pose_pred]
            
            if distance < 3:
                success += 1
                
            itr += 1
        
    sr = success/itr
    print('sr = {}/{} = {}'.format(success,itr,sr))


