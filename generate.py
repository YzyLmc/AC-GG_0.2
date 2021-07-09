#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 01:11:41 2021

@author: ziyi
"""


import utils
from utils import try_cuda, read_vocab, Tokenizer
import train_speaker
import env
import numpy as np
import json
import sys
sys.path.append('build')
import MatterSim
import math
import torch
from speaker import Seq2SeqSpeaker
from model import SpeakerEncoderLSTM, SpeakerDecoderLSTM

from vocab import SUBTRAIN_VOCAB, TRAIN_VOCAB, TRAINVAL_VOCAB
import argparse
parser = argparse.ArgumentParser()
from env import ImageFeatures
ImageFeatures.add_args(parser)

args, _ = parser.parse_known_args()

image_features_list= ImageFeatures.from_args(args)
angle_inc = np.pi / 6.
def build_viewpoint_loc_embedding(viewIndex):
    """
    Position embedding:
    heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    embedding = np.zeros((36, 128), np.float32)
    for absViewIndex in range(36):
        relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
        rel_heading = (relViewIndex % 12) * angle_inc
        rel_elevation = (relViewIndex // 12 - 1) * angle_inc
        embedding[absViewIndex,  0:32] = np.sin(rel_heading)
        embedding[absViewIndex, 32:64] = np.cos(rel_heading)
        embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
        embedding[absViewIndex,   96:] = np.cos(rel_elevation)
    return embedding


# pre-compute all the 36 possible paranoram location embeddings
_static_loc_embeddings = [
    build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]

class rdv():
    '''
    traj = {'scan':'scanId',
    'path':['viewpointId_0', 'viewpointId_1',...],
    'heading_init': 0.0,
    'elevaion_init':0.0}
    '''
    def __init__(self, traj, elevation = 0):

        self.scanId = traj['scan']
        self.path = traj['path']
        viewPointInit = self.path[0]
        self.heading = traj['heading']
        self.elevation = elevation
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(640, 480)
        self.sim.setCameraVFOV(math.radians(60))
        self.sim.newEpisode(self.scanId, viewPointInit, self.heading, self.elevation)
        
        
    def _get_adj_loc_ls(self):  
        _, adj_loc_ls = env._get_panorama_states(self.sim)
        return adj_loc_ls

                
    def _buildOb(self):
        state, adj_loc_ls = env._get_panorama_states(self.sim)
        
        assert self.scanId== state.scanId
        #filePath = 'img_features_36*2048/'+ self.scanId + '/' + state.location.viewpointId + '.pt'
        #feature = torch.load(filePath)
        feature = [f.get_features(state) for f in image_features_list]
        
        #print(feature,_static_loc_embeddings[state.viewIndex])
        #print(feature.size(),_static_loc_embeddings[state.viewIndex].size())
        feature_with_loc = np.concatenate((feature[0], _static_loc_embeddings[state.viewIndex]), axis=-1)
        action_embedding = env._build_action_embedding(adj_loc_ls, feature[0])
        ob = {
            'scan' : state.scanId,
            'viewpoint' : state.location.viewpointId,
            'viewIndex' : state.viewIndex,
            'heading' : state.heading,
            'elevation' : state.elevation,
            'feature' : [feature_with_loc],
            'step' : state.step,
            'adj_loc_list' : adj_loc_ls,
            'action_embedding': action_embedding,
            'navigableLocations' : state.navigableLocations,
        }
        
        return ob
    
    def _act_and_view(self, adj_loc_ls, nextViewpointId):
        for i in range(len(adj_loc_ls)):
            if adj_loc_ls[i]['nextViewpointId'] == nextViewpointId:
                return i, adj_loc_ls[i]['absViewIndex']
            
    def obs_and_acts(self):
        obs = []
        acts = []
        ob_init = self._buildOb()
        obs.append(ob_init)
        #curViewpointId = self.sim.getState().location.viewpointId
        for i in range(1,len(self.path)):
            
            nextViewpointId = self.path[i]
            adj_loc_ls_i = self._get_adj_loc_ls()

            act, viewIndex = self._act_and_view(adj_loc_ls_i,nextViewpointId)
            env._navigate_to_location(self.sim, nextViewpointId, viewIndex)
            ob_new = self._buildOb()
            obs.append(ob_new)
            acts.append(act)
            
        return [obs], [acts]
            
            
    

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
learning_rate = 0.0001
weight_decay = 0.0005
feature_size = 2048+128
glove = np.load(glove_path)

vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)

encoder = try_cuda(SpeakerEncoderLSTM(
        action_embedding_size, feature_size, hidden_size, dropout_ratio
        ))

decoder = try_cuda(SpeakerDecoderLSTM(
    len(vocab), word_embedding_size, hidden_size, dropout_ratio,
    glove=glove))
agent = Seq2SeqSpeaker(
    tok, "", encoder, decoder, MAX_INSTRUCTION_LENGTH)
agent.load('tasks/R2R/snapshots/release/speaker_final_release', map_location = 'cpu')
if __name__ == "__main__":
    traj = {'scan':'5q7pvUzZiYa', 'path':["7dc12a67ddfc4a4a849ce620db5b777b", "0e84cf4dec784bc28b78a80bee35c550", "a77784b955454209857d745976a1676d", "67971a17c26f4e2ca117b4fca73507fe", "8db06d3a0dd44508b3c078d60126ce19", "43ac37dfa1db4a13a8a9df4e454eb016", "4bd82c990a6548a994daa97c8f52db06", "6d11ca4d41e04bb1a725c2223c36b2aa", "29fb3c58b29348558d36a9f9440a1379", "c23f26401359426982d11ca494ee739b", "397403366d784caf804d741f32fd68b9", "3c6a35e15ada4b649990d6568cce8bd9", "55e4436f528c4bf09e4550079c572f7b", "69fad7dd177847dbabf69e8fb7c00ddf", "c629c7f1cf6f47a78c45a8ae9ff82247", "21fca0d6192940e580587fe317440f56", "4b85d61dd3a94e8a812affe78f3a322d", "3c025b8e3d2040969cd00dd0e9f29b09"], 'heading':0.0,'elevation_init':0.0}
    
    rdv_test = rdv(traj)
    
    path_obs, path_actions = rdv_test.obs_and_acts()   
        # predicted
    decoded_words = agent.speak(path_obs,path_actions,vocab)
    
    print(' '.join(decoded_words))
# =============================================================================
#     
#     with open('tasks/R2R/data/R2R_val_unseen.json') as f:
#         data = json.load(f)
#         
#     traj = data[8]
#     
#     rdv_test = rdv(traj)
#     
#     path_obs, path_actions = rdv_test.obs_and_acts()   
#          # predicted
#     decoded_words = agent.generate(path_obs,path_actions,vocab)  
#     
#     instr_generated = [' '.join(decoded_words)]        
#     instr_refs = [traj['instructions']]
#     
#     from bert_score import BERTScorer
#     scorer = BERTScorer(lang='en', rescale_with_baseline = True)
#     
#     P, R ,F1 = scorer.score(instr_generated, instr_refs)
#     print(F1)
# =============================================================================
    
    
        
    




