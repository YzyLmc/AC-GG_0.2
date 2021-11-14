#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:42:09 2021

@author: ziyi
"""


import json
import sys
import numpy as np
import random
from collections import namedtuple
sys.path.append('build')
import MatterSim
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D

from utils import vocab_pad_idx, vocab_bos_idx, vocab_eos_idx, flatten, try_cuda, read_vocab, Tokenizer



def batch_instructions_from_encoded(encoded_instructions, max_length, reverse=False, sort=False):
    # encoded_instructions: list of lists of token indices (should not be padded, or contain BOS or EOS tokens)
    #seq_tensor = np.array(encoded_instructions)
    # make sure pad does not start any sentence
    num_instructions = len(encoded_instructions)
    seq_tensor = np.full((num_instructions, max_length), vocab_pad_idx)
    seq_lengths = []
    for i, inst in enumerate(encoded_instructions):
        
        #if len(inst) > 0:
        #    assert inst[-1] != vocab_eos_idx
        if reverse:
            inst = inst[::-1]
        inst = np.concatenate((inst.cpu(), [vocab_eos_idx]))
        inst = inst[:max_length]
        seq_tensor[i,:len(inst)] = inst
        seq_lengths.append(len(inst))

    seq_tensor = torch.from_numpy(seq_tensor)
    if sort:
        seq_lengths, perm_idx = torch.from_numpy(np.array(seq_lengths)).sort(0, True)
        seq_lengths = list(seq_lengths)
        seq_tensor = seq_tensor[perm_idx]

    mask = (seq_tensor == vocab_pad_idx)[:, :max(seq_lengths)]

    ret_tp = try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
             try_cuda(mask.bool()), \
             seq_lengths
    if sort:
        ret_tp = ret_tp + (list(perm_idx),)
    return ret_tp


class compatModel():
    
    def __init__(self, env, results_path, visEncoder, lanEncoder, dotSimModel, max_episode_len=10, instruction_len = 60):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

        self.visEncoder = visEncoder
        self.lanEncoder = lanEncoder
        self.dotSim = dotSimModel
        self.max_episode_len = max_episode_len
        self.instruction_len = instruction_len
        
        self.losses = []     
        
        
    def write_results(self):
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f)
            
    def _feature_variable(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        features = [ob['feature'] for ob in (flatten(obs) if beamed else obs)]
        assert all(len(f) == 1 for f in features)  #currently only support one image featurizer (without attention)
        features = np.stack(features)
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))
    
    def _batch_observations_and_actions(self, path_obs, path_actions, encoded_instructions):
        seq_lengths = np.array([len(a) for a in path_actions])
        max_path_length = seq_lengths.max()

        # DO NOT permute the sequence, since here we are doing manual LSTM unrolling in encoder
        # perm_indices = np.argsort(-seq_lengths)
        perm_indices = np.arange(len(path_obs))
        #path_obs, path_actions, encoded_instructions, seq_lengths = zip(*sorted(zip(path_obs, path_actions, encoded_instructions, seq_lengths), key=lambda p: p[-1], reverse=True))
        # path_obs = [path_obs[i] for i in perm_indices]
        # path_actions = [path_actions[i] for i in perm_indices]
        # if encoded_instructions:
        #     encoded_instructions = [encoded_instructions[i] for i in perm_indices]
        # seq_lengths = [seq_lengths[i] for i in perm_indices]

        batch_size = len(path_obs)
        assert batch_size == len(path_actions)

        mask = np.ones((batch_size, max_path_length), np.uint8)
        action_embedding_dim = path_obs[0][0]['action_embedding'].shape[-1]
        batched_action_embeddings = [
            np.zeros((batch_size, action_embedding_dim), np.float32)
            for _ in range(max_path_length)]
        feature_list = path_obs[0][0]['feature']
        assert len(feature_list) == 1
        image_feature_shape = feature_list[0].shape
        batched_image_features = [
            np.zeros((batch_size,) + image_feature_shape, np.float32)
            for _ in range(max_path_length)]
        for i, (obs, actions) in enumerate(zip(path_obs, path_actions)):
            # don't include the last state, which should result after the stop action
            assert len(obs) == len(actions) + 1
            obs = obs[:-1]
            mask[i, :len(actions)] = 0
            for t, (ob, a) in enumerate(zip(obs, actions)):
                assert a >= 0
                batched_image_features[t][i] = ob['feature'][0]
                batched_action_embeddings[t][i] = ob['action_embedding'][a]
        batched_action_embeddings = [
            try_cuda(Variable(torch.from_numpy(act), requires_grad=False))
            for act in batched_action_embeddings]
        batched_image_features = [
            try_cuda(Variable(torch.from_numpy(feat), requires_grad=False))
            for feat in batched_image_features]
        mask = try_cuda(torch.from_numpy(mask))

        start_obs = [obs[0] for obs in path_obs]

        return start_obs, \
               batched_image_features, \
               batched_action_embeddings, \
               mask, \
               list(seq_lengths), \
               encoded_instructions, \
               list(perm_indices)
               
    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions):
            assert len(path_obs) == len(path_actions)
            assert len(path_obs) == len(encoded_instructions)
            start_obs, batched_image_features, batched_action_embeddings, path_mask, \
                path_lengths, encoded_instructions, perm_indices = \
                self._batch_observations_and_actions(
                    path_obs, path_actions, encoded_instructions)
    
            instr_seq, instr_mask, instr_lengths = batch_instructions_from_encoded(encoded_instructions, self.instruction_len)
    
            batch_size = len(start_obs)
    
            ctx_vis, h_t_vis, c_t_vis = self.visEncoder(batched_action_embeddings, batched_image_features)
            
            ctx_lan, h_t_lan ,c_t_lan = self.lanEncoder(instr_seq, instr_lengths)
            
            m = torch.ones(batch_size)
            for i in range(batch_size):
                ob = start_obs[i]
                if 'label' in ob:
                    m[i] = 0       
            comp_matrix, loss = self.dotSim(h_t_vis, h_t_lan,m)
            probs = torch.zeros(batch_size)
            for i in range(len(probs)):
                probs[i] = comp_matrix[i,i]
            
            outputs = []
            for i in range(batch_size):
                item = {'label':m[i],
                       'predict':probs[i],
                       'instr_id':start_obs[i]['instr_id'] + '_' + str(int(m[i]))}
                outputs.append(item)
            
            success = 0
            for i in range(batch_size):
                if m[i] == 1:
                    if probs[i] > 0.5:
                        success += 1
                elif m[i] == 0:
                    if probs[i] < 0.5:
                        success += 1
            accuracy = success/batch_size
            
            npy_suc = np.load('compat_suc.npy')
            npy_suc = np.append(npy_suc,accuracy)
            with open('compat_suc.npy', 'wb') as f:
                np.save(f, npy_suc)              
            
            return outputs, loss
    
    def predict(self,path_obs, path_actions, encoded_instructions):
        start_obs, batched_image_features, batched_action_embeddings, path_mask, \
        path_lengths, encoded_instructions, perm_indices = \
        self._batch_observations_and_actions(
            path_obs, path_actions, encoded_instructions)

        batched_action_embeddings = batched_action_embeddings
        batched_image_features = batched_image_features
        
        instr_seq, instr_mask, instr_lengths = batch_instructions_from_encoded(encoded_instructions, self.instruction_len)
        ctx_vis, h_t_vis, c_t_vis = self.visEncoder(batched_action_embeddings, batched_image_features)
        ctx_lan, h_t_lan ,c_t_lan = self.lanEncoder(instr_seq, instr_lengths)
        
        prob = self.dotSim._instance_predict(h_t_vis, h_t_lan)
        
        return prob
        
    def rollout(self, load_next_minibatch=True):
        path_obs, path_actions, encoded_instructions = self.env.gold_obs_actions_and_instructions(self.max_episode_len, load_next_minibatch=load_next_minibatch)
        outputs, loss = self._score_obs_actions_and_instructions(path_obs, path_actions, encoded_instructions)
        
        self.loss = loss
        try:
            self.losses.append(loss.item())
        except:
            self.losses.append(loss)
        return outputs
    
    def train(self, visEncoder_optimizer, lanEncoder_optimizer, dotSim_optimizer, n_iters):
        ''' Train for a given number of iterations '''
        self.visEncoder.train()
        self.lanEncoder.train()
        self.dotSim.train()
        self.losses = []
        it = range(1, n_iters + 1)
        try:
            import tqdm
            it = tqdm.tqdm(it)
        except:
            pass
        for _ in it:
            visEncoder_optimizer.zero_grad()
            lanEncoder_optimizer.zero_grad()
            dotSim_optimizer.zero_grad()
            self.rollout()
            if not self.loss == 0:
                self.loss.backward()
            visEncoder_optimizer.step()
            lanEncoder_optimizer.step()
            dotSim_optimizer.step()
            
    def test(self, use_dropout=False):
        ''' Evaluate once on each instruction in the current environment '''
        if use_dropout:
            self.visEncoder.train()
            self.lanEncoder.train()
            self.dotSim.train()
        else:
            self.visEncoder.eval()
            self.lanEncoder.eval()
            self.dotSim.eval()
        self.env.reset_epoch()
        self.losses = []
        self.results = {}


        # We rely on env showing the entire batch before repeating anything
        looped = False
        # rollout_scores = []
        # beam_10_scores = []
        while True:
            rollout_results = self.rollout()
            # if self.feedback == 'argmax':
            #     path_obs, path_actions, _ = self.env.gold_obs_actions_and_instructions(self.max_episode_len, load_next_minibatch=False)
            #     beam_results = self.beam_search(1, path_obs, path_actions)
            #     assert len(rollout_results) == len(beam_results)
            #     for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #         assert rollout_traj['instr_id'] == beam_trajs[0]['instr_id']
            #         assert rollout_traj['word_indices'] == beam_trajs[0]['word_indices']
            #         assert np.allclose(rollout_traj['score'], beam_trajs[0]['score'])
            #     print("passed check: beam_search with beam_size=1")
            #
            #     self.env.set_beam_size(10)
            #     beam_results = self.beam_search(10, path_obs, path_actions)
            #     assert len(rollout_results) == len(beam_results)
            #     for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #         rollout_score = rollout_traj['score']
            #         rollout_scores.append(rollout_score)
            #         beam_score = beam_trajs[0]['score']
            #         beam_10_scores.append(beam_score)
            #         # assert rollout_score <= beam_score
            # # print("passed check: beam_search with beam_size=10")

            for result in rollout_results:
                if result['instr_id'] in self.results:
                    print(result['instr_id'])
                    looped = True
                else:
                    self.results[result['instr_id']] = result
            if looped:
                break
        # if self.feedback == 'argmax':
        #     print("avg rollout score: ", np.mean(rollout_scores))
        #     print("avg beam 10 score: ", np.mean(beam_10_scores))
        return self.results
    
    def _encoder_and_decoder_paths(self, base_path):
        return base_path + "_vis", base_path + "_lan", base_path + '_dot'

    def save(self, path):
        ''' Snapshot models '''
        visEncoder_path, lanEncoder_path, dotSim_path = self._encoder_and_decoder_paths(path)
        torch.save(self.visEncoder.state_dict(), visEncoder_path)
        torch.save(self.lanEncoder.state_dict(), lanEncoder_path)
        torch.save(self.dotSim.state_dict(), dotSim_path)

    def load(self, path, **kwargs):
        ''' Loads parameters (but not training state) '''
        visEncoder_path, lanEncoder_path, dotSim_path = self._encoder_and_decoder_paths(path)
        self.visEncoder.load_state_dict(torch.load(visEncoder_path, **kwargs))
        self.lanEncoder.load_state_dict(torch.load(lanEncoder_path, **kwargs))
        self.dotSim.load_state_dict(torch.load(dotSim_path, **kwargs))
        print('loaded pretrained model under',path)
            
            
            
        
        
        
        
        
