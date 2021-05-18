#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 01:11:41 2021

@author: ziyi
"""


import utils
import train_speaker
import env
import numpy as np
import sys
sys.path.append('build')
import MatterSim
import math


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
    def __init__(self, traj):

        self.scanId = traj['scan']
        self.traj = traj['path']
        viewPointInit = self.traj[0]
        self.heading = traj['heading_init']
        self.elevation = traj['elevation_init']
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(640, 480)
        self.sim.setCameraVFOV(math.radians(60))
        self.sim.newEpisode(self.scanId, viewPointInit, self.heading, self.elevation)
        
        
    def _get_adj_loc_ls(self):  
        _, adj_loc_ls = env._get_panorama_states(self.sim)
        return adj_loc_ls
    
    def _forward_one(self, nextViewIndex):
        adj_loc_ls = self._get_adj_loc_ls()
        for i in len(adj_loc_ls):
            if nextViewpointId == self.traj[1]:
                heading = adj_loc_ls[i]['rel_heading']
                elelvation = adj_loc_ls[i]['rel_elevation']
                nextViewpointId = adj_loc_ls['nextViewpointId']
                
    def buildOb(self):
        state, adj_loc_ls = env._get_panorama_states(self.sim)
        
        assert self.scanId== state.scanId
        filePath = 'img_features_36*2048/'+ self.scanId + '/' + state.location.viewpointId + '.pt'
        feature_pano = torch.load(filePath)
        feature = feature_pano[state.viewIndex]
        
        assert len(feature) == 1, 'for now, only work with MeanPooled feature'
        feature_with_loc = np.concatenate((feature, _static_loc_embeddings[state.viewIndex]), axis=-1)
        action_embedding = env._build_action_embedding(adj_loc_list, feature_pano)
        ob = {
            'instr_id' : item['instr_id'],
            'scan' : state.scanId,
            'viewpoint' : state.location.viewpointId,
            'viewIndex' : state.viewIndex,
            'heading' : state.heading,
            'elevation' : state.elevation,
            'feature' : [feature_with_loc],
            'step' : state.step,
            'adj_loc_list' : adj_loc_list,
            'action_embedding': action_embedding,
            'navigableLocations' : state.navigableLocations,
        }
        
        return ob



        
                
    
    
    

def generate_use_pretrained(args):
    agent, train_env, val_envs = train_speaker.train_setup(args)
    agent.load(args.model_prefix)
    

    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        agent.env = val_env
        path_obs, path_actions, encoded_instructions = agent.env.gold_obs_actions_and_instructions(agent.max_episode_len)
        # # gold
        # gold_results = agent.test(
        #     use_dropout=False, feedback='teacher', allow_cheat=True)
        # gold_score_summary = evaluator.score_results(
        #     gold_results, verbose=False)
        #
        # for metric,val in gold_score_summary.items():
        #     print("gold {} {}\t{}".format(env_name, metric, val))
        #
        # if args.gold_results_output_file:
        #     fname = "{}_{}.json".format(
        #         args.gold_results_output_file, env_name)
        #     with open(fname, 'w') as f:
        #         utils.pretty_json_dump(gold_results, f)

        # predicted
        decoded_words = agent.generate(path_obs[:1],path_actions[:1])

    print(' '.join(decoded_words))

def make_arg_parser():
    parser = train_speaker.make_arg_parser()
    parser.add_argument("model_prefix")
    parser.add_argument("--gold_results_output_file")
    parser.add_argument("--pred_results_output_file")
    # parser.add_argument("--beam_size", type=int, default=1)
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), generate_use_pretrained)
