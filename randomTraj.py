#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 22:31:46 2021

@author: ziyi
"""

import env
import MatterSim
import math
import random
import json
import numpy as np
import networkx as nx

#init m3d
sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setCameraResolution(640, 480)
sim.setCameraVFOV(math.radians(60))

#params
step = 5

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

def roll_out(scan, viewpoint, sim=sim, heading = 0, elevation = 0):
    sim.newEpisode(scan, viewpoint, heading, elevation)
    ended = False
    traj = []
    traj.append(viewpoint)
    for i in range(step):
        _, adj_loc_ls = env._get_panorama_states(sim)

        if len(adj_loc_ls) == 1:
            ended == True
        elif i == 0:
            next_loc = random.choice(adj_loc_ls[1:])
            nextViewpointId = next_loc['nextViewpointId']
            nextViewIndex = next_loc['absViewIndex']
            env._navigate_to_location(sim, nextViewpointId,nextViewIndex)
            traj.append(nextViewpointId)
        else:
            next_loc = adj_loc_ls[1]
            nextViewpointId = next_loc['nextViewpointId']
            nextViewIndex = next_loc['absViewIndex']
            env._navigate_to_location(sim, nextViewpointId,nextViewIndex)
            traj.append(nextViewpointId)
            
        if ended :
            print(adj_loc_ls)
            break
    
    return traj

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
#     scan = '5q7pvUzZiYa'
#     viewpoint = "7dc12a67ddfc4a4a849ce620db5b777b"
#     traj = roll_out(scan,viewpoint)
#     print(traj)
# =============================================================================
    
    with open('tasks/R2R/data/R2R_val_seen.json') as f:
        data = json.load(f)
        
    scans = []
    for traj in data:
        if traj['scan'] not in scans:
            scans.append(traj['scan'])
            
    distances = _load_nav_graphs(scans)
        
    itr = 0
    success = 0
    distance_all = 0    
    for i in range(len(data)):
        scan = data[i]['scan']
        viewpoint_st = data[i]['path'][0]
        #gt end pose
        viewpoint_end = data[i]['path'][-1]
        

        #predicted end pose
        traj = roll_out(scan,viewpoint_st)
        traj_end = traj[-1]
       
        
        distance = distances[scan][viewpoint_end][traj_end]
        
        if distance < 3:
            success += 1
        distance_all += distance
            
        itr += 1
    
    sr = success/itr
    avg_dis = distance_all/itr
    print('sr = {}/{} = {}, avg_dist={}'.format(success,itr,sr,avg_dis))
            
        
            
            
            
            
            
            
            