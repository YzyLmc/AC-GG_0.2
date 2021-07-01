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

#init m3d
sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setCameraResolution(640, 480)
sim.setCameraVFOV(math.radians(60))

#params
step = 5

def roll_out(scan, viewpoint, sim=sim, heading = 0, elevation = 0):
    sim.newEpisode(sim, scan, viewpoint, heading, elevation)
    ended = False
    
    for i in range(step):
        _, adj_loc_ls = env._get_panorama_states(sim)
        traj = []
        traj.append(viewpoint)
        if len(adj_loc_ls) == 1:
            ended == True
        elif i == 0:
            next_loc = random.choice(adj_loc_ls)
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
            
        if ended : break
    
    return traj

if __name__ == '__main__':
    scan = '5q7pvUzZiYa'
    viewpoint = "7dc12a67ddfc4a4a849ce620db5b777b"
    traj = roll_out(scan,viewpoint)
    print(traj)
            
        
            
            
            
            
            
            
            