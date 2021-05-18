#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 19:23:13 2021

@author: ziyi
"""


import sys
sys.path.append('/build')
import MatterSim
import math

sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setCameraResolution(640,480)
sim.setCameraVFOV(math.radians(60))
sim.init()

sim.newEpisode('17DRP5sb8fy','0e92a69a50414253a23043758f111cec', 0, 0)

state = sim.getState()

