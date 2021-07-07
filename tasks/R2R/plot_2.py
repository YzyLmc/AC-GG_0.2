#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:10:29 2021

@author: ziyi
"""


import numpy as np

data = np.load('VLN_training_nobackprop.npy')

import matplotlib.pyplot as plt

data_smooth = np.zeros(len(data))
num = 100
for i in range(len(data_smooth)):
    if i > num-1:
        data_smooth[i] = sum(data[i-num:i+1])/(num+1)
    else:
        data_smooth[i] = sum(data[:i+1])/(i+1)
plt.plot(data_smooth[1:])
plt.show()

#%%
npy = np.array(0)
with open('VLN_training_rnobackprop.npy', 'wb') as f:

    np.save(f, npy)
# =============================================================================
# #%%
# data_1 = np.load('BLEU_training.npy')
# 
# 
# 
# plt.plot(data_1)
# plt.show()    
# =============================================================================
# =============================================================================
# #%%
# val_seen = np.array([0.35106334,0.3468508,0.34565064,0.342531])
# val_unseen = np.array([0.33788922,0.33416596,0.33695048])
# 
# plt.plot(val_seen)
# plt.plot(val_unseen)
# plt.show()
# =============================================================================
    
    
    
