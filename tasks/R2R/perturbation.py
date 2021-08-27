#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:34:50 2021

@author: ziyi
"""
import json
import re
import random
import numpy as np
from copy import deepcopy
from utils import Tokenizer

with open('R2R_train_aug.json') as f:
    data = json.load(f)

pairs_idx = []
phrase_ls = []
for i in range(len(data)):
    length = len(data[i]['instructions'])
    for j in range(length):
        
        t_i = (i,j)
        pairs_idx.append(t_i)
        ins_k = data[i]['instructions'][j]
        if len(re.split('\.',ins_k)) > 1:
            phrase_ls.append(t_i)

# preselect phrase swap
phrase = []                 
perturb_num = int(len(pairs_idx)/6)
while len(phrase) < perturb_num:
    i = np.random.randint(len(pairs_idx))
    t_i = pairs_idx[i]
    ins_i = data[t_i[0]]['instructions'][t_i[1]]
    if len(re.split('\.', ins_i)) > 1:
        phrase.append(pairs_idx.pop(i))

# preselect direction swap    
d_1 = ['around','left','right']
d_2 = ['bottom','middle','top']
d_3 = ['up','down']
d_4 = ['front','back']
d_5 = ['above','under']
d_6 = ['enter','exit']
d_7 = ['backward','forward']
d_8 = ['away from', 'towards']
d_9 = ['into','out of']
d_10 = ['inside','outside']
#d_ls = [d_1,d_2,d_3,d_4,d_5,d_6,d_7,d_8,d_9,d_10]
d_ls = d_1+d_2+d_3+d_4+d_5+d_6+d_7+d_8+d_9+d_10

direct = []                 
while len(direct) < perturb_num:
    i = np.random.randint(len(pairs_idx))
    t_i = pairs_idx[i]
    ins_i = data[t_i[0]]['instructions'][t_i[1]].lower()
    words = Tokenizer.split_sentence(ins_i)
    if any(word in words for word in d_ls):   
        direct.append(pairs_idx.pop(i))
        
# preselect viewpoint swap  
upper = 0.6
lower = 0.3

import networkx as nx
from ndtw import DTW,load_nav_graphs
     # Load connectiviy graph
scans = []
for traj in data:
    if traj['scan'] not in scans:
        scans.append(traj['scan'])
graphs = load_nav_graphs(scans)
DTWs = {}
for scan in scans:
    graph_i = graphs[scan]
    DTWs[scan] = DTW(graph_i) 
     
viewpoint = []
traj_viewpoint = []
while len(viewpoint) < 2*perturb_num:
    i = np.random.randint(len(pairs_idx))
    t_i = pairs_idx[i]
    data_i = data[t_i[0]]
    path_i = data_i['path']
    scan_i = data_i['scan']
    graph_i = graphs[scan_i]
    start = path_i[0]
    end = path_i[-1]
    
    all_path = nx.all_simple_paths(graph_i, source=start, target=end,cutoff=length + 5)
    for path_simple in all_path:
        dtw_score = DTWs[scan_i](path_simple, path_i)
        if dtw_score < upper and dtw_score > lower:
            viewpoint.append(pairs_idx.pop(i))
            traj_viewpoint.append(path_simple)
            break
    
    if len(viewpoint)//10 == 0:
        print(len(viewpoint))

with open('traj_viewpoint.json','w') as f:
    json.dump(traj_viewpoint,f)
        

#%%
random.shuffle(pairs_idx)
rest_ls = [pairs_idx[perturb_num*i:perturb_num*(i+1)] for i in range(3)]

partitions = {}
partitions['entity'] = rest_ls[0]
partitions['random'] = rest_ls[1]
partitions['reversal'] = rest_ls[2]
partitions['viewpoint'] = viewpoint
partitions['phrase'] = phrase
partitions['direction'] = direct
#%%
with open('parted_idx.json','w') as f:
    json.dump(partitions,f)
#%%
with open('parted_idx.json','r') as f:
    partitions = json.load(f)
#%%
hardNeg = []
gt = {}

for t_i in partitions['reversal']:
    data_i = data[t_i[0]]
    path_i = deepcopy(data_i['path'])
    path_i.reverse()
    new_item = dict(data_i)
    new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
    new_item['instructions'] = data_i['instructions'][t_i[1]]
    new_item['path'] = path_i
    new_item['label'] = 0
    hardNeg.append(new_item)
    path_id = str(data_i['path_id']) + '_0'
    gt[path_id] = data_i
    
#%%


# =============================================================================
# for t_i in partitions['direction']:
#     data_i = data[t_i[0]]
#     ins_i = deepcopy(data_i['instructions'][t_i[1]].lower())
#     #words =Tokenizer.split_sentence(ins_i)
#     
#     if any(word in ins_i for word in d_ls):
#         continue
#     else: print(ins_i)
#     
# #%% 
# upper = 0.6
# lower = 0.3
# 
# import networkx as nx
# from ndtw import DTW,load_nav_graphs
#      # Load connectiviy graph
# scans = []
# for traj in data:
#     if traj['scan'] not in scans:
#         scans.append(traj['scan'])
# graphs = load_nav_graphs(scans)
# DTWs = {}
# for scan in scans:
#     graph_i = graphs[scan]
#     DTWs[scan] = DTW(graph_i) 
#     
# for t_i in partitions['viewpoint']:
#     data_i = data[t_i[0]]
#     path_i = data_i['path']
#     scan_i = data_i['scan']
#     graph_i = graphs[scan_i]
#     start = path_i[0]
#     end = path_i[-1]
#     
#     all_path = nx.all_simple_paths(graph_i, source=start, target=end,cutoff=length + 5)
#     for path in all_path:
#         dtw_score = DTWs[scan_i](path, path_i)
#         if dtw_score < upper and dtw_score > lower:
#             data[t_i[0]]['path'] = path
#             print(t_i)
#             break
# =============================================================================


            
            

        
    
 
    