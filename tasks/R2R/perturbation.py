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
#%%
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
    
    if len(viewpoint)%10 == 0:
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
with open('traj_viewpoint.json','r') as f:
    traj_viewpoint = json.load(f)
    
#%%
part_2 = partitions['random'] + partitions['phrase']
phrase = []
random = []
for t_i in part_2:
    ls = re.split('\, +|\. ',data[t_i[0]]['instructions'][t_i[1]])
    ls = [elm for elm in ls if elm != '']
    if len(ls) > 1 and len(phrase) < 7392:
        phrase.append(t_i)
    else:
        random.append(t_i)
partitions['random'] = random
partitions['phrase'] = phrase 
       
with open('parted_idx.json','w') as f:
   json.dump(partitions,f)       
    
#%% start constructing
   
hardNeg = []
gt = {}
#%% entity and random (which is reversal actually)

## load responses
with open('responses_train_new.json') as f:
    resp_1 = json.load(f)
    
with open('responses_train_new_random.json') as f:
    resp_random = json.load(f)
  
def getEnts(entity_dic): # get entity lists from responses
    entity_ls = entity_dic['entities']
    nums = entity_ls[-5:]
    num_ls = []
    for num_i in nums:
        try:
            num = num_i['mentions'][0]['text']['beginOffset']
            num_ls.append(num)
        except:
            continue
    num_ls_sorted = sorted(num_ls)
    entities = [[] for _ in range(6)]
    for ent in entity_ls[:-5]:
        try:
            if ent['name'] == '1250':
                continue
            num = ent['mentions'][0]['text']['beginOffset']
            if num < num_ls_sorted[0]:
                entities[0].append(ent['name'].lower())
            elif num < num_ls_sorted[1]:
                entities[1].append(ent['name'].lower())
            elif num < num_ls_sorted[2]:
                entities[2].append(ent['name'].lower())
            elif num < num_ls_sorted[3]:
                entities[3].append(ent['name'].lower())
            elif num < num_ls_sorted[4]:
                entities[4].append(ent['name'].lower())
            else:
                entities[5].append(ent['name'].lower())
        except:
            continue
                    
    return entities

#read random responses
ents_ls_random = []
for ent_dic in resp_random:
    ents = getEnts(ent_dic)
    ents_ls_random += ents
#read all 4 sections of responses
ents_ls_1 = []
for ent_dic in resp_1:
    ents = getEnts(ent_dic)
    ents_ls_1 += ents
    

to_random = []  # transaction from entity to random
def checkEqual(ls):  # check if a list containes different elements
    count = 0
    note = []
    for item in ls:
        if item not in note:
            note.append(item)
            count += 1
    if count <= 1:
        return True
    elif count > 1:
        return False
    
for i in range(len(ents_ls_1[:7392])):
    ents_ls_i = ents_ls_1[i]
    if checkEqual(ents_ls_i):
        to_random.append(i)
        #print(ents_ls_i)
    
ents_ls_needed = len(to_random) # number of data needed from rndom to entity


to_entity = [] # transaction from random to entity
for i in range(7392):
    ents = ents_ls_random[i]
    if len(to_entity) < ents_ls_needed:   
        if not checkEqual(ents):
            to_entity.append(i)
            #t_i = partitions['random'][i]
            #sentence = data[t_i[0]]['instructions'][t_i[1]]
            #print(ents,sentence)
    else:break

for i in range(len(partitions['entity'])):
    t_i = partitions['entity'][i]
    data_i = data[t_i[0]]
    ins_i = deepcopy(data_i['instructions'][t_i[1]])
    ent_ls = Tokenizer.split_sentence(ins_i)
    ent_ls = [elm for elm in ent_ls if elm != '']
    
    ents_i = ents_ls_1[i]
    if i in to_random: # skip to_random datas
        continue
    new_ent_ls = []
    for ent in ent_ls:
        if ent in ents_i:
            new_ent = ent
            while new_ent == ent:
                new_ent = random.choice(ents_i)
            ent = new_ent
        new_ent_ls.append(ent)
        


    new_ins = ' '.join(new_ent_ls)  
    new_item = dict(data_i)
    new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
    new_item['instructions'] = new_ins
    new_item['path'] = data_i['path']
    new_item['label'] = 0
    new_item['perturb'] = 'entity'
    hardNeg.append(new_item)
    path_id = str(data_i['path_id']) + '_0'
    gt[path_id] = data_i 
#%%   
for i in range(len(to_entity)): # from random to entity
    t_i = partitions['random'][to_entity[i]]
    data_i = data[t_i[0]]
    ins_i = deepcopy(data_i['instructions'][t_i[1]])
    ent_ls = Tokenizer.split_sentence(ins_i)
    ent_ls = [elm for elm in ent_ls if elm != '']
    
    ents_i = ents_ls_random[to_entity[i]]
    #print(ins_i, ents_i)
    new_ent_ls = []
    for ent in ent_ls:
        if ent in ents_i:
            new_ent = deepcopy(ent)
            while new_ent == ent:
                new_ent = random.choice(ents_i)
            ent = new_ent
        new_ent_ls.append(ent)
        


    new_ins = ' '.join(new_ent_ls)  
    new_item = dict(data_i)
    new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
    new_item['instructions'] = new_ins
    new_item['path'] = data_i['path']
    new_item['label'] = 0
    new_item['perturb'] = 'entity'
    hardNeg.append(new_item)
    path_id = str(data_i['path_id']) + '_0'
    gt[path_id] = data_i 
#%%    
 # I messed up 'random' and 'reversal'...
for i in range(len(partitions['random'])):
    if i in to_entity:
        continue
    t_i = partitions['random'][i]
    data_i = data[t_i[0]]
    path_i = deepcopy(data_i['path'])
    path_i.reverse()
    new_item = dict(data_i)
    new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
    new_item['instructions'] = data_i['instructions'][t_i[1]]
    new_item['path'] = path_i
    new_item['label'] = 0
    new_item['perturb'] = 'reversal'
    hardNeg.append(new_item)
    path_id = str(data_i['path_id']) + '_0'
    gt[path_id] = data_i
    
for i in range(len(to_random)):
    t_i = partitions['random'][to_random[i]]
    data_i = data[t_i[0]]
    path_i = deepcopy(data_i['path'])
    path_i.reverse()
    new_item = dict(data_i)
    new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
    new_item['instructions'] = data_i['instructions'][t_i[1]]
    new_item['path'] = path_i
    new_item['label'] = 0
    new_item['perturb'] = 'reversal'
    hardNeg.append(new_item)
    path_id = str(data_i['path_id']) + '_0'
    gt[path_id] = data_i
    
#%%
random_count = 1
for i in range(len(partitions['viewpoint'])):
    t_i = partitions['viewpoint'][i]
    data_i = data[t_i[0]]
    path_i = traj_viewpoint[i]
    
    if len(path_i) > 10 and random_count < 7392 :
        path_i = path_i[:5]
        new_item = dict(data_i)
        new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
        new_item['instructions'] = data_i['instructions'][t_i[1]]
        new_item['path'] = path_i
        new_item['label'] = 0
        new_item['perturb'] = 'random'
        hardNeg.append(new_item)
        path_id = str(data_i['path_id']) + '_0'
        gt[path_id] = data_i
        random_count += 1
        continue
    

    new_item = dict(data_i)
    new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
    new_item['instructions'] = data_i['instructions'][t_i[1]]
    new_item['path'] = path_i
    new_item['label'] = 0
    new_item['perturb'] = 'viewpoint'
    hardNeg.append(new_item)
    path_id = str(data_i['path_id']) + '_0'
    gt[path_id] = data_i      
#%%
import random
for t_i in partitions['phrase']:
    data_i = data[t_i[0]]
    ins_i = deepcopy(data_i['instructions'][t_i[1]])
    phrase_ls = re.split('\, +|\. ',ins_i)
    phrase_ls = [elm for elm in phrase_ls if elm != '']
    swapped = False
    while not swapped:
        new_phrase_ls = deepcopy(phrase_ls)
        random.shuffle(new_phrase_ls)

        if not new_phrase_ls == phrase_ls:
            swapped = True
    new_ins = '. '.join(new_phrase_ls) + '. '   
    new_item = dict(data_i)
    new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
    new_item['instructions'] = new_ins
    new_item['path'] = data_i['path']
    new_item['label'] = 0
    new_item['perturb'] = 'phrase'
    hardNeg.append(new_item)
    path_id = str(data_i['path_id']) + '_0'
    gt[path_id] = data_i
#%%

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
d_ls = [d_1,d_2,d_3,d_4,d_5,d_6,d_7,d_8,d_9,d_10]
d_all = d_1+d_2+d_3+d_4+d_5+d_6+d_7+d_8+d_9+d_10    
for t_i in partitions['direction']:
    data_i = data[t_i[0]]
    ins_i = deepcopy(data_i['instructions'][t_i[1]])
    words = Tokenizer.split_sentence(ins_i)
    words_new = []
    for word in words:
        if word in d_all:
            for d_ls_i in d_ls:
                if word in d_ls_i:
                    word_new = word
                    while word == word_new:
                        word_new = random.choice(d_ls_i)
                    word = word_new
        words_new.append(word)
    new_ins = ' '.join(words_new)
    new_item = dict(data_i)
    new_item['instr_id'] = '%s_%d' % (data_i['path_id'], t_i[1])
    new_item['instructions'] = new_ins
    new_item['path'] = data_i['path']
    new_item['label'] = 0
    new_item['perturb'] = 'direction'
    hardNeg.append(new_item)
    path_id = str(data_i['path_id']) + '_0'
    gt[path_id] = data_i
    
#%%
with open('hardNeg_train.json','w') as f:
    json.dump(hardNeg,f)
#%%
with open('gt_train.json','w') as f:
    json.dump(gt,f)
#%%
    
with open('hardNeg_train.json','r') as f:
    hardNeg = json.load(f)
    
from vocab import SUBTRAIN_VOCAB, TRAIN_VOCAB, TRAINVAL_VOCAB    
from utils import read_vocab, Tokenizer    
vocab = read_vocab('data/train_vocab.txt')
tok = Tokenizer(vocab=vocab)
#%%
for item in hardNeg:
    instr = item['instructions']
    item['instr_encoding'], item['instr_length'] = tok.encode_sentence(instr)
    


        
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


            
            

        
    
 
    
