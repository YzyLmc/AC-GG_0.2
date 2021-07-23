#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 21:18:10 2021

@author: ziyi
"""


import json
import numpy as np
import torch

with open('speaker/origin_val_unseen.json') as f:   
    
    result = json.load(f)
    
with open('data/R2R_val_unseen.json') as f:
    
    answer = json.load(f)

pairs = []  
for traj in answer:
    item = {}
    instr_id = traj['path_id']
    for i in range(3):
        instr_id_i = '{}_{}'.format(instr_id, i)
        instr_pred = ' '.join(result[instr_id_i]['words'])
        
        item["image_id"] = instr_id_i
        item["refs"] = traj['instructions']
        item["test"] = instr_pred
        
    pairs.append(item)
    
jsfile = 'origin_unseen.json'
with open(jsfile,'w') as f:
    json.dump(pairs,f)