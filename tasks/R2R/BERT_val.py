#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:40:07 2021

@author: ziyi
"""


import json
import numpy as np
import torch

with open('speaker/BERT400_val_unseen.json') as f:   
    
    result = json.load(f)
    
with open('data/R2R_val_unseen.json') as f:
    
    answer = json.load(f)
    
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from bert_score import BERTScorer
scorer = BERTScorer(lang='en', rescale_with_baseline = True)
#%%
scores = []
for traj in answer:
    instr_id = traj['path_id']
    for i in range(3):
        instr_id_i = '{}_{}'.format(instr_id, i)
        instr_ans = [traj['instructions'][i]]
        instr_pred = [' '.join(result[instr_id_i]['words'])]
        
        _, _, F1 = scorer.score(instr_pred, [instr_ans])
        
        scores.append(F1.numpy())
        
print(np.mean(scores))

        
    