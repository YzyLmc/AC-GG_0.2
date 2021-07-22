#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:06:30 2021

@author: ziyi
"""
import os
# load if there is snapshot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", type=int)
args = parser.parse_args()

import json
# read snapshot
if args.snapshot:
    checkpoint = args.snapshot
    json_name = 'R2R_aug_snapshot/R2R_train_augmented_{}.json'.format(args.snapshot)
else:
    # read R2R dataset
    json_name = 'data/R2R_train.json'
with open(json_name,'r') as f:
    data = json.load(f)
from copy import deepcopy
data_new = deepcopy(data)
# languages for augmentation
lang_ls = ['ar','es','de','fr','hi','it','pt','ru','tr','zh']

# =============================================================================
# # generate itr raw ins for each language
# itr = 1
# =============================================================================

# translator
from google_trans_new import google_translator
translator = google_translator(timeout=5)
# filter out ins outside BLEU equals [0.25,0.7] (per ZHao et al.)
from nltk.translate.bleu_score import sentence_bleu as BLEU
upper_bleu = 0.7
lower_bleu = 0.25


############################################################
#                       augmenting                         # 
############################################################
# timer
import time
time_st = time.time()

# multiprocess
from multiprocessing.dummy import Pool as ThreadPool

def add_backed_ins(lang): # translate one instance
    
    interval_sentence = translator.translate(ins,lang_tgt=lang)
    translated_back = translator.translate(interval_sentence, lang_tgt='en') 
    
    #time.sleep(5)
    
    return translated_back
    
pool = ThreadPool(16)

# locate checkpoint
if args.snapshot:
    i_init = checkpoint
else:
    i_init = 0

# augmenting    
for i in range(i_init, len(data)):
    print(i)
    instance = data[i]
    ins_ls = instance['instructions']
    for ins in ins_ls:
        

# =============================================================================
#             #translate to target lang and then back
#             interval_sentence = translator.translate(ins,lang_tgt=lang)
#             translated_back = translator.translate(interval_sentence, lang_tgt='en')
# =============================================================================
        results = pool.map(add_backed_ins, lang_ls)
        
        # calculte bleu score and add good samples to the dataset
        for res_i in results:
            bscore = BLEU([ins], res_i) 
            if bscore > lower_bleu and bscore < upper_bleu:
                data_new[i]['instructions'].append(res_i)
                
    # sleep in case of being blocked
    if i % 5 == 0:
        time.sleep(60)
        
    # timer and save snapshots
    if i % 100 == 0:        
        time_i = time.time()
        print('done %s out of %s , time elapsed: %s, remaining: %s'%(i, len(data), time_i-time_st, (time_i-time_st)*len(data)/(3600*(i+1))))
    
    # save snapshot and delete last one
    if i % 10 == 0:
        filename = 'R2R_aug_snapshot/R2R_train_augmented_{}.json'.format(i)
        with open(filename,'w') as f:
            json.dump(data_new,f)
        try:
            old_i = i - 10
            filename_old = 'R2R_aug_snapshot/R2R_train_augmented_{}.json'.format(old_i)
            os.remove(filename_old)
        except:
            continue
        
#save new dataset
with open('R2R_train_augmented.json','w') as f:
    json.dump(data_new,f)
        
        
