#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:06:30 2021

@author: ziyi
"""

# read R2R dataset
import json
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

for i in range(len(data)):
    # report time EST
    if i % 1 == 0:        
        time_i = time.time()
        print('done %s out of %s , time elapsed: %s, remaining: %s'%(i, len(data), time_i-time_st, (time_i-time_st)*len(data)/(3600*(i+1))))
        filename = 'R2R_train_augmented_{}.json'.format(i)
        with open(filename,'w') as f:
            json.dump(data_new,f)
    instance = data[i]
    ins_ls = instance['instructions']
    for ins in ins_ls:
        for lang in lang_ls:

            #translate to target lang and then back
            interval_sentence = translator.translate(ins,lang_tgt=lang)
            translated_back = translator.translate(interval_sentence, lang_tgt='en')
            # calculte bleu score
            bscore = BLEU([ins], translated_back) 
            #print(translated_back,bscore)
            if bscore > lower_bleu and bscore < upper_bleu:
                data_new[i]['instructions'].append(translated_back)

    
    
#save new dataset
with open('R2R_train_augmented.json','w') as f:
    json.dump(data_new,f)
        
        