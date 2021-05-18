#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 23:42:52 2021

@author: ziyi
"""


from nltk.translate.bleu_score import sentence_bleu as BLEU
seq_i = [1,2,3,4,5]
pred_i = [1,2,3,4,6]



for i in range(len(pred_i)):
    G = 0
    print('i = {}'.format(i))
    for j in range(len(pred_i)-i,len(pred_i)+1):
        if j > 0:
            
            G = G + BLEU([seq_i],pred_i[:j],weights=(1/2,1/2)) - BLEU([seq_i],pred_i[:j-1],weights=(1/2,1/2))
        else:
            G = G + BLEU([seq_i],pred_i[:j],weights=(1/2,1/2))
        print('j = {}'.format(j))
        print(G)
    print(G)
            
        
