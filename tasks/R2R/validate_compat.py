#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 20:27:50 2021

@author: ziyi
"""


import utils
import train_compat

def validate_entry_point(args):
    agent, train_env, val_envs = train_compat.train_setup(args)
    agent.load(args.model_prefix)

    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        agent.env = val_env
        
        # predicted
        pred_results = agent.test(use_dropout=False)

        TN = 0
        FN = 0
        TP = 0
        FP = 0
        for item in pred_results.values():
            if item['label'] == 0:
                if item['predict'] < 0.5:
                    TN += 1
                elif item['predict'] > 0.5:
                    FP += 1
                    
            if item['label'] == 1:
                if item['predict'] < 0.5:
                    FN += 1
                elif item['predict'] > 0.5:
                    TP += 1
        accuracy = (TN + TP) / (TN + FN + TP + FP)  
        precision = TP / (TP + FP)
        recall = TP/ (TP + FN)
        f1 = 2*TP / (2*TP + FP + FN)
        print('TN = {}, FN = {}, TP = {}, FP ={}'.format(TN,FN,TP,FP))
        print("{} accuracy = {}, precision = {}, recall = {}, f1 = {}".format(env_name, accuracy, precision, recall, f1))

def make_arg_parser():
    parser = train_compat.make_arg_parser()
    parser.add_argument("model_prefix")
    parser.add_argument("--gold_results_output_file")
    parser.add_argument("--pred_results_output_file")
    # parser.add_argument("--beam_size", type=int, default=1)
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
