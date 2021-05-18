#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 01:05:58 2021

@author: ziyi
"""


class test1():
    def __init__(self):
        print('test1 init...')
    def plus(self,a,b):
        return a+b
    
def test2(a,b):
    t_test = test1()
    return t_test.plus(a,b)