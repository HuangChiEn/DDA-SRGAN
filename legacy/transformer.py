#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:09:27 2020

@author: josephn

TODO : 
      1. Uniform the data_type of input (nd_array).
      2. Uniform the data_type of output.
"""
import numpy as np
import cv2

class processor:
    def __init__(self, data_src):
        self.data_src = data_src
    
    def __call__(self, trfs_typ, *args, **kwargs):
        self.trfs(trfs_typ, *args, **kwargs)
        
    ## interface for call trfs from outside of dataManager
    def trfs(self, trfs_typ, *args, **kwargs):
        assert isinstance(self.data_src, np.ndarray), \
                "ERROR_MESSAGE : The input data type of transformer should be np.ndarray type."
        
        transformer = trfs_dict[trfs_typ]
        self.data_src = transformer(self.data_src, *args, **kwargs)
        return self
    
## @ Transformer definition - alpha order : 
def flipper(data_src, flip_typ='mixed'):
    flip_dict = {'lr':np.fliplr, 'ud':np.flipud, 'mixed':np.flip}
    data_src = [ flip_dict[flip_typ](data) for data in data_src ]
    return data_src
        
def normalize(data_src, scalr, move_val):
    return data_src / scalr - move_val

def shuffle(data_src):
    np.random.shuffle(data_src)
    return data_src  ## dangling return.

def target_resize(data_src, target_size):
    data_src = [cv2.resize(data, target_size, interpolation=cv2.INTER_CUBIC)\
                    for data in data_src]
    return data_src

## execute self-define transformation
def self_trfs(data_src, trfs_func):
    return trfs_func(data_src)

## Alias for function name.
trfs_dict = {'flip':flipper, 'norm':normalize, 'shuf':shuffle,
             'target_resize':target_resize, 'self_trfs':self_trfs}

if __name__ == "__main__":
    prcs = processor(np.array([1, 3, 5, 7, 9, 11, 13, 17]))
    trf = lambda x: x+1
    print(prcs.trfs("shuf").trfs("self_trfs", trf).data_src)
    #print(prcs.data_src)
