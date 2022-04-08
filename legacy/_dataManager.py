#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Thu Aug 20 14:24:13 2020
@author: josephn

dataloader - 

"""

import cv2
from data_loader import ld_func_dict
from data_loader import processor

class DataManager:
    
    def __init__(self, ld_typ, gen_typ):
        self.data_ld = ld_func_dict[ld_typ]
        
    def ld_data(self, *args, **kwargs):
        self.filLst = self.data_ld(*args, **kwargs)
        
    def trfs_gen(self, trans_typ):
        gen = self.get_genenator() ## TODO:request adjustment..
        for data in gen:
            prcs = processor(data)
            for trf_typ in trans_typ:
                prcs(trf_typ)
            yield prcs.data_src
        
    def get_genenator(self, epochs=1, batch_size=12):
        
        def get_batch_img():
            ## Add limitation of iteration for StopIteration security
            for _ in iteration:
                yield [cv2.imread(next(fil_itr)) \
                           for _ in batch_size]
        
        iteration = len(self.filLst) // batch_size
        fil_itr = iter(self.filLst)
        for _ in range(epochs):
            yield from get_batch_img(fil_itr)
        
    ## For asynchronously loading the data.
    def manger_socket():
        raise NotImplementedError
    
    
if __name__ == "__main__":
    pass