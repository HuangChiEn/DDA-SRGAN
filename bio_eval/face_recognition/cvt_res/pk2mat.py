#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:38:39 2020

@author: joseph
"""
from glob import glob
import pickle
import scipy.io as sio

def pk2mat(pk_path_lst):
    def get_save_tag(path):
            raw_tag = path.split(".")[0]
            
            return raw_tag + ".mat"
    
    for path in pk_path_lst:
        tag = get_save_tag(path)
        print("saving : {}\n".format(tag))
        with open(path, "rb+") as f_ptr:
            np_mtr = pickle.load(f_ptr)
            
        sio.savemat(tag, {"conf_mtr":np_mtr})

if __name__ == "__main__":
    conf_pk_path_lst = glob("conf*.pk")
    #lab_pk_path_lst = glob("./lab*.pk")
    #pk2mat(conf_pk_path_lst)
    pk2mat(["conf_mtr_lr.pk"])
    #pk2mat(lab_pk_path_lst)
