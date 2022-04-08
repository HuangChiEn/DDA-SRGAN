#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:38:39 2020

@author: joseph
"""
from glob import glob
import pickle
import scipy.io as sio
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler

def pk2mat(conf_pk_lst, lab_pk_lst):
    def get_save_tag(path):
            raw_tag = path.split(".")[0]
            
            return raw_tag + ".mat"
    
    for cf, lb in zip(conf_pk_lst, lab_pk_lst):
        tag = get_save_tag(cf)
        print("saving : {}\n".format(tag))
        
        with open(cf, "rb") as f, open(lb, "rb") as t:
            cf_mtr, lab_mtr = pickle.load(f), pickle.load(t)
        norm = MinMaxScaler() # into range [ 0 ~ 1 ] 
        conf_tmp_mtr = norm.fit_transform(cf_mtr)
        conf_vec, label_vec = \
                    conf_tmp_mtr.reshape(-1), lab_mtr.reshape(-1)
            
        fpr, tpr, threshold = roc_curve(label_vec, conf_vec)
        
        sio.savemat(tag, {"fpr":fpr, "tpr":tpr})
        

if __name__ == "__main__":
    conf_pk_path_lst = glob("conf*.pk")
    lab_pk_path_lst = glob("lab*.pk")
    
    pk2mat(conf_pk_path_lst, lab_pk_path_lst)
