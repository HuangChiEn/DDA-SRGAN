#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:45:01 2020

@author: joseph
"""

from keras.layers import Input, Conv2D, Dense, Lambda
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as KTF

def build_LVGG(input_shape=None, blk_typ_lst=[2, 3], 
         init_nfilt=64, ker_siz=(3, 3), dense_neur=1024, cls_num=1000):
    
    def ConvMax_blk(n_conv, filt_num, ker_siz, layIdx, tensor):
        for blkIdx in range(1, n_conv+1):
            tensor = Conv2D(filt_num, ker_siz,
                              activation='relu',
                              padding='same',
                              name="block{}_conv{}".format(layIdx, blkIdx))(tensor)
            
        tensor = MaxPooling2D((2, 2), strides=(2, 2), 
                                  name="block{}_pool".format(layIdx))(tensor)
        return tensor
    
    x = Input(input_shape)
    tensor = x
    for layIdx, n_conv in enumerate(blk_typ_lst, 1):
        tensor = ConvMax_blk(n_conv, init_nfilt*layIdx, ker_siz, layIdx, tensor)
        
    x_fea = GlobalAveragePooling2D()(tensor)
    x_fea = Dense(1024, activation='relu', name='fc1')(x_fea)
    x_cls = Dense(cls_num, activation='softmax', name='prediction')(x_fea)
    
    return Model(inputs=x, outputs=x_cls, name='lvgg')