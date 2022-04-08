#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:13:29 2021

Thanks for the SR benchmark open-source : 
    ## github repo : https://github.com/hieubkset/Keras-Image-Super-Resolution/blob/master/model/edsr.py
    
@author: hieubkset & josef.

All right's in benchmark will be reserved as the main author -- hieubkset,
    I (josef) just refactor the code and plugin my code..
"""
# system related
import os
import sys
sys.path.append("../../shared_module")
sys.setrecursionlimit(10000)


# self-define package
from common_interface import SR_base_model
from Residual_Group import Residual_Group
from dev_dataloader import Data_loader

# image read/write
import PIL.Image as pil_image
from glob import glob
import cv2


# Deep learning and data science
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Add, Lambda, Activation
from keras.utils import multi_gpu_model

SR_base_model
class RCAN(SR_base_model):
    
    def __init__(self, lr_height=50, lr_width=40, **_):
        self.lr_height, self.lr_width, self.channels = self.lr_shape = lr_height, lr_width, 3
        self.img_scalar = 4
        self.hr_height, self.hr_width, _ = [dim*self.img_scalar for dim in self.lr_shape]
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        
        tmpModel = self.build_generator()
        self.gen = multi_gpu_model(tmpModel, gpus=4)
        self.gen.compile(loss='mse', optimizer='adam')

    def build_generator(self, filters=32, n_sub_block=2):
        
        def Residual_in_Residual(input_tensor, filters, n_rg=10):
            x = input_tensor
            for _ in range(n_rg):
                x = Residual_Group(x, filters=filters)
            x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
            x = Add()([x, input_tensor])
        
            return x
        
        def upsample(input_tensor, filters):
        
            def sub_pixel_conv2d(scale=2, **kwargs):
                return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)
            
            x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
            x = sub_pixel_conv2d(scale=2)(x)
            x = Activation('relu')(x)
            return x
        
        inputs = Input(shape=self.lr_shape)
    
        x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
        x = Residual_in_Residual(x, filters=filters)
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = Add()([x_1, x])
    
        for _ in range(n_sub_block):
            x = upsample(x, filters)
        x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)
    
        model = Model(inputs=inputs, outputs=x)
        return model
    
    
    def training(self, batch_size=32, file_ext='bmp', **_):
        data_src_path = os.path.join("..", "..", 'datasets', 'CelebA', 'tra_set')
        data_loader = Data_loader(data_src_path, hr_img_size=(self.hr_height, self.hr_width), 
                                     scalr=self.img_scalar, ext="jpg")
        data_gen = data_loader.ld_data_gen(batch_size, fliplr=True)
        
        self.gen.fit_generator(data_gen, epochs=4, verbose=1, 
                               steps_per_epoch=10000//batch_size)
        self.gen.save_weights('./tmp_model.h5')
    
    
    def generating(self, **_):
        self.gen.load_weights('./tmp_model.h5')
        data_src_path = os.path.join("..", "..", 'datasets', 'CelebA', 'eval_set')
        
        data_loader = Data_loader(data_src_path, hr_img_size=(self.hr_height, self.hr_width), 
                                      scalr=self.img_scalar, ext="jpg")
        img_gen = data_loader.ld_generate_data_gen(1, fliplr=True)
        
        for idx, (lr, file_name) in enumerate(img_gen):
            #lr_img = cv2.resize(cv2.imread(file_name), (40, 50), interpolation=cv2.INTER_CUBIC)
            sr = self.gen.predict(lr)
            norm_img = np.clip(sr[0], a_min=-1, a_max=1)
            tmp = 0.5 * norm_img + 0.5
            
            #path = file_name.split(os.sep)[-1]
            imgs_templt = os.path.join("./tmp_gen/{}.jpg".format(file_name))
            nd_img = pil_image.fromarray(np.uint8(tmp * 255))
            nd_img.save(imgs_templt)
            print(idx)
    
    
    def evaluation(self):
        raise NotImplementedError("The evaluation procedure of RCAN should prograss on iris recognition.")
    
    
if __name__ == "__main__":
    model = RCAN()
    model.training()
    model.generating()
        
