#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:40:36 2020

@author: k0 (first contributor) && 
               joseph*(same contribution)
"""

import keras.backend.tensorflow_backend as KTF 
import tensorflow as tf

## Relativistic Leaste squre Average GAN loss function : 
# Define generator and discriminator losses according to RaGAN described in Jolicoeur-Martineau (2018).
# Dummy predictions and trues are needed in Keras. 

## RLaGAN - loss  <more stable training process>
def custom_rela_dis_loss(dis_real, dis_fake):
    
    def rel_dis_loss(dummy_pred, dummy_true):
        ## Critic term ( output before activation ) 
        real_diff = dis_real - KTF.mean(dis_fake, axis=0)
        fake_diff = dis_fake - KTF.mean(dis_real, axis=0)

        return KTF.mean(KTF.pow(real_diff-1,2),axis=0)+\
                KTF.mean(KTF.pow(fake_diff+1,2),axis=0)
                 
    return rel_dis_loss

def custom_rela_gen_loss(dis_real, dis_fake):
    
    def rel_gen_loss(dummy_pred, dummy_true):
        ## Critic term ( output before activation ) 
        real_diff = dis_real - KTF.mean(dis_fake, axis=0)
        fake_diff = dis_fake - KTF.mean(dis_real, axis=0)
        
        return KTF.mean(KTF.pow(fake_diff-1,2),axis=0)+\
                KTF.mean(KTF.pow(real_diff+1,2),axis=0)
            
    return rel_gen_loss


# perceptual loss
def perceptual_loss(fake_fea, real_fea):
    percept_loss = tf.losses.mean_squared_error(fake_fea, real_fea)
    return percept_loss

# pixel loss
def pixel_loss(fake_hr, img_hr):
    pixel_loss = tf.losses.absolute_difference(fake_hr, img_hr) 
    return pixel_loss