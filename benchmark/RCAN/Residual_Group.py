#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:13:28 2021

@author: joseph
"""

from keras.layers import Conv2D, Activation, Add, Lambda, GlobalAveragePooling2D, Multiply, Dense, Reshape


def Residual_Channel_Attention_Block(input_tensor, filters, scale=0.1):

    def Channel_Attention(input_tensor, filters, reduce=16):
        x = GlobalAveragePooling2D()(input_tensor)
        x = Reshape((1, 1, filters))(x)
        x = Dense(filters//reduce,  activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
        x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
        x = Multiply()([x, input_tensor])
        return x
    
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Channel_Attention(x, filters)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x


def Residual_Group(input_tensor, filters, n_rcab=20):
    x = input_tensor
    for _ in range(n_rcab):
        x = Residual_Channel_Attention_Block(x, filters=filters)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x

    