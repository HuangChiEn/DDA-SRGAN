from __future__ import print_function

import tensorflow as tf

from generator import build_generator
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from keras import backend as KTF
from keras.models import Model

from math import sqrt, floor
from numpy import expand_dims, argmax
from matplotlib import pyplot
from collections import namedtuple
import cv2
import argparse
from os import environ

tf.compat.v1.disable_eager_execution()

def visu_net(model, n_map):
    sub_mod = model
    # prepare the image (e.g. scale pixel values for the vgg)
    img = load_img('./064698.jpg')
    img = cv2.resize(src=img_to_array(img), dsize=(40, 50), interpolation=cv2.INTER_CUBIC)
    exp_img = expand_dims(img, axis=0)
    
    feature_maps = sub_mod.predict(exp_img)

    n_fmap = floor(sqrt(feature_maps.shape[-1]))
    n = n_map
    square = n if n <= n_fmap else n_fmap
    print(feature_maps.shape)
    ix = 1
    
    if False:
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        pyplot.show()

    
if __name__ == "__main__":
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    gen = build_generator((50, 40, 3), 32, 3, 3, 0.2, 4)
    wei_path = "../../pretrain/SWCA_GAN/CelebA/generator/000_best.h5"
    
    gen.load_weights(wei_path)
    model = Model(inputs=gen.input, outputs=gen.layers[-2].output)
    #visu_net(model, 4)
    #print("layers : {}".format(len(gen.layers)))  # 548 layers
    #model.summary()
    
    sub_mod = model
    # prepare the image (e.g. scale pixel values for the vgg)
    img = load_img('./064698.jpg')
    img = cv2.resize(src=img_to_array(img), dsize=(40, 50), interpolation=cv2.INTER_CUBIC)
    exp_img = expand_dims(img, axis=0)
    
    feature_maps = sub_mod.predict(exp_img)
    # 
    feature_maps = feature_maps.sum(axis=3)  
    print(feature_maps.shape)
    ix = 1
    pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
    
    
    
    