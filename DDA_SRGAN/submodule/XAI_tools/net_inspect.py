from __future__ import print_function

import tensorflow as tf

from generator import build_generator
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from keras import backend as KTF
from keras.models import Model

from math import sqrt, floor
from numpy import expand_dims, argmax
from matplotlib import pyplot as plt
from collections import namedtuple
import cv2
import argparse
import glob
from os import environ
from sklearn.preprocessing import MinMaxScaler

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
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        plt.show()

    
if __name__ == "__main__":
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    gen = build_generator((50, 40, 3), 32, 3, 3, 0.2, 4)
    wei_path = "../../pretrain/SWCA_GAN/CelebA/generator/000_best.h5"
    
    gen.load_weights(wei_path)
    model = Model(inputs=gen.input, outputs=gen.layers[-1].output)
    #visu_net(model, 4)
    #print("layers : {}".format(len(gen.layers)))  # 548 layers
    #model.summary()
    
    sub_mod = model
    # prepare the image (e.g. scale pixel values for the vgg)
    im_paths = glob.glob("./swca_sample/*")
    img_lst = [ load_img(path) for path in im_paths ]
    img_lst = [ cv2.resize(src=img_to_array(img), dsize=(40, 50), interpolation=cv2.INTER_CUBIC) for img in img_lst ]
    img_lst = [ img / 127.5 - 1.0 for img in img_lst]
    img_lst = [ expand_dims(img, axis=0) for img in img_lst ]
    
    norm = MinMaxScaler(feature_range=(0.0, 255.0))
    feature_map_lst = []
    
    for img in img_lst:
        feature_maps = sub_mod.predict(img)
        feature_maps = feature_maps * 0.5 + 0.5
        feature_map = feature_maps.sum(axis=3)
        feature_map = norm.fit_transform(feature_map[0])
        feature_map_lst.append(feature_map)
    
    #print(feature_map_lst[0].min(), feature_map_lst[0].max())
    
    
    for fea_map in feature_map_lst:
        fig = plt.imshow(fea_map, cmap=plt.cm.hot_r)
        plt.xticks([]) ; plt.yticks([])
        plt.colorbar(fig)
        plt.show()
    
    
    
    