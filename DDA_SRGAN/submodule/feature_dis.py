import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.convolutional import Conv2D
import keras.backend as K

## keras pretraining model : for extract activated feature
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


def build_feature_dis(hr_shape):
    #FREEUSE.1 VGG19 with 512 feature map in last layer (fine-tune).
    #          fine-tune - loss(VGG(hr) - VGG(sr)) -> classification.
    ##---------------<VGG19 - structure : >-----------------------##
    oriVGG = VGG16(weights="imagenet", include_top=False)
    VGG_out = oriVGG.get_layer('block5_conv3').output
    VGGBef= Model(inputs=oriVGG.input, outputs=VGG_out)
    ## ----------------------------------------------------------------------##    
    
    ## prerpocess input.
    img = Input(shape=hr_shape)
    prepro_img = preprocess_vgg(img)
    img_features = VGGBef(prepro_img)
    return Model(img, img_features, name='feaDis')

def preprocess_vgg(x):
    ## Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network
    if isinstance(x, np.ndarray):
        return preprocess_input((x + 1) * 127.5)
    else:
        return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x) 
