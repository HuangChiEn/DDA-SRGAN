from __future__ import print_function

import tensorflow.keras as tf_ker
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras import backend as KTF
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from math import sqrt, floor
from numpy import expand_dims, argmax
from matplotlib import pyplot
from collections import namedtuple
from os import environ
import argparse

tf.compat.v1.disable_eager_execution()

def visu_net(model, n_map):
    sub_mod = model
    # prepare the image (e.g. scale pixel values for the vgg)
    img = load_img('./cls1_1.bmp')
    prepro = lambda img : expand_dims(img_to_array(img), axis=0)
    img = preprocess_input(prepro(img))
    
    feature_maps = sub_mod.predict(img)

    n_fmap = floor(sqrt(feature_maps.shape[-1]))
    n = n_map
    square = n if n <= n_fmap else n_fmap
    ix = 1
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
            
    parser = argparse.ArgumentParser()
    parser.add_argument("--lay_idx", type=int, default=15)
    parser.add_argument("--show_mod", type=bool, default=True)
    parser.add_argument("--n_fmap", type=int, default=4)
    args = parser.parse_args()
    
    res_net = ResNet50(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None
    )
    
    layer_info_lst = []
    Lay_struct = namedtuple('Lay_struct', 'lay_idx, name, shape')
    for lay_idx, layer in enumerate(res_net.layers):
        # check for convolutional layer
        #if 'conv' not in layer.name: continue
        
        lay_strut = Lay_struct(lay_idx=lay_idx, name=layer.name, shape=layer.output.shape)
        layer_info_lst.append(lay_strut)
        
        # summarize output shape
        lay_str = "layer index - {}, name - {} ; shape = {}\n"
        print(lay_str.format(lay_idx, layer.name, layer.output.shape))
        
    sub_mod = Model(inputs=res_net.inputs, outputs=res_net.layers[args.lay_idx].output)
    args.show_mod and sub_mod.summary()
    
    visu_net(sub_mod, args.n_fmap)
    
    KTF.clear_session()