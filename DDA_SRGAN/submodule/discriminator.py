'''keras package : the high-level module for deep learning '''
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Add, Flatten, GlobalAveragePooling2D, LeakyReLU
from keras.layers.convolutional import Conv2D

def build_discriminator(hr_shape, num_of_filts):
    
    def d_block(layer_input, filters, strides=1, bn=True, kerSiz=3):
        d = Conv2D(filters, kernel_size=kerSiz, strides=strides, padding='same',\
                   activation='selu', kernel_initializer='lecun_uniform')(layer_input)
        return d
    
    in_tensor = Input(shape=hr_shape)
    d1 = Conv2D(num_of_filts, strides=1, kernel_size=3, padding='same',\
                activation='selu', kernel_initializer='lecun_uniform')(in_tensor)
    d_rec = Conv2D(num_of_filts, strides=2, kernel_size=4, padding='same',\
                activation='selu', kernel_initializer='lecun_uniform')(d1)

    cnt = 1
    n_filt = num_of_filts
    for idx in range(0, 4):
        if(idx%2 == 1):
            mul_filt = 2**cnt
            n_filt = num_of_filts * mul_filt
        d_rec = d_block(layer_input=d_rec, filters=n_filt)
        d_rec = d_block(layer_input=d_rec, filters=n_filt, strides=2, kerSiz=4)

    d8 = d_rec
    d9 = GlobalAveragePooling2D()(d8)
    d10 = Dense(100, activation='selu', kernel_initializer='lecun_uniform')(d9)
    validity = Dense(1, activation='tanh', kernel_initializer='glorot_uniform')(d10)
    return Model(inputs=in_tensor, outputs=validity, name='Discriminator')
    