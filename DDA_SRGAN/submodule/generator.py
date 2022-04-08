from keras.models import Model
from keras.layers import Add, Concatenate, Input, Dense, Activation, Subtract
from keras.layers import GlobalAveragePooling2D, Multiply, Layer, AveragePooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D, Conv2D, Conv2DTranspose
from keras.layers.core import Lambda, Reshape
from keras.engine import InputSpec
import keras.backend as KTF
import tensorflow as tf 
import cv2
## HACKME.1 : https://mlfromscratch.com/activation-functions-explained/
##          In currently, I attempt to use SELU to prevent any gradient related problem,
##              however, the speed will slower due to expotential computation.  
##          https://arxiv.org/pdf/1905.01338.pdf ; and the effect is unknown for SELU.

##          Add. https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
##          The above tips for training GAN suggest ReLU to prevent gradient vanishment,
##              I think that, we use SELU is suitable for solve this probelm.

def SWCA_Blk(input_tensor, n_filt, resScal=0.2):
    
    ## Spatial Attention Block
    def SA_Blk(input_tensor, reduction_ratio=4, dilated_rate=(2, 2)):
        raw_tensor = input_tensor
        lst = input_tensor.get_shape().as_list()
        loc_n_filt = lst[-1]
        
        cmpress_fea_map = Conv2D(filters=(loc_n_filt//reduction_ratio), kernel_size=1, 
                                     kernel_initializer='glorot_uniform', 
                                         padding='same')(input_tensor)
        
        spat_att_map = Conv2D(filters=(loc_n_filt//reduction_ratio), kernel_size=(3, 3), 
                                  dilation_rate=dilated_rate, 
                                      kernel_initializer='glorot_uniform', 
                                          padding='same')(cmpress_fea_map)
        
        spat_att_map = Conv2D(filters=(loc_n_filt//reduction_ratio), kernel_size=(3, 3), 
                                  dilation_rate=dilated_rate,
                                      kernel_initializer='glorot_uniform', 
                                          padding='same')(spat_att_map)
        
        cmpress_spat_map = Conv2D(filters=loc_n_filt, kernel_size=1, activation='sigmoid',
                                      kernel_initializer='glorot_uniform', 
                                          padding='same')(spat_att_map)
        
        spatial_atten_fea_map = Multiply()([cmpress_spat_map, raw_tensor])
        
        return spatial_atten_fea_map
    
    
    ## Channel Attention Block
    def CA_Blk(x, reduction_ratio=4):
        x = Conv2D(filters=n_filt, kernel_size=3, activation='selu', 
                                       kernel_initializer='lecun_uniform', padding='same')(x)
        fea_map_out = fea_map_innr = Conv2D(filters=n_filt, kernel_size=3, padding='same')(x)
        
        ## Channel Attention inner blk -- 
        ## GAP with keep 4-d shape.
        avg_fea = Lambda(lambda x : tf.reduce_mean(x, axis=[1, 2]))(fea_map_innr)
        fea_map_scalr = Lambda(lambda x : tf.reshape(x, (-1, 1, 1, n_filt)))(avg_fea)
        ## Botton-Net structure -
        dwn_sam_scalr = Conv2D(filters=(n_filt//reduction_ratio), 
                               kernel_size=1, activation='selu', 
                                   kernel_initializer='lecun_uniform')(fea_map_scalr)
        
        up_sam_scalr = Conv2D(filters=n_filt, kernel_size=1, activation='sigmoid', 
                              kernel_initializer='glorot_normal')(dwn_sam_scalr)
        return Multiply()([fea_map_out, up_sam_scalr])
        
    
    
    # for residual addition
    long_skip_connt = short_skip_connt = input_tensor  
    # for concatenation
    x_1 = input_tensor   
    
    # Spatial Attention weighted feature map 
    #  take as the input of the Channel Attention Block with dense connection
    x = CA_Blk(input_tensor)
    x = SA_Blk(x)
    x_2 = x = Concatenate(axis=-1)([x_1, x])
    
    x = CA_Blk(x)
    x = SA_Blk(x)
    x = Concatenate(axis=-1)([x_2, x])
    
    
    ## Add residual connection per 2 SWCA Blk
    #   Down-dimension via express the filter channel with 1x1 convBlock.
    x = Conv2D(filters=n_filt, kernel_size=1, activation='selu', 
           kernel_initializer='lecun_uniform')(x)
    x_3 = short_skip_connt = x = Add()([short_skip_connt, x])
    
    
    ## again,  SWCA Blk with Dense connection
    x = CA_Blk(x)
    x = SA_Blk(x)
    x_4 = x = Concatenate(axis=-1)([x_3, x])
    
    x = CA_Blk(x)
    x = SA_Blk(x)
    x = Concatenate(axis=-1)([x_4, x])  
    
    ## Add residual connection per 2 SWCA Blk
    #   Down-dimension via express the filter channel with 1x1 convBlock.
    x = Conv2D(filters=n_filt, kernel_size=1, activation='selu', 
                     kernel_initializer='lecun_uniform', padding='same')(x)
    x = Add()([short_skip_connt, x])  
    
    ## Last Conv2D + long skip connection with residual scaling
    x = Conv2D(filters=n_filt, kernel_size=3, strides=1, padding='same')(x)
    x = Lambda(lambda x: x * resScal)(x)
    x = Add()([long_skip_connt, x])
    
    return x


class ICNR:
    """ICNR initializer for checkerboard artifact free sub pixel convolution
    Ref:
     [1] Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
     https://arxiv.org/pdf/1707.02937.pdf)
    Args:
    initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: scale factor of sub pixel convolution
    """

    def __init__(self, initializer, scale=2):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype=tf.float32, partition_info=None):
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)

        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        x = self.initializer(new_shape, dtype, partition_info)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize_nearest_neighbor(x, size=(shape[0] * self.scale, shape[1] * self.scale))
        x = tf.space_to_depth(x, block_size=self.scale)
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        return x

def SubpixelConv2D(input_shape, scale=2):
    def subpixel_shape(input_shape):
        dims = [input_shape[0], 
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3]/(scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    return Lambda(lambda x: tf.depth_to_space(x, scale), output_shape=subpixel_shape)


def build_generator(lr_shape, num_of_filts, num_of_RRDRB, 
                    num_of_DRB, resScal, upScalar):

    ## Note : Conv2D should use 'tanh' activation and ICNR(glorot_uniform_initializer) !!
    ## Note - and we should progressive upsampling !!
    def cv_upsampling(in_tensor, filters, scalar=2): 
        
        def bicubic(tensor, height, width):
            return Lambda(lambda x : tf.image.resize_bicubic(x, size=(height*2, width*2)))(tensor)

        ## Work in tf 2.x
        def lanczo(tensor, height, width):
            return Lambda(lambda x :  tf.image.resize(x, size=(height*2, width*2), method='lanczos5'))(tensor)

        lst = in_tensor.get_shape().as_list()
        x = Lambda(lambda tensor : bicubic(tensor, height=lst[1], width=lst[2]))(in_tensor)  ## see HACKME.0
        out_tensor = Conv2D(filters, kernel_size=3, strides=1, padding='same', \
                   kernel_initializer='glorot_uniform')(x)
        
        return out_tensor
    
    
    def subpix_upsampling(in_tensor, filters, scalar=2, reduction_ratio=4):
        ## Convolution and the pixel shuffling.. 
        x = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='tanh',
               kernel_initializer=ICNR(tf.glorot_uniform_initializer()))(in_tensor)
        fea_map = x = SubpixelConv2D(in_tensor.shape, scale=scalar)(x)
      
        return fea_map
        
    
    inputs = Input(shape=lr_shape)
    ## < Feature extractor structure/ > 
    out_skip = in_skip = x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same', \
                                    activation='selu', kernel_initializer='lecun_uniform')(inputs)
    
    for _ in range(num_of_RRDRB):
        ## Residual in Residual Dense Block : 
        for _ in range(num_of_DRB):
            ## Residual Dense Blocks with channel attention block :
            x = SWCA_Blk(x, num_of_filts, resScal)
            
                
        ## out block process : (scaling and add)
        x = Lambda(lambda x: x * resScal)(x)
        x = in_skip = Add()([in_skip, x])
        
    x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same')(x)
    ## < /Feature extractor structure > 
    
    ## Source code of edsr do not contain residual scaling : 
    x = Lambda(lambda x: x * 0.8)(x)  ## residual scaling beta=0.8
    x = Add()([out_skip, x])
    
    for _ in range(upScalar//2):
        x = cv_upsampling(x, num_of_filts, scalar=2)
        #x = subpix_upsampling(x, num_of_filts, scalar=2)
        
    
    
    x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same', \
               activation='selu', kernel_initializer='lecun_uniform')(x)
    
    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)
    
    return Model(inputs=inputs, outputs=x, name='generator')
    