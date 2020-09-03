from __future__ import division, print_function

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense
from keras.layers import MaxPooling2D, Cropping2D, concatenate, UpSampling2D
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

def conv_block_1 (x_in, conv2Dfilters, padding='same', maxpool2Dsize=(2,2), 
                    dropout=0.25, kernel_initializer="he_normal") :
    x = Conv2D(conv2Dfilters, kernel_size=(3,3), activation='relu', padding=padding, 
               kernel_initializer=kernel_initializer) (x_in)
    x = Conv2D(conv2Dfilters, kernel_size=(3, 3), activation='relu', padding=padding, 
               kernel_initializer=kernel_initializer) (x)
    x_inter = BatchNormalization() (x)
    x = MaxPooling2D(maxpool2Dsize) (x_inter) 
    x = Dropout(dropout)(x) if dropout>0 else x
    
    return x, x_inter


def conv_block_2 (x_in, conv2Dfilters, padding='same', kernel_initializer="he_normal") :
    x = BatchNormalization() (x_in)
    x = Conv2D(conv2Dfilters, kernel_size=(3, 3), activation='relu', padding=padding, 
               kernel_initializer=kernel_initializer) (x)
    x = Conv2D(conv2Dfilters, kernel_size=(3, 3), activation='relu', padding=padding, 
               kernel_initializer=kernel_initializer) (x) 
    
    return x


def lung_seg(input_shape, num_filters=[16,32,128], padding='same') :
    """Generate CN-Net model to train on CT scan images for lung seg
    Arbitrary number of input channels and output classes are supported.

    Arguments:
      input_shape  - (? (number of examples), 
                      input image height (pixels), 
                      input image width  (pixels), 
                      input image features (1 for grayscale, 3 for RGB))
      num_filters - number of filters (exactly 4 should be passed)
      padding - 'same' or 'valid'

    Output:
      CN-Net model expecting input shape (height, width, channels) and generates
      output with the same shape (height, width, channels).
    """
    x_input = Input(input_shape)
    
    ###  LUNG SEGMENTATION
    x = Conv2D(num_filters[0], kernel_size=3, activation='relu', padding=padding)(x_input)
    x = MaxPooling2D(pool_size=2, padding=padding)(x)

    x = Conv2D(num_filters[1], kernel_size=3, activation='relu', padding=padding)(x)
    x = MaxPooling2D(pool_size=2, padding=padding)(x)

    x = Conv2D(num_filters[2], kernel_size=3, activation='relu', padding=padding)(x)
    x = MaxPooling2D(pool_size=2, padding=padding)(x)

    x = Dense(num_filters[2], activation='relu')(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(num_filters[2], kernel_size=3, activation='sigmoid', padding=padding)(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(num_filters[1], kernel_size=3, activation='sigmoid', padding=padding)(x)

    x = UpSampling2D(size=2)(x)
    lung_seg = Conv2D(1, kernel_size=3, activation='sigmoid', 
                        padding=padding)(x) # identifying lungs

    lseg = Model(inputs=x_input, outputs=lung_seg, name='lung_seg')
    
    return lseg


def infect_seg (input_shape, num_filters=[32,64,64,128], padding='same',
                dropout=0.2) :
    """Generate CN-Net model to train on CT scan images
    Arbitrary number of input channels and output classes are supported.
    Sizes are noted with 100x100 input images.

    Arguments:
      input_shape  - (? (number of examples), 
                      input image height (pixels), 
                      input image width  (pixels), 
                      input image features (1 for grayscale, 3 for RGB))
      num_filters - number of filters (exactly 4 should be passed)
      padding - 'same' or 'valid'
      dropout - fraction of units to dropout, 0 to keep all units

    Output:
      CN-Net model expecting input shape (height, width, channels) and generates
      two outputs with shape (height, width, channels).
    """
    assert len(num_filters) == 4

    x_input = Input(input_shape)
    
    x, x1 = conv_block_1 (x_input, num_filters[0], padding=padding, 
                        maxpool2Dsize=(2,2), dropout=dropout, 
                        kernel_initializer="he_normal") #x: 50x50

    x, x2 = conv_block_1 (x, num_filters[1], padding=padding, 
                        maxpool2Dsize=(2,2), dropout=dropout, 
                        kernel_initializer="he_normal") #x: 25x25
    x, _ = conv_block_1 (x, num_filters[2], padding=padding, 
                        maxpool2Dsize=(1,1), dropout=dropout, 
                        kernel_initializer="he_normal") #x: 25x25

    x, _ = conv_block_1 (x, num_filters[3], padding=padding, 
                        maxpool2Dsize=(1,1), dropout=dropout, 
                        kernel_initializer="he_normal") #x: 25x25
    
    x = conv_block_2 (x, num_filters[3], padding=padding,
                        kernel_initializer="he_normal") #x: 25x25
    
    x = Conv2DTranspose(num_filters[2], (2, 2), strides=(2,2), padding=padding) (x) #x: 50x50
    x = conv_block_2 (x, num_filters[2], padding=padding,
                        kernel_initializer="he_normal") #x: 50x50

    x = Conv2DTranspose(num_filters[1], (2, 2), padding='same') (x) #x: 50x50
    x = concatenate([x, x2]) #x: 50x50
    x = conv_block_2 (x, num_filters[1], padding=padding, 
                        kernel_initializer="he_normal") #x: 50x50

    x = Conv2DTranspose(num_filters[0], (2, 2), strides=(2,2), padding=padding) (x) #x: 100x100
    x = concatenate([x, x1], axis=3) #x: 100x100
    x = conv_block_2 (x, num_filters[0], padding=padding, 
                        kernel_initializer="he_normal") #x: 100x100

    infect_seg = Conv2D(1, (1, 1), activation='sigmoid', name='infect_output') (x) # identifying infections

    iseg = Model(inputs=x_input, outputs=infect_seg, name='cts_model')
    
    return iseg
