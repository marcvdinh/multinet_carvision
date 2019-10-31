import re
import collections
import math
import numpy as np
import tensorflow as tf
from tools.utils import compose



def Conv2D( ch, kernel, strides=1):
    return compose(
            tf.keras.layers.Conv2D(ch,
                                kernel,
                                padding='same',
                                strides=strides,
                                use_bias=False),
            tf.keras.layers.BatchNormalization(fused=True), tf.keras.layers.ReLU(6.))

def StemBlock(Input):
    
    x = Conv2D(32, 3, 2)(Input)
    left = Conv2D(16, 1, 1)(x)
    left = Conv2D(32, 3, 2)(left)
    right = tf.keras.layers.MaxPool2D(strides=2)(x)

    merge = tf.keras.layers.concatenate([left,right])
    final = Conv2D(32, 1, 1)(merge)

    return final

def DenseLayer(Input, num_layers, growth_rate, bottleneck_width):
    growth_rate = growth_rate // 2
    for i in range(num_layers):
        inter_channel = int(growth_rate*bottleneck_width/4) * 4
        left = Conv2D(inter_channel, 1)(Input)
        left = Conv2D(growth_rate, 3)(left)

        right = Conv2D(inter_channel, 1)(Input)
        right = Conv2D(growth_rate, 3)(right)
        right = Conv2D(growth_rate, 3)(right)

        merge = tf.keras.layers.concatenate([left, right, Input])

    return  merge

def ResBlock(Input, out_filter):
    
    left = Conv2D(out_filter//4,1,1)(Input)
    left = Conv2D(out_filter//2,3,1)(left)
    left = Conv2D(out_filter//2,1,1)(left)

    right = Conv2D(out_filter//2,1,1)(Input)

    added = tf.keras.layers.add([left,right])

    return  added


def PeleeNet(Input, classes=1000, last_layer=True):
    n_dense_layers = [3,4,8,6]
    bottleneck_width = [1,2,4,4]
    out_layers = [128,256,512,704]
    growth_rate = 32

    #Input = tf.keras.Input(input_shape)
    #Stage 0
    x = StemBlock(Input)
    #Stage 1
    x = DenseLayer(x,n_dense_layers[0],growth_rate,bottleneck_width[0])
    x = Conv2D(out_layers[0],1)(x)
    x = tf.keras.layers.AveragePooling2D(strides=2, name="stage_1")(x)
    #Stage 2
    x = DenseLayer(x,n_dense_layers[1],growth_rate,bottleneck_width[1])
    x = Conv2D(out_layers[1],1)(x)
    x = tf.keras.layers.AveragePooling2D(strides=2,name="stage_2")(x)
    #Stage 3
    x = DenseLayer(x,n_dense_layers[2],growth_rate,bottleneck_width[2])
    x = Conv2D(out_layers[2],1)(x)
    x = tf.keras.layers.AveragePooling2D(strides=2, name="stage_3")(x)
    #Stage 4
    x = DenseLayer(x,n_dense_layers[3],growth_rate,bottleneck_width[3])
    x = Conv2D(out_layers[3],1)(x)
    

    if last_layer:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(classes, activation = "softmax")(x)
    
    return tf.keras.Model(Input, x)


if __name__ == '__main__':
    """
    test code
    """
    #backbone = BACKBONE.MOBILENETV2
    #test_in_tensor = tf.keras.backend.placeholder(dtype=tf.float32, shape=(1, 224, 224, 3), name='input')
    model = PeleeNet(tf.keras.Input([224,224,3]), False)
    model.summary()