"""YOLO_v3 Model Defined in Keras."""

import collections
import tensorflow as tf
from typing import List, Tuple
from utils import compose
from override import mobilenet_v2


class CarNet:
    def __init__(self,shape, n_class,n_anchors, n_lane_embedding, n_drive_embedding, alpha):
        self.shape = shape
        self.n_class = n_class
        self.n_anchors = n_anchors
        self.n_lane_embedding = n_lane_embedding
        self.n_drive_embedding = n_drive_embedding
        self._net_intermediate_results = collections.OrderedDict()
        self._alpha = alpha
    
    def MobilenetSeparableConv2D(self,filters,
                                kernel_size,
                                strides=(1, 1),
                                padding='valid',
                                use_bias=True):
        return compose(
            tf.keras.layers.DepthwiseConv2D(kernel_size,
                                            padding=padding,
                                            use_bias=use_bias,
                                            strides=strides),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.),
            tf.keras.layers.Conv2D(filters,
                                1,
                                padding='same',
                                use_bias=use_bias,
                                strides=1), tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(6.))


    def make_last_layers_mobilenet(self,x, id, num_filters, out_filters):
        x = compose(
            tf.keras.layers.Conv2D(num_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_' + str(id) + '_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9,
                                            name='block_' + str(id) + '_BN'),
            tf.keras.layers.ReLU(6., name='block_' + str(id) + '_relu6'),
            self.MobilenetSeparableConv2D(2 * num_filters,
                                    kernel_size=(3, 3),
                                    use_bias=False,
                                    padding='same'),
            tf.keras.layers.Conv2D(num_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_' + str(id + 1) + '_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9,
                                            name='block_' + str(id + 1) + '_BN'),
            tf.keras.layers.ReLU(6., name='block_' + str(id + 1) + '_relu6'),
            self.MobilenetSeparableConv2D(2 * num_filters,
                                    kernel_size=(3, 3),
                                    use_bias=False,
                                    padding='same'),
            tf.keras.layers.Conv2D(num_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_' + str(id + 2) + '_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9,
                                            name='block_' + str(id + 2) + '_BN'),
            tf.keras.layers.ReLU(6., name='block_' + str(id + 2) + '_relu6'))(x)
        y = compose(
            self.MobilenetSeparableConv2D(2 * num_filters,
                                    kernel_size=(3, 3),
                                    use_bias=False,
                                    padding='same'),
            tf.keras.layers.Conv2D(out_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False))(x)
        return x, y

    def _make_divisible(self,v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def _identity(self,x,**kw):
        return tf.keras.layers.Lambda(lambda x: x, **kw)

    def MobilenetConv2D(self,kernel, alpha, filters):
        last_block_filters = self._make_divisible(filters * alpha, 8)
        return compose(
            tf.keras.layers.Conv2D(last_block_filters,
                                kernel,
                                padding='same',
                                use_bias=False),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.))

    def build_encoder(self,inputs, alpha=1.0):
        mobilenetv2 = mobilenet_v2(default_batchnorm_momentum=0.9,
                                alpha=alpha,
                                input_tensor=inputs,
                                include_top=False,
                                weights='imagenet')
        #encoder_output = mobilenetv2.output
        #residual_block5 = mobilenetv2.get_layer('block_5_project_BN').output
        #residual_block12 = mobilenetv2.get_layer('block_12_project_BN').output

    
        return  mobilenetv2

    def build_lane_detection(self, inputs,residual, n_seg_class=None, alpha=1.0, upsample_output=True, last_layer_name=None):       
            if n_seg_class is None:
                n_seg_class=self.n_lane_embedding
            with tf.name_scope("lane_seg"):

            # 1x1 conv
                x_up = tf.keras.layers.Conv2D(128, (1, 1), padding='same',
                            use_bias=False)(inputs)
                x_up = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x_up)
                x_up = tf.keras.layers.Activation('relu')(x_up)

                # avg pool
                # TODO: AvgPool2D with such as large value, in effect, result in 1x1 value...
                # x_mid = AveragePooling2D((49, 49), strides=(16, 20))(middle)
                x_mid = tf.keras.layers.GlobalAveragePooling2D()(inputs)
                x_mid = tf.keras.layers.Reshape((1, 1, tf.keras.backend.int_shape(x_mid)[-1]))(x_mid)
                x_mid = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(x_mid)
                x_mid = tf.keras.layers.Activation('sigmoid')(x_mid)
                #x_mid = tf.keras.layers.ResizeImages(output_dim=tf.keras.backend.int_shape(inputs)[1:3])(x_mid)
                x_mid = tf.image.resize_images(x_mid,size=tf.keras.backend.int_shape(inputs)[1:3])

                # skip conn
                x_lo = tf.keras.layers.Conv2D(n_seg_class, (1, 1), padding='same')(residual)

                
                # merge up and mid
                x_up_mid_merged = tf.keras.layers.Multiply()([x_up, x_mid])
                x_up_mid_merged = tf.keras.layers.UpSampling2D(size=2)(x_up_mid_merged)
                x_up_mid_merged = tf.keras.layers.Conv2D(n_seg_class, (1, 1),
                                        padding='same')(x_up_mid_merged)

                # merge up_and_mid and lo
                x_final = tf.keras.layers.Add()([x_up_mid_merged, x_lo])
                x_final = tf.keras.layers.Activation('sigmoid')(x_final)
                # TODO:
                if upsample_output:
                    x_final = tf.keras.layers.UpSampling2D(size=8)(x_final)

                if last_layer_name:
                    x_final = self._identity(x_final, name=last_layer_name)

                return x_final

    def build_drivable_detection(self, inputs, residual, n_seg_class=None, upsample_output=True,alpha=1.0, last_layer_name=None):
            
            
            if n_seg_class is None:
                n_seg_class = self.n_drive_embedding   
            with tf.name_scope("drive_seg"): 
            # 1x1 conv
                x_up = tf.keras.layers.Conv2D(128, (1, 1), padding='same',
                            use_bias=False)(inputs)
                x_up = tf.keras.layers.BatchNormalization( epsilon=1e-5)(x_up)
                x_up = tf.keras.layers.Activation('relu')(x_up)

                # avg pool
                # TODO: AvgPool2D with such as large value, in effect, result in 1x1 value...
                # x_mid = AveragePooling2D((49, 49), strides=(16, 20))(middle)
                x_mid = tf.keras.layers.GlobalAveragePooling2D()(inputs)
                x_mid = tf.keras.layers.Reshape((1, 1, tf.keras.backend.int_shape(x_mid)[-1]))(x_mid)
                x_mid = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(x_mid)
                x_mid = tf.keras.layers.Activation('sigmoid')(x_mid)
                #x_mid = tf.keras.layers.UpSampling2D(output_dim=tf.keras.backend.int_shape(inputs)[1:3])(x_mid)
                x_mid = tf.image.resize_images(x_mid,size=tf.keras.backend.int_shape(inputs)[1:3])

                # skip conn
                x_lo = tf.keras.layers.Conv2D(n_seg_class, (1, 1), padding='same')(residual)

                # merge up and mid
                x_up_mid_merged = tf.keras.layers.Multiply()([x_up, x_mid])
                x_up_mid_merged = tf.keras.layers.UpSampling2D(size=2)(x_up_mid_merged)
                x_up_mid_merged = tf.keras.layers.Conv2D(n_seg_class, (1, 1),
                                        padding='same')(x_up_mid_merged)

                # merge up_and_mid and lo
                x_final = tf.keras.layers.Add()([x_up_mid_merged, x_lo])
                x_final = tf.keras.layers.Activation('sigmoid')(x_final)
                # TODO:
                if upsample_output:
                    x_final = tf.keras.layers.UpSampling2D(size=8)(x_final)

                if last_layer_name:
                    x_final = self._identity(x_final, name=last_layer_name)

                return x_final

    def build_yolo_body(self,inputs, residual_block5, residual_block12,num_anchors, num_classes, alpha=1.0):

        x, y1 = self.make_last_layers_mobilenet(inputs, 17, 512,
                                        num_anchors * (num_classes + 5))
        x = compose(
            tf.keras.layers.Conv2D(256,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_20_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
            tf.keras.layers.ReLU(6., name='block_20_relu6'),
            tf.keras.layers.UpSampling2D(2))(x)
        x = tf.keras.layers.Concatenate()([
            x,
            self.MobilenetConv2D(
                (1, 1), alpha,
                384)(residual_block12) #block12
        ])
        x, y2 = self.make_last_layers_mobilenet(x, 21, 256,
                                        num_anchors * (num_classes + 5))
        x = compose(
            tf.keras.layers.Conv2D(128,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_24_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9, name='block_24_BN'),
            tf.keras.layers.ReLU(6., name='block_24_relu6'),
            tf.keras.layers.UpSampling2D(2))(x)
        x = tf.keras.layers.Concatenate()([
            x,
            self.MobilenetConv2D((1, 1), alpha,
                            128)(residual_block5) #block5
        ])
        x, y3 = self.make_last_layers_mobilenet(x, 25, 128,
                                        num_anchors * (num_classes + 5))
        y1=tf.keras.layers.Lambda(lambda y: tf.reshape(y,[-1,tf.shape(y)[1],tf.shape(y)[2],num_anchors,num_classes + 5]), name='y1')(y1)
        y2=tf.keras.layers.Lambda(lambda y: tf.reshape(y,[-1,tf.shape(y)[1], tf.shape(y)[2], num_anchors, num_classes + 5]), name='y2')(y2)
        y3=tf.keras.layers.Lambda(lambda y: tf.reshape(y,[-1,tf.shape(y)[1], tf.shape(y)[2], num_anchors, num_classes + 5]), name='y3')(y3)
        return [y1, y2, y3]
    def build_tiny_yolo_body(self, inputs, num_anchors, num_classes):
        '''Create Tiny YOLO_v3 model CNN body in keras.'''
        x1 = compose(
                tf.keras.layers.Conv2D(16,
                                kernel_size=3,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6., name='block_20_relu6'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                tf.keras.layers.Conv2D(32,
                                kernel_size=3,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                tf.keras.layers.Conv2D(64,
                                kernel_size=3,
                                padding='same',
                                use_bias=False
                                ),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                tf.keras.layers.Conv2D(128,
                                kernel_size=3,
                                padding='same',
                                use_bias=False
                                ),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                tf.keras.layers.Conv2D(256,
                                kernel_size=3,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.))(inputs)
        x2 = compose(
                tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                tf.keras.layers.Conv2D(512,
                                kernel_size=3,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
                tf.keras.layers.Conv2D(1024,
                                kernel_size=3,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.),
                tf.keras.layers.Conv2D(256,
                                kernel_size=1,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.))(x1)
        y1 = compose(
                tf.keras.layers.Conv2D(512,
                                kernel_size=3,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.),
                tf.keras.layers.Conv2D(num_anchors*(num_classes+5),
                                kernel_size=1,
                                padding='same'))(x2)

        x2 = compose(
                tf.keras.layers.Conv2D(128,
                                kernel_size=1,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.),
                tf.keras.layers.UpSampling2D(2))(x2)
        y2 = compose(
                tf.keras.layers.Concatenate(),
                tf.keras.layers.Conv2D(256,
                                kernel_size=3,
                                padding='same',
                                use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
                tf.keras.layers.ReLU(6.),
                tf.keras.layers.Conv2D(num_anchors*(num_classes+5),
                                kernel_size=1,
                                padding='same'))([x2,x1])

        return  [y1,y2]

    def build(self,inputs=None, freeze_layers=None):
        if inputs is None:
            inputs = tf.keras.layers.Input(shape=self.shape)

        encoder = self.build_encoder(inputs, alpha=self._alpha)
        encoder_output = encoder.output
        
        residual_block12 = encoder.get_layer('block_12_project_BN').output
        residual_block5 = encoder.get_layer('block_5_project_BN').output
        segmentation_head = residual_block12 #output stride 16 with block 5, output stride 8 with block 5
        #print(segmentation_head.get_shape().as_list)
        lane_seg_decoder = self.build_lane_detection(segmentation_head, residual_block5, alpha=self._alpha)
        drive_seg_decoder =  self.build_drivable_detection(segmentation_head, residual_block5, alpha=self._alpha) 
        yolo_decoder = self.build_yolo_body(  encoder_output, residual_block5, residual_block12,self.n_anchors, self.n_class, alpha=self._alpha)
        #tiny_yolo_decoder = self.build_tiny_yolo_body(encoder_output, self.n_anchors, self.n_class)
         
        #model = tf.keras.Model(inputs, encoder.output)
        model = tf.keras.Model(inputs, [lane_seg_decoder, drive_seg_decoder, yolo_decoder])
        
        # Freeze the encoder.
        freeze_layers = min(freeze_layers, 155)
        for i in range(freeze_layers):
            encoder.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(
            freeze_layers, len(encoder.layers)))
        return model
        


if __name__ == '__main__':
    """
    test code
    """
    test_in_tensor = tf.keras.backend.placeholder(dtype=tf.float32, shape=(1, 224, 224, 3), name='input')
    model = CarNet(shape=None,n_class=11,n_anchors=7, n_lane_embedding=4, n_drive_embedding=4, alpha=1.4)
    ret = model.build(inputs=test_in_tensor, freeze_layers=155)
    tf.keras.utils.plot_model(
        ret,
        to_file='model.png',
        show_shapes=False,
        show_layer_names=True,
        rankdir='LR'
                )

    ret.summary()