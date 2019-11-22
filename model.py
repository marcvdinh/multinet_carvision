"""YOLO_v3 Model Defined in Keras."""

import collections
import tensorflow as tf
from typing import List, Tuple
from tools.utils import compose
from encoder_zoo.override import mobilenet_v2
from encoder_zoo.efficientnet import EfficientNetB4, MBConvBlock, get_model_params, BlockArgs, EfficientConv2DKernelInitializer
from encoder_zoo.pelee import PeleeNet, ResBlock, DenseLayer, Conv2D
from tools.modes import OPT, BACKBONE
class CarNet:
    def __init__(self,backbone,inputs=tf.keras.layers.Input(shape=(None, None, 3)),weights_path=None, n_class=11,n_anchors=None, n_lane_embedding=None, n_drive_embedding=None, alpha=1.0):
        self.backbone =  backbone
        self.inputs = inputs
        self.weights_path = weights_path
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
            tf.keras.layers.BatchNormalization(fused=True), tf.keras.layers.ReLU(6.),
            tf.keras.layers.Conv2D(filters,
                                1,
                                padding='same',
                                use_bias=use_bias,
                                strides=1), tf.keras.layers.BatchNormalization(fused=True),
            tf.keras.layers.ReLU(6.))


    def make_last_layers_mobilenet(self,x, id, num_filters, out_filters):
        x = compose(
            tf.keras.layers.Conv2D(num_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_' + str(id) + '_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9, fused=True,
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
            tf.keras.layers.BatchNormalization(momentum=0.9, fused=True,
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
            tf.keras.layers.BatchNormalization(momentum=0.9, fused=True,
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
            tf.keras.layers.BatchNormalization(fused=True), tf.keras.layers.ReLU(6.))

    def build_encoder(self,inputs, alpha=1.0):
        #inputs_shape = inputs.shape[1::]
        
        if self.backbone == BACKBONE.MOBILENETV2:
            encoder = mobilenet_v2(default_batchnorm_momentum=0.9,
                                alpha=alpha,
                                input_tensor=inputs,
                                include_top=False,
                                weights='imagenet')
        elif self.backbone == BACKBONE.EFFICIENTNET:
            encoder = EfficientNetB4(include_top=False,
                                        weights='imagenet',
                                        classes=self.n_class,
                                        input_tensor=inputs)
        elif self.backbone == BACKBONE.PELEE:
            encoder = PeleeNet(inputs, last_layer=False)

    
        return  encoder

    def build_mobilenet_yolo(self,input_shape, resmid_shape, resfin_shape,num_anchors, num_classes, alpha=1.0):

        feat = tf.keras.Input(input_shape)
        res5 = tf.keras.Input(resmid_shape)
        res12 = tf.keras.Input(resfin_shape)

        x, y1 = self.make_last_layers_mobilenet(feat, 17, 512,
                                        num_anchors * (num_classes + 5))
        x = compose(
            tf.keras.layers.Conv2D(256,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_20_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9, fused=True, name='block_20_BN'),
            tf.keras.layers.ReLU(6., name='block_20_relu6'),
            tf.keras.layers.UpSampling2D(2))(x)
        x = tf.keras.layers.Concatenate()([
            x,
            self.MobilenetConv2D(
                (1, 1), alpha,
                384)(res12) #block12
        ])
        x, y2 = self.make_last_layers_mobilenet(x, 21, 256,
                                        num_anchors * (num_classes + 5))
        x = compose(
            tf.keras.layers.Conv2D(128,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_24_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9, fused=True, name='block_24_BN'),
            tf.keras.layers.ReLU(6., name='block_24_relu6'),
            tf.keras.layers.UpSampling2D(2))(x)
        x = tf.keras.layers.Concatenate()([
            x,
            self.MobilenetConv2D((1, 1), alpha,
                            128)(res5) #block5
        ])
        x, y3 = self.make_last_layers_mobilenet(x, 25, 128,
                                        num_anchors * (num_classes + 5))
        y1 = tf.keras.layers.Reshape(
            (y1.shape[1], y1.shape[2], num_anchors, num_classes + 5), name='y1')(y1)
        y2 = tf.keras.layers.Reshape(
            (y2.shape[1], y2.shape[2], num_anchors, num_classes + 5), name='y2')(y2)
        y3 = tf.keras.layers.Reshape(
            (y3.shape[1], y3.shape[2], num_anchors, num_classes + 5), name='y3')(y3)
        return tf.keras.Model([feat, res5, res12],[y1,y2,y3])
###############EFFICIENTNET###############
    def make_last_layers_efficientnet(self,x, block_args, global_params):
        if global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        num_filters = block_args.input_filters * block_args.expand_ratio
        x = compose(
            tf.keras.layers.Conv2D(num_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False),
            tf.keras.layers.BatchNormalization(
                axis=channel_axis,
                epsilon=global_params.batch_norm_epsilon,
                momentum=global_params.batch_norm_momentum),
            tf.keras.layers.ReLU(6.),
            MBConvBlock(block_args,
                        global_params,
                        drop_connect_rate=global_params.drop_connect_rate),
            tf.keras.layers.Conv2D(num_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False),
            tf.keras.layers.BatchNormalization(
                axis=channel_axis,
                epsilon=global_params.batch_norm_epsilon,
                momentum=global_params.batch_norm_momentum),
            tf.keras.layers.ReLU(6.),
            MBConvBlock(block_args,
                        global_params,
                        drop_connect_rate=global_params.drop_connect_rate),
            tf.keras.layers.Conv2D(num_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False),
            tf.keras.layers.BatchNormalization(
                axis=channel_axis,
                epsilon=global_params.batch_norm_epsilon,
                momentum=global_params.batch_norm_momentum),
            tf.keras.layers.ReLU(6.))(x)
        y = compose(
            MBConvBlock(block_args,
                        global_params,
                        drop_connect_rate=global_params.drop_connect_rate),
            tf.keras.layers.Conv2D(block_args.output_filters,
                                kernel_size=1,
                                padding='same',
                                use_bias=False))(x)
        return x, y

    def build_efficientnet_yolo(self,features_shape, res_mid_shape, res_end_shape,
                                     num_anchors, model_name='efficientnet-b4', **kwargs):
        _, global_params, input_shape = get_model_params(model_name, kwargs)
        num_classes = global_params.num_classes
        if global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        
        block_args = BlockArgs(kernel_size=3,
                            num_repeat=1,
                            input_filters=512,
                            output_filters=num_anchors * (num_classes + 5),
                            expand_ratio=1,
                            id_skip=True,
                            se_ratio=0.25,
                            strides=[1, 1])
        Input = tf.keras.Input(features_shape)
        swish_29 = tf.keras.Input(res_mid_shape)
        swish_65 = tf.keras.Input(res_end_shape)
        x, y1 = self.make_last_layers_efficientnet(Input, block_args,
                                            global_params)
        x = compose(
            tf.keras.layers.Conv2D(256,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_20_conv'),
            tf.keras.layers.BatchNormalization(axis=channel_axis,
                                            momentum=0.9,
                                            name='block_20_BN'),
            tf.keras.layers.ReLU(6., name='block_20_relu6'),
            tf.keras.layers.UpSampling2D(2))(x)
        block_args = block_args._replace(input_filters=256)
        x = tf.keras.layers.Concatenate()(
            [x, swish_65])
        x, y2 = self.make_last_layers_efficientnet(x, block_args, global_params)
        x = compose(
            tf.keras.layers.Conv2D(128,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                name='block_24_conv'),
            tf.keras.layers.BatchNormalization(axis=channel_axis,
                                            momentum=0.9,
                                            name='block_24_BN'),
            tf.keras.layers.ReLU(6., name='block_24_relu6'),
            tf.keras.layers.UpSampling2D(2))(x)
        block_args = block_args._replace(input_filters=128)
        x = tf.keras.layers.Concatenate()(
            [x, swish_29])
        x, y3 = self.make_last_layers_efficientnet(x, block_args, global_params)
        y1 = tf.keras.layers.Reshape(
            (y1.shape[1], y1.shape[2], num_anchors, num_classes + 5), name='y1')(y1)
        y2 = tf.keras.layers.Reshape(
            (y2.shape[1], y2.shape[2], num_anchors, num_classes + 5), name='y2')(y2)
        y3 = tf.keras.layers.Reshape(
            (y3.shape[1], y3.shape[2], num_anchors, num_classes + 5), name='y3')(y3)
        return tf.keras.Model([Input, swish_29, swish_65],[y1, y2, y3])

#####################PELEENET############################################
    def make_last_layers_pelee(self, Input, num_filters, out_filters):
        x= ResBlock(Input,num_filters)

        y = DenseLayer(x,1,32,2)
        y = Conv2D(out_filters,1)(y)
        #y = tf.keras.layers.AveragePooling2D(strides=2)(y)

        return x,y

    def build_pelee_yolo(self,features_shape, res_mid_shape, res_end_shape,
                                     num_anchors, num_classes):

        feat = tf.keras.Input(features_shape)
        stage1 = tf.keras.Input(res_mid_shape)
        stage2 = tf.keras.Input(res_end_shape)

        x,y1 = self.make_last_layers_pelee(feat, 512,num_anchors * (num_classes + 5))
        x = compose(
            tf.keras.layers.Conv2D(256,
                                kernel_size=1,
                                padding='same',
                                use_bias=False
                                ),
            tf.keras.layers.BatchNormalization(fused=True,
                                            momentum=0.9
                                           ),
            tf.keras.layers.ReLU(6.),
            tf.keras.layers.UpSampling2D(2))(x)
        x = tf.keras.layers.Concatenate()(
            [x, stage2])

        x,y2 = self.make_last_layers_pelee(x, 256,num_anchors * (num_classes + 5))

        x = compose(
            tf.keras.layers.Conv2D(128,
                                kernel_size=1,
                                padding='same',
                                use_bias=False),
            tf.keras.layers.BatchNormalization(fused=True,
                                            momentum=0.9),
            tf.keras.layers.ReLU(6.),
            tf.keras.layers.UpSampling2D(2))(x)
        x = tf.keras.layers.Concatenate()(
            [x, stage1])

        x,y3 = self.make_last_layers_pelee(x, 128,num_anchors * (num_classes + 5))

        #y1=tf.keras.layers.Lambda(lambda y: tf.reshape(y,[-1,tf.shape(y)[1],tf.shape(y)[2],num_anchors,num_classes + 5]), name='y1')(y1)
        #y2=tf.keras.layers.Lambda(lambda y: tf.reshape(y,[-1,tf.shape(y)[1], tf.shape(y)[2], num_anchors, num_classes + 5]), name='y2')(y2)
        #y3=tf.keras.layers.Lambda(lambda y: tf.reshape(y,[-1,tf.shape(y)[1], tf.shape(y)[2], num_anchors, num_classes + 5]), name='y3')(y3)

        y1 = tf.keras.layers.Reshape(
            (y1.shape[1], y1.shape[2], num_anchors, num_classes + 5), name='y1')(y1)
        y2 = tf.keras.layers.Reshape(
            (y2.shape[1], y2.shape[2], num_anchors, num_classes + 5), name='y2')(y2)
        y3 = tf.keras.layers.Reshape(
            (y3.shape[1], y3.shape[2], num_anchors, num_classes + 5), name='y3')(y3)

        return tf.keras.Model([feat, stage1, stage2],[y1,y2,y3])
        

################SEGMENTATION HEAD########################################
    def build_lane_detection(self, inputs,residual, n_seg_class=None, alpha=1.0, upsample_output=True, last_layer_name=None):       
            if n_seg_class is None:
                n_seg_class=self.n_lane_embedding
            inputs = tf.keras.layers.Conv2D(256, (3,3), padding="same", dilation_rate=2)(inputs)
            with tf.name_scope("lane_seg"):  
            # 1x1 conv
                x_up = tf.keras.layers.Conv2D(128, (1, 1), padding='same',
                            use_bias=False)(inputs)
                x_up = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(x_up)
                x_up = tf.keras.layers.Activation('relu')(x_up)
                size = (x_up.shape[1], x_up.shape[2])
                # avg pool
                # TODO: AvgPool2D with such as large value, in effect, result in 1x1 value...
                x_mid = tf.keras.layers.AveragePooling2D((49, 49), strides=(16, 20), padding="same")(inputs)
                #x_mid = tf.keras.layers.GlobalAveragePooling2D()(residual)
                #x_mid = tf.keras.layers.Reshape((1, 1, tf.keras.backend.int_shape(x_mid)[-1]))(x_mid)
                x_mid = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(x_mid)
                x_mid = tf.keras.layers.Activation('sigmoid')(x_mid)
                x_mid = tf.keras.layers.UpSampling2D(size=size, interpolation="bilinear")(x_mid)
                #x_mid = tf.image.resize_images(x_mid,size=tf.keras.backend.int_shape(inputs)[1:3])

                # skip conn
                x_lo = tf.keras.layers.Conv2D(n_seg_class, (1, 1), padding='same')(residual)

                
                # merge up and mid
                x_up_mid_merged = tf.keras.layers.Multiply()([x_up, x_mid])
                x_up_mid_merged = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")(x_up_mid_merged)
                x_up_mid_merged = tf.keras.layers.Conv2D(n_seg_class, (1, 1),
                                        padding='same')(x_up_mid_merged)

                # merge up_and_mid and lo
                x_final = tf.keras.layers.Add()([x_up_mid_merged, x_lo])
                x_final = tf.keras.layers.Activation('sigmoid')(x_final)
                # TODO:
                if upsample_output:
                    lane_output = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear", name="lane_seg")(x_final)

                if last_layer_name:
                    x_final = self._identity(x_final, name=last_layer_name)

                return lane_output

    def build_drivable_detection(self, input_shape, res_shape,n_seg_class=None, upsample_output=True,alpha=1.0, last_layer_name=None):     
            if n_seg_class is None:
                n_seg_class = self.n_drive_embedding  
            feat = tf.keras.Input(input_shape)
            res = tf.keras.Input(res_shape)
            inputs = tf.keras.layers.Conv2D(256, (3,3), padding="same", dilation_rate=2)(feat)
            
            with tf.name_scope("drive_seg"): 
            # 1x1 conv
                x_up = tf.keras.layers.Conv2D(128, (1, 1), padding='same',
                            use_bias=False)(inputs)
                x_up = tf.keras.layers.BatchNormalization( epsilon=1e-5, fused=True)(x_up)
                x_up = tf.keras.layers.Activation('relu')(x_up)
                size = (x_up.shape[1], x_up.shape[2])
                # avg pool
                # TODO: AvgPool2D with such as large value, in effect, result in 1x1 value...
                x_mid = tf.keras.layers.AveragePooling2D((49, 49), strides=(16, 20), padding="same")(inputs)
                #x_mid = tf.keras.layers.GlobalAveragePooling2D()(inputs)
                #x_mid = tf.keras.layers.Reshape((1, 1, tf.keras.backend.int_shape(x_mid)[-1]))(x_mid)
                x_mid = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(x_mid)
                x_mid = tf.keras.layers.Activation('sigmoid')(x_mid)
                x_mid = tf.keras.layers.UpSampling2D(size=size, interpolation="bilinear")(x_mid)
                #x_mid = tf.image.resize_images(x_mid,size=tf.keras.backend.int_shape(inputs)[1:3])

                # skip conn
                x_lo = tf.keras.layers.Conv2D(n_seg_class, (1, 1), padding='same')(res)

                # merge up and mid
                x_up_mid_merged = tf.keras.layers.Multiply()([x_up, x_mid])
                x_up_mid_merged = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")(x_up_mid_merged)
                x_up_mid_merged = tf.keras.layers.Conv2D(n_seg_class, (1, 1),
                                        padding='same')(x_up_mid_merged)

                # merge up_and_mid and lo
                x_final = tf.keras.layers.Add()([x_up_mid_merged, x_lo])
                x_final = tf.keras.layers.Activation('sigmoid')(x_final)
                # TODO:
                if upsample_output:
                    drive_output = tf.keras.layers.UpSampling2D(size=8,interpolation="bilinear", name="drive_seg")(x_final)

                if last_layer_name:
                    x_final = self._identity(x_final, name=last_layer_name)

                return tf.keras.Model([feat,res], drive_output)

    def build(self,inputs=None, freeze_layers=None):
        if inputs is None:
            inputs = self.inputs
    
        encoder = self.build_encoder(inputs, alpha=self._alpha)
        encoder_output = encoder.output
        features_shape = encoder_output.shape[1:]
        if self.backbone == BACKBONE.MOBILENETV2:
            residual_end = encoder.get_layer('block_12_project_BN').output
            residual_middle = encoder.get_layer('block_5_project_BN').output
            res_end_shape = residual_end.shape[1:]
            res_mid_shape = residual_middle.shape[1:]
            yolo_decoder = self.build_mobilenet_yolo( features_shape, res_mid_shape, res_end_shape,self.n_anchors, self.n_class, alpha=self._alpha)
        elif self.backbone == BACKBONE.EFFICIENTNET:
        #TODO add members/setters for efficientnet hyperparameters instead of hardcoded values        
            residual_end = encoder.get_layer('swish_65').output
            residual_middle = encoder.get_layer('swish_29').output
            res_end_shape = residual_end.shape[1:]
            res_mid_shape = residual_middle.shape[1:]
            yolo_decoder = self.build_efficientnet_yolo( features_shape, res_mid_shape, res_end_shape,self.n_anchors,"efficientnet-b4", batch_norm_momentum=0.9,
                                    batch_norm_epsilon=1e-3,
                                    num_classes=self.n_class,
                                    drop_connect_rate=0.2,
                                    data_format="channels_first")
        elif self.backbone == BACKBONE.PELEE:
        #TODO add members/setters for efficientnet hyperparameters instead of hardcoded values    
            residual_end = encoder.get_layer('stage_2').output
            residual_middle = encoder.get_layer('stage_1').output
            res_end_shape = residual_end.shape[1:]
            res_mid_shape = residual_middle.shape[1:]
            yolo_decoder = self.build_pelee_yolo(  features_shape, res_mid_shape, res_end_shape,self.n_anchors, self.n_class )
        #segmentation_head = residual_block12 #output stride 16 with block 5, output stride 8 with block 12
        
        #lane_seg_decoder = self.build_lane_detection(encoder_output, residual_middle, alpha=self._alpha)
        drive_seg_decoder =  self.build_drivable_detection(features_shape, res_mid_shape, alpha=self._alpha) 
       
        #TODO implement tiny yolo as a detector head
        detections = yolo_decoder([encoder_output, residual_middle, residual_end])
        drivable_map = drive_seg_decoder([encoder_output, residual_middle])
        model = tf.keras.Model(inputs, [drivable_map, detections])
        #model = tf.keras.Model(inputs, encoder.output)
        # Freeze the encoder.
        for i in range(len(encoder.layers)):
            encoder.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(
                freeze_layers, len(model.layers)))
        return model
        


if __name__ == '__main__':
    """
    test code
    """

    backbone = BACKBONE.PELEE
    test_in_tensor = tf.keras.Input([224,224,3])
    model = CarNet(inputs=None,backbone=backbone,n_class=11,n_anchors=7, n_lane_embedding=5, n_drive_embedding=3, alpha=1.4)
    ret = model.build(inputs=test_in_tensor, freeze_layers=155)
    #for layers in ret.yolo_decoder:
    #    layers.trainable = False
    #ret.layers[-1].trainable = False
    for layer in ret.layers[:]:
        print(layer.trainable)
    #tf.keras.utils.plot_model(
    #    ret,
    #    to_file='mobilenet_model.png',
    #    show_shapes=False,
    #    show_layer_names=True,
    #    rankdir="TB"
    #            )

    ret.summary()