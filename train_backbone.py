import tensorflow as tf
import pathlib
import numpy as np
from data import Dataset
from encoder_zoo.override import mobilenet_v2
#from yolo3.darknet import darknet_body
from encoder_zoo.efficientnet import EfficientNetB4
from encoder_zoo.pelee import PeleeNet as pelee
from tools.utils import get_classes, ModelFactory
from tools.modes import BACKBONE
import os
import datetime

AUTOTUNE = tf.data.experimental.AUTOTUNE

class CaltechDataset():
    """Backbone's Dataset extends Dataset,only support txt files now.
    """
    def __init__(self,dataset_glob, input_shape, class_names, batch_size):
        dataset_glob = pathlib.Path(dataset_glob)
        self.CLASS_NAMES = np.array([item.name for item in dataset_glob.glob('*') if item.name != "License.txt"])
        self.IMG_WIDTH = input_shape[1]
        self.IMG_HEIGHT = input_shape[0]
        self.batch_size = batch_size
        self.dataset_glob = str(dataset_glob/'*/*')
    def get_label(self,file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, '/')
        # The second to last is the class-directory
        return tf.cast(tf.equal(parts[-2], self.CLASS_NAMES), tf.float32)
    
    def decode_img(self,img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self,file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label


    def build(self):
        
        list_ds = tf.data.Dataset.list_files(self.dataset_glob)
        ds = list_ds.cache()
        ds = list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE).shuffle(10000).batch(self.batch_size).prefetch(AUTOTUNE).repeat()

        return ds

class VOCDataset(Dataset):
    def parse_tfrecord(self, example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
        }
        features = tf.io.parse_single_example(example_proto,
                                              feature_description)
        image = tf.image.decode_image(features['image/encoded'],
                                      channels=3,
                                      dtype=tf.float32)
        image.set_shape([None, None, 3])
        image =  tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.input_shape)
        labels = tf.one_hot(features['image/object/bbox/label'].values, self.num_classes)
        
        return image, labels


def mobilenetv2(inputs, alpha, classes):
    """MobilenetV2 wrapper function
    
    Arguments:
        inputs {np.array} -- [train images]
        alpha {float} -- [controls the width of the network. This is known as the
        width multiplier in the MobileNetV2 paper.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.]
        classes {int} -- [classes total number]
    
    Returns:
        [tf.keras.Model] -- [mobilenetv2 model]
    """
    return mobilenet_v2(default_batchnorm_momentum=0.9,
                        alpha=alpha,
                        include_top = True,
                        weights = None,
                        input_tensor=inputs,
                        classes=classes)


def EfficientNet(inputs, classes, input_shape):
    return EfficientNetB4(classes=classes,
                          input_shape=input_shape,
                          input_tensor=inputs)

def PeleeNet(inputs, classes, input_shape):
    return pelee(inputs, classes, True, False)

def train(FLAGS):
    batch_size = FLAGS['batch_size']
    #use_tpu = FLAGS['use_tpu']
    class_names = get_classes(FLAGS['classes_path'])
    epochs = FLAGS['epochs']
    input_size = FLAGS['input_size']
    model_path = FLAGS['model']
    backbone = FLAGS['backbone']
    train_dataset_glob = FLAGS['train_dataset']
    val_dataset_glob = FLAGS['val_dataset']
    lr = FLAGS['learning_rate']
    log_dir = FLAGS['log_directory'] or os.path.join(
        'logs',
        str(backbone).split('.')[1].lower() + str(datetime.date.today()))
    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync
    Input = tf.keras.Input((input_size[0], input_size[1],3))
    with strategy.scope():
        if backbone == BACKBONE.MOBILENETV2:
            model = mobilenetv2(Input, 1.4, len(class_names))
        elif backbone == BACKBONE.EFFICIENTNET:
            model = EfficientNet(Input, len(class_names), input_size)
        elif backbone == BACKBONE.PELEE:
            model = PeleeNet(Input, len(class_names), input_size)
        model.compile(tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=[tf.keras.metrics.categorical_accuracy])
    #if use_tpu:
    #    tpu = tf.contrib.cluster_resolver.TPUClusterResolver()
    #    tpu_strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
    ##    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=tpu_strategy)

    #train_dataset = CaltechDataset(train_dataset_glob,
    #                                           batch_size = batch_size,
    #                                           class_names=class_names,
    #                                           input_shape=input_size).build()

    train_dataset, _= VOCDataset(train_dataset_glob,
                                               batch_size,
                                               num_classes=len(class_names),
                                               input_shapes=input_size).build()
    val_dataset, _ = VOCDataset(val_dataset_glob,
                                               batch_size,
                                               num_classes=len(class_names),
                                               input_shapes=input_size).build()
    cos_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, _: tf.keras.experimental.CosineDecay(lr[0], epochs[0])(epoch).numpy(),
        1)
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
        log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period=3)
    model.fit(train_dataset,
              epochs=epochs[0],
              initial_epoch=0,
              steps_per_epoch=max(1, 50000 // batch_size),
              validation_data = val_dataset,
              #validation_split=0.2,
              validation_steps=50000,
              callbacks=[cos_lr, logging, checkpoint])

    cos_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, _: tf.keras.experimental.CosineDecay(lr[1], epochs[1])(epoch - epochs[0]).numpy(),
        1)
    model.fit(train_dataset,
              epochs=epochs[0]+epochs[1],
              initial_epoch = epochs[0],
              steps_per_epoch=max(1, 50000 // batch_size),
              #validation_split=0.2,
              validation_data = val_dataset,
              validation_steps=50000,
              callbacks=[cos_lr, logging, checkpoint])
    #for pelee only
    headless = tf.keras.Model(Input, model.get_layer('re_lu_112').output)
    model.save_weights(
        os.path.join(
            log_dir,
            str(backbone).split('.')[1].lower() + '_trained_weights.h5'))
    headless.save_weights(
        os.path.join(
            log_dir,
            str(backbone).split('.')[1].lower() + 'headless_trained_weights.h5'))
