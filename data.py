from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from functools import reduce
from tools.utils import get_random_data,preprocess_true_boxes, get_anchors, expand_seg_label
from tools.modes import DATASET_MODE
from random import random
import tensorflow_datasets as tfds
import math
import matplotlib.pyplot as plt 
import sys

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset(tf.keras.callbacks.Callback):

    def parse_tfrecord(self, example_proto):
        feature_description = {
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/lane': tf.io.FixedLenFeature([], tf.string),
            'image/drive': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
        }
        features = tf.io.parse_single_example(example_proto,
                                              feature_description)
        filename = tf.compat.as_str_any(features['image/filename'])
        
        image = tf.image.decode_image(features['image/encoded'],
                                      channels=3,
                                      dtype=tf.float32)
        image.set_shape([None, None, 3])
        lane = expand_seg_label(tf.image.decode_image(features['image/lane'],
                                      channels=1,
                                      dtype=tf.uint8
                                      ),
                                      self.num_lane, 
                                      "lane")
        lane.set_shape([None, None, self.num_lane])
        
        drive = expand_seg_label(tf.image.decode_image(features['image/drive'],
                                      channels=1,
                                      dtype=tf.uint8
                                      ),
                                      self.num_drive,
                                      "drive") 
        drive.set_shape([None, None, self.num_drive])

        xmins = features['image/object/bbox/xmin'].values
        xmaxs = features['image/object/bbox/xmax'].values
        ymins = features['image/object/bbox/ymin'].values
        ymaxs = features['image/object/bbox/ymax'].values
        labels = features['image/object/bbox/label'].values
        image, lane_label, drive_label, bbox = get_random_data(image,
                                      lane,
                                      drive,
                                      xmins,
                                      xmaxs,
                                      ymins,
                                      ymaxs,
                                      labels,
                                      self.input_shape,
                                      train=self.mode == DATASET_MODE.TRAIN)
        
        y1, y2, y3 = tf.py_function(
            preprocess_true_boxes,
            [bbox, self.input_shape, self.anchors, self.num_classes],
            [tf.float32, tf.float32, tf.float32])
        y1.set_shape([None, None, len(self.anchors)//3, self.num_classes + 5])
        y2.set_shape([None, None, len(self.anchors)//3, self.num_classes + 5])
        y3.set_shape([None, None, len(self.anchors)//3, self.num_classes + 5])

        return image, (lane_label, drive_label, y1, y2, y3)

    def _dataset_internal(self,files,dataset_builder,parser):
        dataset = dataset_builder(files)
        if self.mode == DATASET_MODE.TRAIN:
            #train_num = reduce(
            #    lambda x, y: x + y,
            #    map(lambda file: int(self._get_num_from_name(file)), files))
            train_num = 90000
            dataset = dataset.shuffle(train_num).map(
                    parser, num_parallel_calls=AUTOTUNE).prefetch(
                        self.batch_size).batch(self.batch_size).repeat()
        elif self.mode == DATASET_MODE.VALIDATE:
            dataset = dataset.shuffle(10000).map(
                    parser, num_parallel_calls=AUTOTUNE).prefetch(
                        self.batch_size).batch(self.batch_size).repeat()
        elif self.mode == DATASET_MODE.TEST:
            dataset = dataset.map(
                    parser, num_parallel_calls=AUTOTUNE).prefetch(
                        self.batch_size).batch(self.batch_size)
        return dataset

    def __init__(self,
                 glob_path: str,
                 batch_size: int,
                 anchors=None,
                 num_lane=None,
                 num_drive=None,
                 num_classes=None,
                 input_shapes=None,
                 mode=DATASET_MODE.TRAIN):
        self.glob_path = glob_path
        self.batch_size = batch_size
        if isinstance(input_shapes, list):
            self.input_shapes = input_shapes
            self.input_shape = tf.Variable(name="input_shape",
                                           initial_value=self.input_shapes,
                                           trainable=False)
        else:
            self.input_shape = input_shapes
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_lane = num_lane
        self.num_drive = num_drive
        self.mode = mode

    def _get_num_from_name(self, name):
        return int(name.split('/')[-1].split('.')[0].split('_')[-3])

    def build(self,split=None):
        if self.glob_path in tfds.list_builders():
            return tfds.load(name=self.glob_path, split=split, with_info=True, as_supervised=True, try_gcs=tfds.is_dataset_on_gcs(self.glob_path))
        files = tf.io.gfile.glob(self.glob_path)
        if len(files)==0:
            raise ValueError('No file found')
        try:
            #TODO fix num
            num = reduce(lambda x, y: x + y,
                         map(lambda file: self._get_num_from_name(file), files))
            num = 100000
        except Exception:
            raise ValueError(
                'Please format file name like <name>_<number>.<extension>')
        else:
            tfrecords=list(filter(lambda file:file.endswith('.tfrecords'),files))
            txts=list(filter(lambda file: file.endswith('.txt'), files))
            if len(tfrecords)>0:
                tfrecords_dataset=self._dataset_internal(tfrecords,tf.data.TFRecordDataset, self.parse_tfrecord)
            #if len(txts) > 0:
            #    txts_dataset=self._dataset_internal(txts,tf.data.TextLineDataset,self.parse_text)
            if len(tfrecords)>0 and len(txts) > 0:
                return tfrecords_dataset.concatenate(txts_dataset),num
            elif len(tfrecords)>0:
                return tfrecords_dataset, num
            elif len(txts)>0:
                return txts_dataset, num


if __name__ == '__main__':
    """
    test code
    """
    anchors = get_anchors('config/yolo_anchors.txt')
    #print(anchors)
    #[[ 10.,  13.],
    #   [ 16.,  30.],
    #   [ 33.,  23.],
    #   [ 30.,  61.],
    #   [ 62.,  45.],
    #   [ 59., 119.],
    #   [116.,  90.],
    #   [156., 198.],
    #   [373., 326.]]
    dataset_callback = Dataset("data/train/*.tfrecords",
                                     8,
                                     anchors,
                                     11,
                                     [224,224],
                                     mode=DATASET_MODE.TEST)

   
    with tf.Session() as sess:
        dataset, num = dataset_callback.build()


        for n,image_features in dataset.enumerate():
            gt_image = sess.run(image_features[0].eval())
            gt_lane = sess.run(image_features[1].eval())
            gt_drive = sess.run(image_features[2].eval())
        
            print(gt_lane.shape)
            #gt_image = image_features[0].numpy()
            #gt_lane = image_features[1].numpy()
            #gt_drive = image_features[2].numpy()
            fig = plt.figure()
            fig.add_subplot(1,3,1)
            plt.imshow(gt_image)
            fig.add_subplot(1,3,2)
            plt.imshow(gt_lane)
            fig.add_subplot(1,3,3)
            plt.imshow(gt_drive)
            plt.show()
