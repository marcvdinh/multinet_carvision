from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from functools import reduce
from tools.utils import get_random_data,preprocess_true_boxes, get_anchors, expand_seg_label
from tools.modes import DATASET_MODE
from random import random
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
        
        image = tf.io.decode_image(features['image/encoded'],
                                      channels=3,
                                      dtype=tf.float32)
        image.set_shape([None, None, 3])
        lane = expand_seg_label(tf.io.decode_image(features['image/lane'],
                                      channels=1,
                                      dtype=tf.uint8
                                      ),
                                      2,
                                      "lane")
        lane.set_shape([None, None, 2])
        
        drive = tf.io.decode_image(features['image/drive'],
                                      channels=1,
                                      dtype=tf.uint8
                                      )
        drive.set_shape([None, None, 1])

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

        return image, ( y1, y2, y3, drive_label, lane_label)

    def _dataset_internal(self,files,dataset_builder,parser):
        dataset = tf.data.Dataset.list_files(files)
        #dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_reads,
        #        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.mode == DATASET_MODE.TRAIN:
            #train_num = reduce(
            #    lambda x, y: x + y,
            #    map(lambda file: int(self._get_num_from_name(file)), files))
            num = 70000
            dataset = dataset.interleave(
                lambda file: dataset_builder(file),
                cycle_length=len(files),
                num_parallel_calls=AUTOTUNE).shuffle(num).map(
                    parser, num_parallel_calls=AUTOTUNE).repeat().batch(self.batch_size).prefetch(
                        AUTOTUNE)
        elif self.mode == DATASET_MODE.VALIDATE:
            num = 10000
            dataset = dataset.interleave(
                lambda file: dataset_builder(file),
                cycle_length=len(files),
                num_parallel_calls=AUTOTUNE).shuffle(num).map(
                    parser, num_parallel_calls=AUTOTUNE).repeat().batch(self.batch_size).prefetch(
                        AUTOTUNE)	
        elif self.mode == DATASET_MODE.TEST:
            num = 0
            dataset = dataset.map(
                    parser, num_parallel_calls=AUTOTUNE).batch(self.batch_size).prefetch(AUTOTUNE)
        return dataset, num

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
        files = tf.io.gfile.glob(self.glob_path)
        if len(files)==0:
            raise ValueError('No file found')
        else:
            tfrecords=list(filter(lambda file:file.endswith('.tfrecords'),files))
            txts=list(filter(lambda file: file.endswith('.txt'), files))
            if len(tfrecords)>0:
                tfrecords_dataset, num = self._dataset_internal(tfrecords,tf.data.TFRecordDataset, self.parse_tfrecord)
            #if len(txts) > 0:
            #    txts_dataset=self._dataset_internal(txts,tf.data.TextLineDataset,self.parse_text)
            if len(tfrecords)>0 and len(txts) > 0:
                return tfrecords_dataset.concatenate(txts_dataset),num
            elif len(tfrecords)>0:
                return tfrecords_dataset, num
            elif len(txts)>0:
                return txts_dataset, num


class LaneDataset():

    def __init__(self, src_dir,  batch_size,
                                 anchors,
                                  num_lane,
                                  num_drive,
                                  num_classes,
                                  input_shape, dummy_data):
        self.batch_size = batch_size
        self.input_size = input_shape
        self.src_dir = src_dir
        self.anchors = anchors
        self.num_classes = num_classes
        for input, label in dummy_data:
            self.y1 = label[0]
            self.y2 = label[1]
            self.y3 = label[2]
            self.freespace = label[3]

    def generate_sample(self, path):

        img = tf.io.read_file(path)
        img = tf.io.decode_png(img, channels=3)
        img = tf.cond( tf.less(
                    tf.random.uniform([]),
                    0.5), lambda: tf.image.adjust_gamma(img, 0.4), lambda: img)
        img = tf.image.resize(img, self.input_size)
        img = tf.cast(img, tf.float32)/ 255.
        mask_path= tf.strings.regex_replace(path, "gt_image", "gt_binary_image")
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_png(mask, channels=1)
        mask = mask / 255
        mask = tf.image.resize(mask, self.input_size)
        
        return img,  (self.y1[0], self.y2[0], self.y3[0], self.freespace[0], mask)
    
    def build(self):
        list_ds = tf.data.Dataset.list_files(self.src_dir)
        num = tf.data.experimental.cardinality(list_ds)
        #print(self.y1)
        ds = list_ds.shuffle(2000).map(lambda x: self.generate_sample(x)).repeat().batch(self.batch_size).prefetch(1)
        return ds, num

if __name__ == '__main__':
    """
    test code
    """
    anchors = get_anchors('config/yolo_anchors.txt')
    num_lane =2
    num_drive =2
    num_classes =12
    input_shape = [224,224]
    batch_size = 8
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
    val_dataset_builder = Dataset('data/val/*.tfrecords',
                                  batch_size,
                                  anchors,
                                  num_lane,
                                  num_drive,
                                  num_classes,
                                  input_shape,
                                  mode=DATASET_MODE.VALIDATE)
    val_dataset, val_num = val_dataset_builder.build()

    dummy_data = val_dataset.take(1)
    del val_dataset
    del val_num
    val_dataset_builder = LaneDataset( "/home/mdinh/Pictures/tusimple/testing/gt_image/*.png",
                                  batch_size,
                                  anchors,
                                  num_lane,
                                  num_drive,
                                  num_classes,
                                  input_shape, dummy_data)
    val_dataset, val_num = val_dataset_builder.build()
    print(val_num,val_dataset.take(1))
