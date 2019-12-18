from __future__ import absolute_import, division, print_function, unicode_literals
from absl import app, flags
import tensorflow as tf
import os.path as ops
import json
import cv2
import contextlib2
import numpy as np
import bezier
import six


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):           
    value = six.binary_type(value, encoding='utf-8') 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
    Args:
        exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
        base_path: The base path for all shards
        num_shards: The number of shards
    Returns:
        The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}_{:05d}_of_{:05d}.tfrecords'.format(base_path, idx+1, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def create_tf_example(src_dir,img):
    
    check = False
    image_name = img['name'].split('.')[0]
    image_path = ops.join(src_dir, 'training', "gt_image")
    image_path = ops.join(image_path, img['name'])
    
    drive_name = image_name + "_drivable_id.png"
    drive_path = ops.join(src_dir,'drivable_maps', "labels", "val")
    drive_path = ops.join(drive_path, drive_name)
    assert ops.exists(drive_path), '{:s} not exist'.format(drive_path)
    
    encoded_drive_label = tf.io.gfile.GFile(drive_path, 'rb').read()
    #encoded_drive_label = tf.image.decode_png(encoded_drive_label, channels=1)
    #encoded_drive_label = tf.image.encode_jpeg(encoded_drive_label, format='grayscale', quality=100).numpy()
    labels = img['labels']
    lanes = [label for label in labels if label['category'] == 'lane']

    
    encoded_image_data = tf.io.gfile.GFile(image_path, 'rb').read()
    image_shape = tf.image.decode_jpeg(encoded_image_data, channels=3).shape
    
    #write tfrecord
    if 1:
        cv2.imwrite(lane_path, lane_image)
        encoded_lane_label = tf.io.gfile.GFile(lane_path, 'rb').read()
        tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': _bytes_feature(filename), 
        'image/encoded': _bytes_feature(encoded_image_data),
        'image/lane': _bytes_feature(encoded_lane_label),
    }))
    else: tf_example=[]

    return tf_example


def main(_):
    json_file_path = '/home/marcdinh/Documents/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    src_dir = '/home/marcdinh/Documents/BDD100K/bdd100k/'
    num_shards=10
    output_filebase='data/val/val_dataset'
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
        tf_record_close_stack, output_filebase, num_shards)
        with open(json_file_path, 'r') as file:
            info_dict = json.load(file)
            print(len(info_dict))
            for index, img in enumerate(info_dict):
                tf_example = create_tf_example(src_dir, img)
                if tf_example :         
                    output_shard_index = index % num_shards
                    output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                


if __name__ == '__main__': 
    app.run(main)
