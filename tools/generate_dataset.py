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
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
            # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
            # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box) 
    check = False
    image_name = img['name'].split('.')[0]
    image_path = ops.join(src_dir, 'images', "100k", "train")
    image_path = ops.join(image_path, img['name'])
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    drive_name = image_name + "_drivable_id.png"
    drive_path = ops.join(src_dir,'drivable_maps', "labels", "train")
    drive_path = ops.join(drive_path, drive_name)
    assert ops.exists(drive_path), '{:s} not exist'.format(drive_path)
    
    encoded_drive_label = tf.io.gfile.GFile(drive_path, 'rb').read()
    #encoded_drive_label = tf.image.decode_png(encoded_drive_label, channels=1)
    #encoded_drive_label = tf.image.encode_jpeg(encoded_drive_label, format='grayscale', quality=100).numpy()
    labels = img['labels']
    lanes = [label for label in labels if label['category'] == 'lane']

    
    encoded_image_data = tf.io.gfile.GFile(image_path, 'rb').read()
    image_shape = tf.image.decode_jpeg(encoded_image_data, channels=3).shape
    lane_image = np.zeros([image_shape[0], image_shape[1]], np.uint8)
    lane_path = ops.join(src_dir, "bdd100k_lanes", image_name +'.png')
    for _, lane in enumerate(lanes):

        if lane['attributes']['laneDirection'] == 'parallel':
            
            handle_pts = lane['poly2d'][0]['vertices']
            handle_pts = np.transpose(handle_pts)
            handle_pts = handle_pts.astype(float)
            nodes = np.asfortranarray(handle_pts)
            curve = bezier.Curve.from_nodes(nodes)
            s_vals = np.linspace(0.0, 1.0, 15)
            lane_pts = curve.evaluate_multi(s_vals)
            lane_pts = np.transpose(lane_pts)  
            #binary lane map
            cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
                              color=255, thickness=10)        
            #if lane['attributes']['laneType'] == 'single white':
            #    check = True
            #    cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
            #                  color=120, thickness=10)
            #if lane['attributes']['laneType'] == 'double white':
            #    check = True
            #    cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
            #                  color=170, thickness=10)
            #if lane['attributes']['laneType'] == 'single yellow':
            #    check = True
            #    cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
            #                  color=220, thickness=10)
            #if lane['attributes']['laneType'] == 'double yellow':
            #    check = True
            #    cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
            #                  color=255, thickness=10)
            #else: check = False
    
    
    filename = img['name'].encode('utf-8') # Filename of the image. Empty if image is not from file
    #image_format = b"jpg"
    
    traffic_signs = [label for label in labels if label['category'] == 'traffic sign']
    traffic_lights = [label for label in labels if label['category'] == 'traffic light']
    cars = traffic_signs = [label for label in labels if label['category'] == 'car']
    riders = [label for label in labels if label['category'] == 'rider']
    motors = [label for label in labels if label['category'] == 'motor']
    bikes = [label for label in labels if label['category'] == 'bike']
    buses = [label for label in labels if label['category'] == 'bus']
    trucks = [label for label in labels if label['category'] == 'truck']
    persons = [label for label in labels if label['category'] == 'person']
 

    for _, traffic_sign in enumerate(traffic_signs):
        xmins.append(float(traffic_sign["box2d"]["x1"]))
        xmaxs.append(float(traffic_sign["box2d"]["x2"]))

        ymins.append(float(traffic_sign["box2d"]["y1"]))
        ymaxs.append(float(traffic_sign["box2d"]["y2"]))

        classes_text.append(b"traffic sign")
        classes.append(0)
        
    for _, car in enumerate(cars):
        xmins.append(car["box2d"]["x1"])
        xmaxs.append(car["box2d"]["x2"])

        ymins.append(car["box2d"]["y1"])
        ymaxs.append(car["box2d"]["y2"])

        classes_text.append(b"car")
        classes.append(5)
        
    for _, rider in enumerate(riders):
        xmins.append(rider["box2d"]["x1"])
        xmaxs.append(rider["box2d"]["x2"])

        ymins.append(rider["box2d"]["y1"])
        ymaxs.append(rider["box2d"]["y2"])

        classes_text.append(b"rider")
        classes.append(6)

    for _, motor in enumerate(motors):
        xmins.append(motor["box2d"]["x1"])
        xmaxs.append(motor["box2d"]["x2"])

        ymins.append(motor["box2d"]["y1"])
        ymaxs.append(motor["box2d"]["y2"])

        classes_text.append(b"motor")
        classes.append(7)

    for _, bike in enumerate(bikes):
        xmins.append(bike["box2d"]["x1"])
        xmaxs.append(bike["box2d"]["x2"])

        ymins.append(bike["box2d"]["y1"])
        ymaxs.append(bike["box2d"]["y2"])

        classes_text.append(b"bike")
        classes.append(8)

    for _, bus in enumerate(buses):
        xmins.append(bus["box2d"]["x1"])
        xmaxs.append(bus["box2d"]["x2"])

        ymins.append(bus["box2d"]["y1"])
        ymaxs.append(bus["box2d"]["y2"])

        classes_text.append(b"bus")
        classes.append(9)
        
    for _, truck in enumerate(trucks):
        xmins.append(truck["box2d"]["x1"])
        xmaxs.append(truck["box2d"]["x2"])

        ymins.append(truck["box2d"]["y1"])
        ymaxs.append(truck["box2d"]["y2"])

        classes_text.append(b"truck")
        classes.append(10)
        
    for _, person in enumerate(persons):
        xmins.append(person["box2d"]["x1"])
        xmaxs.append(person["box2d"]["x2"])

        ymins.append(person["box2d"]["y1"])
        ymaxs.append(person["box2d"]["y2"])

        classes_text.append(b"person")
        classes.append(11)
        
    for _, traffic_light in enumerate(traffic_lights):
        xmins.append(traffic_light["box2d"]["x1"])
        xmaxs.append(traffic_light["box2d"]["x2"])

        ymins.append(traffic_light["box2d"]["y1"])
        ymaxs.append(traffic_light["box2d"]["y2"])

        if traffic_light["attributes"]["trafficLightColor"] == "red":
            classes_text.append(b"traffic lig: red")
            classes.append(1)
        if traffic_light["attributes"]["trafficLightColor"] == "yellow":
            classes_text.append(b"traffic lig: none")
            classes.append(2)  
        if traffic_light["attributes"]["trafficLightColor"] == "green":
            classes_text.append(b"traffic lig: green")
            classes.append(3)
        if traffic_light["attributes"]["trafficLightColor"] == "none":
            classes_text.append(b"traffic lig: none")
            classes.append(4)  
    #write tfrecord
    if 1:
        cv2.imwrite(lane_path, lane_image)
        encoded_lane_label = tf.io.gfile.GFile(lane_path, 'rb').read()
        tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': _bytes_feature(filename),
        
        'image/encoded': _bytes_feature(encoded_image_data),
        'image/lane': _bytes_feature(encoded_lane_label),
        'image/drive': _bytes_feature(encoded_drive_label),
        'image/object/bbox/xmin': _float_feature(xmins),
        'image/object/bbox/xmax': _float_feature(xmaxs),
        'image/object/bbox/ymin':_float_feature(ymins),
        'image/object/bbox/ymax': _float_feature(ymaxs),
        'image/object/bbox/label': _int64_feature(classes),
    }))
    else: tf_example=[]

    return tf_example


def main(_):
    json_file_path = '/home/marcdinh/Documents/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    src_dir = '/home/marcdinh/Documents/BDD100K/bdd100k/'
    num_shards=10
    output_filebase='data/train/train_dataset'
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
