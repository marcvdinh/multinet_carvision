import tensorflow as tf
import os.path as ops
import json
import cv2
from object_detection.utils import dataset_util
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
import numpy as np
import bezier
flags = tf.app.flags
flags.DEFINE_string('output_path', '/media/mdinh/3d67e268-6eff-4dc7-b895-4286e7226904/BDD100k/bdd100k/bbox/train_dataset.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(src_dir,img):
  # TODO(user): Populate the following variables from your example.
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
            # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
            # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    
    
    image_path = ops.join(src_dir, 'images', "100k")
    image_path = ops.join(image_path, img['name'])
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    image_name = img['name'].split('.')[0]
    drive_name = image_name + "_drivable_id.png"
    drive_path = ops.join(src_dir,'drivable_maps', "labels")
    drive_path = ops.join(drive_path, drive_name)
    assert ops.exists(drive_path), '{:s} not exist'.format(drive_path)
    labels = img['labels']
    lanes = [label for label in labels if label['category']=='lane']
    
    #drivable_areas = [label for label in labels if label['category']=='drivable area']
    image_name_new = '{:s}.png'.format('{:s}'.format(image_name).zfill(4))
    src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    drive_image = cv2.imread(drive_path, cv2.IMREAD_COLOR )
    height = src_image.shape[0]
    width = src_image.shape[1]
    lane_image = np.zeros([width, height], np.uint8)
    
    for lane_index, lane in enumerate(lanes):

        if lane['attributes']['laneDirection'] == 'parallel':
            
            handle_pts = lane['poly2d'][0]['vertices']
            handle_pts = np.transpose(handle_pts)
            nodes = np.asfortranarray(handle_pts)
            #print(nodes)
            curve = bezier.Curve.from_nodes(nodes)
            s_vals = np.linspace(0.0, 1.0, 15)
            lane_pts = curve.evaluate_multi(s_vals)
            lane_pts = np.transpose(lane_pts)  
            #cv2.polylines(dst_binary_image, np.int32([lane_pts]), isClosed=False,
            #                  color=255, thickness=5)        
            if lane['attributes']['laneType'] == 'road curb':
                continue
            if lane['attributes']['laneType'] == 'single white':
                draw = True
                cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
                              color=120, thickness=10)
            if lane['attributes']['laneType'] == 'double white':
                draw = True
                cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
                              color=170, thickness=10)
            if lane['attributes']['laneType'] == 'single yellow':
                draw = True
                cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
                              color=220, thickness=10)
            if lane['attributes']['laneType'] == 'double yellow':
                draw = True
                cv2.polylines(lane_image, np.int32([lane_pts]), isClosed=False,
                              color=255, thickness=10)

    filename = bytes(img['name'], 'utf-8') # Filename of the image. Empty if image is not from file
    encoded_image_data = cv2.imencode('.jpg', src_image)[1].tostring() # Encoded image bytes
    encoded_lane_label = cv2.imencode('.jpg', lane_image)[1].tostring()
    encoded_drive_label = cv2.imencode('.jpg', drive_image)[1].tostring()
    image_format = b'jpg' # b'jpeg' or b'png'
    
    

    traffic_signs =  [label for label in labels if label['category']=='traffic sign']
    traffic_lights = [label for label in labels if label['category']=='traffic light']
    cars = traffic_signs =  [label for label in labels if label['category']=='car']
    riders = [label for label in labels if label['category']=='rider']
    motors = [label for label in labels if label['category']=='motor']
    bikes = [label for label in labels if label['category']=='bike']
    buses = [label for label in labels if label['category']=='bus']
    trucks = [label for label in labels if label['category']=='truck']
    persons = [label for label in labels if label['category']=='person']

    for traffic_sign_index, traffic_sign in enumerate(traffic_signs):
        xmins.append(traffic_sign["box2d"]["x1"]/1280.0)
        xmaxs.append(traffic_sign["box2d"]["x2"]/1280.0)

        ymins.append(traffic_sign["box2d"]["y1"]/720.0)
        ymaxs.append(traffic_sign["box2d"]["y2"]/720.0)

        classes_text.append(b"traffic sign")
        classes.append(1)
        
    for car_index, car in enumerate(cars):
        xmins.append(car["box2d"]["x1"]/1280.0)
        xmaxs.append(car["box2d"]["x2"]/1280.0)

        ymins.append(car["box2d"]["y1"]/720.0)
        ymaxs.append(car["box2d"]["y2"]/720.0)

        classes_text.append(b"car")
        classes.append(5)
        
    for rider_index, rider in enumerate(riders):
        xmins.append(rider["box2d"]["x1"]/1280.0)
        xmaxs.append(rider["box2d"]["x2"]/1280.0)

        ymins.append(rider["box2d"]["y1"]/720.0)
        ymaxs.append(rider["box2d"]["y2"]/720.0)

        classes_text.append(b"rider")
        classes.append(6)

    for motor_index, motor in enumerate(motors):
        xmins.append(motor["box2d"]["x1"]/1280.0)
        xmaxs.append(motor["box2d"]["x2"]/1280.0)

        ymins.append(motor["box2d"]["y1"]/720.0)
        ymaxs.append(motor["box2d"]["y2"]/720.0)

        classes_text.append(b"motor")
        classes.append(7)

    for bike_index, bike in enumerate(bikes):
        xmins.append(bike["box2d"]["x1"]/1280.0)
        xmaxs.append(bike["box2d"]["x2"]/1280.0)

        ymins.append(bike["box2d"]["y1"]/720.0)
        ymaxs.append(bike["box2d"]["y2"]/720.0)

        classes_text.append(b"bike")
        classes.append(8)

    for bus_index, bus in enumerate(buses):
        xmins.append(bus["box2d"]["x1"]/1280.0)
        xmaxs.append(bus["box2d"]["x2"]/1280.0)

        ymins.append(bus["box2d"]["y1"]/720.0)
        ymaxs.append(bus["box2d"]["y2"]/720.0)

        classes_text.append(b"bus")
        classes.append(9)
        
    for truck_index, truck in enumerate(trucks):
        xmins.append(truck["box2d"]["x1"]/1280.0)
        xmaxs.append(truck["box2d"]["x2"]/1280.0)

        ymins.append(truck["box2d"]["y1"]/720.0)
        ymaxs.append(truck["box2d"]["y2"]/720.0)

        classes_text.append(b"truck")
        classes.append(10)
        
    for person_index, person in enumerate(persons):
        xmins.append(person["box2d"]["x1"]/1280.0)
        xmaxs.append(person["box2d"]["x2"]/1280.0)

        ymins.append(person["box2d"]["y1"]/720.0)
        ymaxs.append(person["box2d"]["y2"]/720.0)

        classes_text.append(b"person")
        classes.append(11)
        
    for traffic_light_index, traffic_light in enumerate(traffic_lights):
        xmins.append(traffic_light["box2d"]["x1"]/1280.0)
        xmaxs.append(traffic_light["box2d"]["x2"]/1280.0)

        ymins.append(traffic_light["box2d"]["y1"]/720.0)
        ymaxs.append(traffic_light["box2d"]["y2"]/720.0)

        if traffic_light["attributes"]["trafficLightColor"] == "red":
            classes_text.append(b"traffic lig: red")
            classes.append(3)
        if traffic_light["attributes"]["trafficLightColor"] == "yellow":
            classes_text.append(b"traffic lig: none")
            classes.append(4)  
        if traffic_light["attributes"]["trafficLightColor"] == "green":
            classes_text.append(b"traffic lig: green")
            classes.append(2)
        if traffic_light["attributes"]["trafficLightColor"] == "none":
            classes_text.append(b"traffic lig: none")
            classes.append(5)  
    #print(classes, len(classes)) 
    tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/lane': dataset_util.bytes_feature(encoded_lane_label),
    'image/drive': dataset_util.bytes_feature(encoded_drive_label),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
}))
    return tf_example


#def main(_):
#    json_file_path = '/media/mdinh/3d67e268-6eff-4dc7-b895-4286e7226904/BDD100k/bdd100k/labels/toy_labels_train.json'
#    src_dir = '/media/mdinh/3d67e268-6eff-4dc7-b895-4286e7226904/BDD100k/bdd100k/'
#    num_shards=1
#    output_filebase='/media/mdinh/3d67e268-6eff-4dc7-b895-4286e7226904/BDD100k/bdd100k/bbox/toy_train/eval_dataset.tfrecord'
  # TODO(user): Write code to read in your dataset to examples variable
#    with contextlib2.ExitStack() as tf_record_close_stack:
#        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
#        tf_record_close_stack, output_filebase, num_shards)
#        with open(json_file_path, 'r') as file:
#            info_dict = json.load(file)
#            for index, img in enumerate(info_dict):
#                tf_example = create_tf_example(src_dir,img)
#                output_shard_index = index % num_shards
#                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

def main(_):
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    json_file_path = '/media/mdinh/3d67e268-6eff-4dc7-b895-4286e7226904/BDD100k/bdd100k/labels/bdd100k_labels_images_train.json'
    src_dir = '/media/mdinh/3d67e268-6eff-4dc7-b895-4286e7226904/BDD100k/bdd100k/'
    examples =[]
  # TODO(user): Write code to read in your dataset to examples variable
    with open(json_file_path, 'r') as file:
            info_dict = json.load(file)
            for index, img in enumerate(info_dict):
                tf_example = create_tf_example(src_dir,img)
                writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
