import colorsys
import cv2
import numpy as np
import tensorflow as tf
from loss import yolo_eval, YoloEval
from timeit import default_timer as timer
from tools.utils import letterbox_image, get_anchors, get_classes

anchors = get_anchors('config/yolo_anchors.txt')
class_names = get_classes('config/bdd100k_classes.txt')
nms = 0.5
score = 0.3

hsv_tuples = [
        (x / len(class_names), 1., 1.)
            for x in range(len(class_names))
        ]
colors = list(
            map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
np.random.seed(10101)  # Fixed seed for consistent colors across runs.
np.random.shuffle(
            colors)  # Shuffle colors to decorrelate adjacent classes.
np.random.seed(None)
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/mdinh/multinet_carvision/export_model/tflite_model/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_shape = input_details[0]['shape']
#image_path = "/home/mdinh/Documents/test/c259431a-fc5e4e05.jpg"
video_path = '/home/mdinh/Videos/test.mp4'
#Load video
vid = cv2.VideoCapture(video_path)
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")
video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
video_fps = vid.get(cv2.CAP_PROP_FPS)
video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
#content = tf.io.read_file(image_path)
    #image = tf.image.decode_image(content,
    #                            channels=3,
    #                            dtype=tf.float32)
    return_value, frame = vid.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.
    input_data = tf.expand_dims(tf.image.resize(image, [416,416]), 0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_y1 = interpreter.get_tensor(output_details[0]['index'])
    pred_mask = interpreter.get_tensor(output_details[1]['index'])
    output_y2 = interpreter.get_tensor(output_details[2]['index'])
    output_y3 = interpreter.get_tensor(output_details[3]['index'])

    out_boxes, out_scores, out_classes = yolo_eval(
                [output_y1, output_y2, output_y3],
                anchors,
                len(class_names),
                video_size,
                score_threshold=score,
                iou_threshold=nms)
    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    for i, c in reversed(list(enumerate(out_classes))):
        box = out_boxes[i]       
        top, left, bottom, right = box        
        #frame = cv2.rectangle(frame,(left,top), (right, bottom), colors[c])
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow('result', frame)
#print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
#print(output_y1)
