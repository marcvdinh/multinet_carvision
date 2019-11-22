# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from loss import yolo_eval, YoloEval
from tools.utils import letterbox_image, get_anchors, get_classes
from tools.modes import OPT, BACKBONE
from tools.callbacks import MAPCallback
import os
from typing import List, Tuple
#from tensorflow_serving.apis import prediction_log_pb2, predict_pb2
from tensorflow.python import debug as tf_debug
from functools import partial
from model import CarNet
tf.keras.backend.set_learning_phase(0)


class YOLO(object):
    _defaults = {
        "score": 0.2,
        "nms": 0.5,
    }

    @classmethod
    def get_defaults(cls, n: str):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, FLAGS):
        self.__dict__.update(self._defaults)  # set up default values
        self.backbone = FLAGS['backbone']
        self.opt = FLAGS['opt']
        self.class_names = get_classes(FLAGS['classes_path'])
        self.anchors = get_anchors(FLAGS['anchors_path'])
        self.input_shape = FLAGS['input_size']
        self.generate(FLAGS)
        self.score= 0.2
        self.nms = 0.5

    def generate(self, FLAGS):
        model_path = os.path.expanduser(FLAGS['model'])
       # if model_path.endswith(
       #     '.h5') is not True:
       #     model_path=tf.train.latest_checkpoint(model_path)

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        num_lane = 2
        num_drive = 3
        try:
            model = tf.keras.models.load_model(model_path)
        #model.summary()
        except:
            #try:
            print("building model and loading_weights")
            multinet = CarNet(self.backbone, tf.keras.layers.Input(shape=(*self.input_shape,3)),
                                                weights_path=model_path,
                                                n_class=num_classes,
                                                n_anchors=len(self.anchors)//3,
                                                n_lane_embedding=num_lane,
                                                n_drive_embedding=num_drive,
                                                alpha=1.4)
        
            model = multinet.build()
            model.load_weights(model_path)
            #except:
            #    imported = tf.saved_model.load(model_path)
        
        
        self.yolo_model = model
        # Generate output tensor targets for filtered bounding boxes.
        hsv_tuples = [
            (x / len(self.class_names), 1., 1.)
            for x in range(len(self.class_names))
        ]
        self.colors = list(
            map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(
            self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

    def detect_image(self, image, draw=True) -> Image:
        if tf.executing_eagerly():
            image_data = tf.expand_dims(image, 0)
            if self.input_shape != (None, None):
                boxed_image, image_shape = letterbox_image(
                    image_data, tuple(self.input_shape))
            else:
                height, width, _ = image_data.shape
                new_image_size = (height - (height % 32), width - (width % 32))
                boxed_image, image_shape = letterbox_image(
                    image_data, new_image_size)
            boxed_image = np.array(boxed_image)
            start = timer()
            try:
                output = self.yolo_model.predict(boxed_image)
            except:
                graph_func = self.yolo_model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
                frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(
    graph_func)
                def wrap_func(*args, **kwargs):
                #Assumes frozen_func has one output tensor
                    return frozen_func(*args, **kwargs)
                output = wrap_func(boxed_image).numpy()

            pred_mask = tf.argmax(output[0], axis=-1)
            out_boxes, out_scores, out_classes = yolo_eval(
                [output[1], output[2], output[3]],
                self.anchors,
                len(self.class_names),
                image.shape[0:2],
                score_threshold=self.score,
                iou_threshold=self.nms)
            end = timer()
            image = Image.fromarray((np.array(image)*255).astype('uint8'),
                                    'RGB')
            pred_mask = pred_mask.numpy()[0]
            drivable_area = Image.fromarray((255 - pred_mask * 127).astype('uint8'), 'L')
            drivable_area = drivable_area.resize([image.size[0], image.size[1]])
            drivable_color = Image.new("RGB", (image.size[0], image.size[1]),"green")
            image = Image.composite(image, drivable_color, drivable_area)
            #image.show()
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        if draw:
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] +
                                                    0.5).astype('int32'))
            thickness = (image.size[1] + image.size[0]) // 300
            draw = ImageDraw.Draw(image)
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)

                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],
                                   outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin),
                     tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
            print(end - start)
            return image
        else:
            return out_boxes, out_scores, out_classes


def export_tfjs_model(yolo, path):
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(yolo.yolo_model,
                                     path)


def export_serving_model(yolo, path):
    if tf.io.gfile.exists(path):
        overwrite = input("Overwrite existed model(yes/no):")
        if overwrite == 'yes':
            tf.io.gfile.rmtree(path)
        else:
            raise ValueError(
                "Export directory already exists, and isn't empty. Please choose a different export directory, or delete all the contents of the specified directory: "
                + path)
    tf.keras.models.save_model(
        yolo.yolo_model,
        path)


"""     asset_extra = os.path.join(path, "assets.extra")
    tf.io.gfile.mkdir(asset_extra)
    with tf.io.TFRecordWriter(
            os.path.join(asset_extra, "tf_serving_warmup_requests")) as writer:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'detection'
        request.model_spec.signature_name = 'serving_default'
        image = Image.open('../download/image3.jpeg')
        scale = yolo.input_shape[0] / max(image.size)
        if scale < 1:
            image = image.resize((int(line * scale) for line in image.size),
                                 Image.BILINEAR)
        image_data = np.array(image, dtype='uint8')
        image_data = np.expand_dims(image_data, 0)
        request.inputs['predict_image:0'].CopyFrom(
            tf.make_tensor_proto(image_data))
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString()) """


def export_tflite_model(yolo, path, test_dataset_path, quant=True):

    converter = tf.lite.TFLiteConverter.from_keras_model(yolo.yolo_model)
    converter.allow_custom_ops = True
    #converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflite_model = converter.convert()
    open(os.path.join(path,"converted_model.tflite"), "wb").write(tflite_model)

    def open_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, yolo.input_shape)
        img = tf.cast(img, tf.float32)/ 255.
        return img
    if quant:
        list_ds = tf.data.Dataset.list_files(test_dataset_path)
        quant_ds = list_ds.shuffle(len(list_ds)).map(lambda x: open_image(x)).batch(1).prefetch(1)
        def representative_data_gen():
            for input_value in quant_ds.take(100):
                yield [input_value]
        
   
        converter.representative_dataset = representative_data_gen
        tflite_model_quant = converter.convert()
        open(os.path.join(path,"converted_model_quant8.tflite"), "wb").write(tflite_model_quant)

    
    
   


def export_trt_model(yolo, path):

    if tf.io.gfile.exists(path):
        overwrite = input("Overwrite existed model(yes/no):")
        if overwrite == 'yes':
            tf.io.gfile.rmtree(path)
        else:
            raise ValueError(
                "Export directory already exists, and isn't empty. Please choose a different export directory, or delete all the contents of the specified directory: "
                + path)
    yolo.yolo_model.save(
        path,
        save_format='tf')

    #params = params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    #precision_mode='FP16')
    #converter = trt.TrtGraphConverterV2(input_saved_model_dir=path, conversion_params=params)
    #converter.convert()
    #converter.save(path)


def calculate_map(yolo, glob):
    mAP = MAPCallback(glob, yolo.input_shape, yolo.anchors, yolo.class_names)
    mAP.set_model(yolo.yolo_model)
    APs = mAP.calculate_aps()
    for cls in range(len(yolo.class_names)):
        if cls in APs:
            print(yolo.class_names[cls] + ' ap: ', APs[cls])
    mAP = np.mean([APs[cls] for cls in APs])
    print('mAP: ', mAP)

def inference_img(image_path, yolo):
    try:
        if tf.executing_eagerly():
            content = tf.io.read_file(image_path)
            image = tf.image.decode_image(content,
                                            channels=3,
                                            dtype=tf.float32)
        else:
            image = Image.open(image_path)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.show()


def detect_img(yolo):
    while True:
        inputs = input('Input image filename:')
        if inputs.endswith('.txt'):
            with open(input) as file:
                for image_path in file.readlines():
                    image_path = image_path.strip()
                    inference_img(image_path, yolo)
        else:
            inference_img(inputs, yolo)
    yolo.close_session()


def detect_video(yolo: YOLO, video_path: str, output_path: str = ""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC),
              type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    detected = False
    trackers = []
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=30)
    thickness = 1
    frame_count = 0
    while True:
        return_value, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        #image.show()
        image_data = np.array(image, dtype=np.float32 ) / 255.
        result = yolo.detect_image(image_data)
        #result.show()
        result = np.asarray(result)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)  
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result,
                    text=fps,
                    org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50,
                    color=(255, 0, 0),
                    thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


    #TODO Segmentation backend