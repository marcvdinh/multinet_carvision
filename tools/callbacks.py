import tensorflow as tf
import numpy as np
from loss import yolo_eval
from tools.utils import letterbox_image, bind, expand_seg_label, get_classes,get_anchors
from timeit import default_timer as timer
from data import Dataset
from tools.modes import DATASET_MODE
import io
from PIL import Image, ImageFont, ImageDraw
#from sklearn import skimage
import colorsys
AUTOTUNE = tf.data.experimental.AUTOTUNE

#TODO implement MIOU for segmentation

#FIXME access to validation dataset at epoch end so that we avoid creating a new test dataset each time.
class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self,
                input_shapes,
                anchors,
                class_names,
                validation_data,
                glob_path,
                tag,
                batch_size = 1):
        super().__init__() 
        
        if isinstance(input_shapes, list):
            self.input_shapes = input_shapes
            self.input_shape = tf.Variable(name="input_shape",
                                           initial_value=self.input_shapes,
                                           trainable=False)
        else:
            self.input_shape = input_shapes
        self.val_data = validation_data
        self.glob_path = glob_path
        self.tag = tag 
        self.batch_size = batch_size
        self.anchors = anchors
        self.class_names = class_names
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
            self.colors)  # Shuffle colors to decorrelate

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
        lane = tf.image.decode_image(features['image/lane'],
                                      channels=1,
                                      dtype=tf.uint8
                                      )
        drive = tf.image.decode_image(features['image/drive'],
                                      channels=1,
                                      dtype=tf.uint8
                                      )
        return image, lane, drive

    def create_image(self, tensor):
        height, width, channel = tensor.shape
        image = tf.keras.preprocessing.image.array_to_img(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

    def create_mask(self,pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return tf.cast(pred_mask, tf.float32)

    def on_epoch_end(self, epoch, logs={}):

        test_dataset_builder = Dataset(self.glob_path,
                                       self.batch_size,
                                       input_shapes=self.input_shape,
                                       mode=DATASET_MODE.TEST)
        bind(test_dataset_builder, self.parse_tfrecord)
        test_dataset, test_num = test_dataset_builder.build()
        
        for image, lane, drive in test_dataset.take(1):
            if self.input_shape != (None, None):
                boxed_image, resized_image_shape = letterbox_image(
                    image, self.input_shape)
                boxed_lane, _ = letterbox_image(
                   lane, self.input_shape)
                boxed_drive, _ = letterbox_image(
                    drive, self.input_shape)
                
            else:
                _, height, width, _ = tf.shape(image)
                new_image_size = (height - (height % 32), width - (width % 32))
                boxed_image, resized_image_shape = letterbox_image(
                    image, new_image_size)
            image_data = np.array(boxed_image)
            output = self.model.predict(image_data)
            image_shape = tf.shape(image)[1:3]
            image_detect = Image.fromarray((np.array(tf.squeeze(image)) * 255).astype('uint8'),
                                    'RGB')
        out_boxes, out_scores, out_classes = yolo_eval(
                [output[2], output[3], output[4]],
                self.anchors,
                len(self.class_names),
                image_shape,
                score_threshold=0.2,
                iou_threshold=0.5)
        
        
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image_detect.size[1] +
                                                    0.5).astype('int32'))
        thickness = (image_detect.size[1] + image_detect.size[0]) // 300
        draw = ImageDraw.Draw(image_detect)
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            #print(label, (left, top), (right, bottom))

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
        #image_detect.show()
        yolo_output = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image_detect), 0)
        pred_lane_mask = self.create_mask(output[0])
        pred_drive_mask = self.create_mask(output[1])
        writer = tf.contrib.summary.create_file_writer('./tboard')
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image("image input", boxed_image)
            tf.contrib.summary.image("lane segmentation", tf.concat([boxed_lane, pred_lane_mask],0))
            tf.contrib.summary.image("drive segmentation", tf.concat([boxed_drive, pred_drive_mask],0))
            tf.contrib.summary.image("yolo output", yolo_output)
        return
#TODO implement confusion matrix for object detection
#class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    
        
class MAPCallback(tf.keras.callbacks.Callback):
    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    """

    def _voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

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
        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
        label = tf.expand_dims(features['image/object/bbox/label'].values, 0)
        bbox = tf.concat([xmin, ymin, xmax, ymax,
                          tf.cast(label, tf.float32)], 0)
        return image, bbox

    def parse_text(self, line):
        values = tf.strings.split([line], ' ').values
        image = tf.image.decode_image(tf.io.read_file(values[0]),
                                      channels=3,
                                      dtype=tf.float32)
        image.set_shape([None, None, 3])
        reshaped_data = tf.reshape(values[1:], [-1, 5])
        xmins = tf.strings.to_number(reshaped_data[:, 0], tf.float32)
        xmaxs = tf.strings.to_number(reshaped_data[:, 2], tf.float32)
        ymins = tf.strings.to_number(reshaped_data[:, 1], tf.float32)
        ymaxs = tf.strings.to_number(reshaped_data[:, 3], tf.float32)
        labels = tf.strings.to_number(reshaped_data[:, 4], tf.int64)
        bbox = tf.concat(
            [xmins, ymins, xmaxs, ymaxs,
             tf.cast(labels, tf.float32)], 0)
        return image, bbox

    def calculate_aps(self):
        test_dataset_builder = Dataset(self.glob_path,
                                       self.batch_size,
                                       input_shapes=self.input_shape,
                                       mode=DATASET_MODE.TEST)
        bind(test_dataset_builder, self.parse_tfrecord)
        bind(test_dataset_builder, self.parse_text)
        test_dataset, test_num = test_dataset_builder.build()
        true_res = {}
        pred_res = []
        idx = 0
        APs = {}
        start = timer()
        for image, bbox in test_dataset:
            if self.input_shape != (None, None):
                boxed_image, resized_image_shape = letterbox_image(
                    image, tuple(reversed(self.input_shape)))
            else:
                _, height, width, _ = tf.shape(image)
                new_image_size = (width - (width % 32), height - (height % 32))
                boxed_image, resized_image_shape = letterbox_image(
                    image, new_image_size)
            output = self.model.predict(boxed_image.numpy())
            out_boxes, out_scores, out_classes = yolo_eval(
                output,
                self.anchors,
                self.num_classes,
                image.shape[1:3],
                score_threshold=self.score,
                iou_threshold=self.nms)
            if len(out_classes) > 0:
                for i in range(len(out_classes)):
                    top, left, bottom, right = out_boxes[i]
                    pred_res.append([
                        idx, out_classes[i].numpy(), out_scores[i].numpy(),
                        left, top, right, bottom
                    ])
            true_res[idx] = tf.transpose(bbox[0]).numpy()
            idx += 1
        end = timer()
        print((end - start) / test_num)
        for cls in range(self.num_classes):
            pred_res_cls = [x for x in pred_res if x[1] == cls]
            if len(pred_res_cls) == 0:
                continue
            true_res_cls = {}
            npos = 0
            for index in true_res:
                objs = [obj for obj in true_res[index] if obj[4] == cls]
                npos += len(objs)
                BBGT = np.array([x[:4] for x in objs])
                true_res_cls[index] = {
                    'bbox': BBGT,
                    'difficult': [False] * len(objs),
                    'det': [False] * len(objs)
                }
            ids = [x[0] for x in pred_res_cls]
            scores = np.array([x[2] for x in pred_res_cls])
            bboxs = np.array([x[3:] for x in pred_res_cls])
            sorted_ind = np.argsort(-scores)
            bboxs = bboxs[sorted_ind, :]
            ids = [ids[x] for x in sorted_ind]

            nd = len(ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for j in range(nd):
                res = true_res_cls[ids[j]]
                bbox = bboxs[j, :].astype(float)
                ovmax = -np.inf
                BBGT = res['bbox'].astype(float)
                if BBGT.size > 0:
                    ixmin = np.maximum(BBGT[:, 0], bbox[0])
                    iymin = np.maximum(BBGT[:, 1], bbox[1])
                    ixmax = np.minimum(BBGT[:, 2], bbox[2])
                    iymax = np.minimum(BBGT[:, 3], bbox[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                if ovmax > self.iou:
                    if not res['difficult'][jmax]:
                        if not res['det'][jmax]:
                            tp[j] = 1.
                            res['det'][jmax] = 1
                        else:
                            fp[j] = 1.
                else:
                    fp[j] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
            APs[cls] = ap
        return APs

    def __init__(self,
                 glob_path,
                 input_shapes,
                 anchors,
                 class_names,
                 score=0.,
                 iou=.5,
                 nms=.5,
                 batch_size=1):
        if isinstance(input_shapes, list):
            self.input_shape = input_shapes[0]
        else:
            self.input_shape = input_shapes
        self.anchors = anchors
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.glob_path = glob_path
        self.score = score
        self.iou = iou
        self.nms = nms
        self.batch_size = batch_size

    def on_train_end(self, logs={}):
        logs = logs or {}
        origin_learning_phase = tf.keras.backend.learning_phase()
        tf.keras.backend.set_learning_phase(0)
        APs = self.calculate_aps()
        tf.keras.backend.set_learning_phase(origin_learning_phase)
        for cls in range(self.num_classes):
            if cls in APs:
                print(self.class_names[cls] + ' ap: ', APs[cls])
        mAP = np.mean([APs[cls] for cls in APs])
        print('mAP: ', mAP)
        logs['mAP'] = mAP

