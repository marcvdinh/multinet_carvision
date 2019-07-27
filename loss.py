import tensorflow as tf 
from typing import List, Tuple
from yolo3.enum import BOX_LOSS
import numpy as np

def yolo_head(feats: tf.Tensor,
              anchors: np.ndarray,
              input_shape: tf.Tensor,
              calc_loss: bool = False
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = tf.shape(feats)[1:3]
    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], -1)
    grid = tf.cast(grid, feats.dtype)

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(
        grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * tf.cast(
        anchors_tensor, feats.dtype) / tf.cast(input_shape[::-1], feats.dtype)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    if calc_loss == True:
        return grid, box_xy, box_wh, box_confidence
    box_class_probs = tf.sigmoid(feats[..., 5:])
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy: tf.Tensor, box_wh: tf.Tensor,
                       input_shape: tf.Tensor, image_shape) -> tf.Tensor:
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, box_yx.dtype)
    image_shape = tf.cast(image_shape, box_yx.dtype)
    max_shape = tf.maximum(image_shape[0], image_shape[1])
    ratio = image_shape / max_shape
    boxed_shape = input_shape * ratio
    offset = (input_shape - boxed_shape) / 2.
    scale = image_shape / boxed_shape
    box_yx = (box_yx * input_shape - offset) * scale
    box_hw *= input_shape * scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat(
        [
            tf.clip_by_value(box_mins[..., 0:1], 0, image_shape[0]),  # y_min
            tf.clip_by_value(box_mins[..., 1:2], 0, image_shape[1]),  # x_min
            tf.clip_by_value(box_maxes[..., 0:1], 0, image_shape[0]),  # y_max
            tf.clip_by_value(box_maxes[..., 1:2], 0, image_shape[1])  # x_max
        ],
        -1)
    return boxes


def yolo_boxes_and_scores(feats: tf.Tensor, anchors: List[Tuple[float, float]],
                          num_classes: int, input_shape: Tuple[int, int],
                          image_shape) -> Tuple[tf.Tensor, tf.Tensor]:
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
        feats, anchors, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs: List[tf.Tensor],
              anchors: np.ndarray,
              num_classes: int,
              image_shape,
              max_boxes: int = 20,
              score_threshold: float = .6,
              iou_threshold: float = .5
             ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
    """Evaluate YOLO model on given input and return filtered boxes."""

    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],
                                                    num_classes, input_shape,
                                                    image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)
    max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        nms_index = tf.image.non_max_suppression(boxes,
                                                 box_scores[:,c],
                                                 max_boxes_tensor,
                                                 iou_threshold=iou_threshold,
                                                 score_threshold=score_threshold)
        class_boxes = tf.gather(boxes, nms_index)
        class_box_scores = tf.gather(box_scores[:,c], nms_index)
        classes = tf.ones_like(class_box_scores, tf.int32) * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0, name='scores')
    classes_ = tf.concat(classes_, axis=0, name='classes')
    boxes_ = tf.cast(boxes_, tf.int32, name='boxes')
    return boxes_, scores_, classes_
class YoloEval(tf.keras.layers.Layer):
    def __init__(self,anchors,num_classes,image_shape,max_boxes=20,score_threshold=.6,iou_threshold=.5,**kwargs):
        super(YoloEval,self).__init__(**kwargs)
        self.anchors=anchors
        self.num_classes=num_classes
        self.image_shape=image_shape
        self.max_boxes=max_boxes
        self.score_threshold=score_threshold
        self.iou_threshold=iou_threshold

    def call(self,yolo_outputs):
        return yolo_eval(yolo_outputs,self.anchors,self.num_classes,self.image_shape,self.max_boxes,self.score_threshold,self.iou_threshold)

    def get_config(self):
        config=super(YoloEval,self).get_config()
        config['anchors']=self.anchors
        config['num_classes']=self.num_classes
        config['image_shape']=self.image_shape
        config['max_boxes']=self.max_boxes
        config['score_threshold']=self.score_threshold
        config['iou_threshold']=self.iou_threshold

        return config

def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_giou(b1, b2):
    # Expand dim to apply broadcasting.
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / union_area

    bc_mins = tf.minimum(b1_mins, b2_mins)
    bc_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(bc_maxes - bc_mins, 0.)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (enclose_area - union_area) / enclose_area
    return giou


if tf.version.VERSION.startswith('1.'):

    def YoloLoss(y_true,
                 yolo_output,
                 idx,
                 anchors,
                 ignore_thresh: float = .5,
                 box_loss=BOX_LOSS.GIOU,
                 print_loss: bool = False):
        '''Return yolo_loss tensor
        Parameters
        ----------
        yolo_output: the output of yolo_body
        y_true: the output of preprocess_true_boxes
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss
        Returns
        -------
        loss: tensor, shape=(1,)
        '''
        grid_steps = [32, 16, 8]
        grid_step = grid_steps[idx]
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        loss = 0
        m = tf.shape(yolo_output)[0]  # batch size, tensor
        mf = tf.cast(m, yolo_output.dtype)
        object_mask = y_true[..., 4:5]
        true_class_probs = y_true[..., 5:]
        input_shape = tf.shape(yolo_output)[1:3] * grid_step
        grid, pred_xy, pred_wh, box_confidence = yolo_head(
            yolo_output, anchors[anchor_mask[idx]], input_shape, calc_loss=True)
        pred_box = tf.concat([pred_xy, pred_wh], -1)
        # Find ignore mask, iterate over each of batch.
        object_mask_bool = tf.cast(object_mask, 'bool')

        true_box = tf.boolean_mask(y_true[..., 0:4], object_mask_bool[..., 0])
        iou = box_iou(tf.expand_dims(pred_box, -2), tf.expand_dims(true_box, 0))
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, true_box.dtype)

        ignore_mask = tf.expand_dims(ignore_mask, -1)
        confidence_loss = (object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                 logits=yolo_output[..., 4:5]) + \
                           (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                       logits=yolo_output[...,
                                                                                              4:5]) * ignore_mask)
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_class_probs, logits=yolo_output[..., 5:])
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf

        if box_loss == BOX_LOSS.GIOU:
            giou = box_giou(pred_box[..., :4], y_true[..., :4])
            giou_loss = object_mask * (1 - tf.expand_dims(giou, -1))
            giou_loss = tf.reduce_sum(giou_loss) / mf
            loss += giou_loss + confidence_loss + class_loss
            if print_loss:
                tf.print(str(idx)+':',giou_loss, confidence_loss, class_loss,tf.reduce_sum(ignore_mask))
        elif box_loss == BOX_LOSS.MSE:
            grid_shape = tf.cast(tf.shape(yolo_output)[1:3], y_true.dtype)
            raw_true_xy = y_true[..., :2] * grid_shape[::-1] - grid
            raw_true_wh = tf.math.log(y_true[..., 2:4] /
                                      anchors[anchor_mask[idx]] *
                                      input_shape[::-1])
            raw_true_wh = tf.keras.backend.switch(object_mask, raw_true_wh,
                                                  tf.zeros_like(raw_true_wh))
            box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=raw_true_xy, logits=yolo_output[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(
                raw_true_wh - yolo_output[..., 2:4])
            xy_loss = tf.reduce_sum(xy_loss) / mf
            wh_loss = tf.reduce_sum(wh_loss) / mf
            loss += xy_loss + wh_loss + confidence_loss + class_loss
            if print_loss:
                tf.print(loss, xy_loss, wh_loss, confidence_loss, class_loss,
                         tf.reduce_sum(ignore_mask))
        return loss