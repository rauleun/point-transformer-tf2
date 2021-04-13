import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend

class MeanIoU_total(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='MeanIoU_total', dtype=None):
        super(MeanIoU_total, self).__init__(name=name, dtype=dtype)
        
        self.num_classes = num_classes
        self.total_cm = self.add_weight('total_confusion_matrix',
                                        shape=(num_classes, num_classes),
                                        initializer=init_ops.zeros_initializer,
                                        dtype=dtypes.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])
        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])

        current_cm = confusion_matrix.confusion_matrix(y_true,
                                                       y_pred,
                                                       self.num_classes,
                                                       weights=sample_weight,
                                                       dtype=dtypes.float64)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        sum_over_row = math_ops.cast(math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = math_ops.cast(math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = math_ops.cast(array_ops.tensor_diag_part(self.total_cm), dtype=self._dtype)

        denominator = sum_over_row + sum_over_col - true_positives
        num_valid_entries = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

        iou = math_ops.div_no_nan(true_positives, denominator)

        return math_ops.div_no_nan(math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    def reset_states(self):
        backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def MeanIoU_batch(y_true, y_pred):
    
    num_classes = y_pred.shape[-1]
    y_pred = tf.argmax(y_pred, axis=-1)
    
    if y_pred.shape.ndims > 1:
        y_pred = array_ops.reshape(y_pred, [-1])
    if y_true.shape.ndims > 1:
        y_true = array_ops.reshape(y_true, [-1])
        
    batch_cm = confusion_matrix.confusion_matrix(y_true, y_pred, num_classes, dtype=dtypes.float64)
    sum_over_row = math_ops.cast(math_ops.reduce_sum(batch_cm, axis=0), dtype=dtypes.float64)
    sum_over_col = math_ops.cast(math_ops.reduce_sum(batch_cm, axis=1), dtype=dtypes.float64)
    true_positives = math_ops.cast(array_ops.tensor_diag_part(batch_cm), dtype=dtypes.float64)

    denominator = sum_over_row + sum_over_col - true_positives
    num_valid_entries = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(denominator, 0), dtype=dtypes.float64))

    iou = math_ops.div_no_nan(true_positives, denominator)
    miou = math_ops.div_no_nan(math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    return miou

