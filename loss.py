import tensorflow as tf


def cross_entropy_with_softmax(label, pred):
    
    num_classes = pred.shape[-1]
    label_onehot = tf.one_hot(label, num_classes, axis=-1)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot, logits=pred, axis=-1)
    mean_loss = tf.math.reduce_mean(tf.math.reduce_mean(loss, axis=-1), axis=0)
    
    print('****current loss:', mean_loss.numpy())
    
    return mean_loss
