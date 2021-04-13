import numpy as np
import tensorflow as tf


def index_sieve(features, indices, fix_dim=2):
    features_sieve = tf.gather(features[:, :,...], indices[:, :,...], axis=-2, batch_dims=fix_dim)
#     features_sieve = tf.concat([tf.expand_dims(tf.gather(features[:, i,...], indices[:, i,...], axis=-2, batch_dims=1), axis=1) for i in range(features.shape[1])], axis=1)
    return features_sieve

def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


