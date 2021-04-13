import os
import sys
import numpy as np
import tensorflow as tf
import math


def knn(k, pts, query):
    total_points = pts.shape[-2]
    query_points = query.shape[-2]

    pts = tf.tile(tf.reshape(pts, (-1, 1, total_points, 3)), [1, query_points, 1, 1])
    query = tf.tile(tf.reshape(query, (-1, query_points, 1, 3)), [1, 1, total_points, 1])

    distance = tf.norm((pts - query)**2, axis=-1)
    dist_sq, ind = tf.nn.top_k(-1 * distance, k)
    dist = tf.math.sqrt(-dist_sq)
    
    return ind, dist

@tf.function
def farthest_point_sampling(k, pts):
    
    dist = tf.norm(pts[:, None, :, :]-pts[:, :, None, :], axis=-1)
    num_points = pts.shape[-2]
    batch_size = pts.shape[-3]

    indices = []
    values = []

    for b in range(batch_size):
    
        batch_indices = [0]
        batch_values = [0]
        d = dist[b, 0, :]
        for i in range(num_points-1):
            idx = tf.math.argmax(d)
            batch_indices.append(idx)
            batch_values.append(d[idx])
            d = tf.math.minimum(d, dist[b, idx, :])
        indices.append(batch_indices)
        values.append(batch_values)
    indices = tf.convert_to_tensor(indices, dtype=tf.int64)
    values = tf.convert_to_tensor(values, dtype=tf.int64)
    return indices[:, :k], values[:, :k]







