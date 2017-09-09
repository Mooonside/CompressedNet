import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np


def thin_conv2D(inputs, shape, strides, name, padding="VALID", mask=None):
    with tf.variable_scope(name):
        weights = math_ops.multiply(
            tf.Variable(tf.random_normal(shape=shape), name="kernel"), mask)
        biases = tf.Variable(tf.zeros([shape[-1]]), name="bias")
        conv = tf.nn.conv2d(inputs, weights,
                            strides=strides, padding=padding)
        return tf.nn.relu(conv + biases, name="relu_out")


def thin_dense(inputs, shape, name, relu=True, mask=None):
    with tf.variable_scope(name):
        weights = math_ops.multiply(
            tf.Variable(tf.random_normal(shape=shape), name="kernel"), mask)
        biases = tf.Variable(tf.zeros([shape[-1]]), name="bias")
        dense = math_ops.matmul(inputs, weights)
        if relu:
            return tf.nn.relu(dense + biases, name="relu_out")
        else:
            return tf.add(dense, biases, name="linear_out")


def thin_dense_share(inputs, name, belongings, centroids, relu=True, mask=None):
    with tf.variable_scope(name + "/share"):
        shareVar = []
        for i in range(centroids.shape[0]):
            shareVar.append(
                tf.Variable(
                    initial_value=centroids[i],
                    dtype=tf.float32,
                    name="var" + str(i)
                )
            )
        sf = [shareVar[i] for i in np.reshape(belongings, [-1])]
        weights = math_ops.multiply(tf.reshape(sf, belongings.shape, name="kernel"), mask)
        biases = tf.Variable(tf.zeros([belongings.shape[-1]]), name="bias")
        dense = math_ops.matmul(inputs, weights)
        if relu:
            return tf.nn.relu(dense + biases, name="relu_out")
        else:
            return tf.add(dense, biases, name="linear_out")


def thin_conv_share(inputs, name, belongings, centroids, strides, padding="VALID",
                    relu=True, mask=None):
    with tf.variable_scope(name + "/share", reuse=True):
        shareVar = []
        for i in range(centroids.shape[0]):
            shareVar.append(
                tf.Variable(
                    initial_value=centroids[i],
                    dtype=tf.float32,
                    name="var" + str(i)
                )
            )
        sf = [shareVar[i] for i in np.reshape(belongings, [-1])]
        weights = math_ops.multiply(tf.reshape(sf, belongings.shape, name="kernel"), mask)
        biases = tf.Variable(tf.zeros([belongings.shape[-1]]), name="bias")

        conv = tf.nn.conv2d(inputs, weights, strides=strides, padding=padding)
        if relu:
            out = tf.nn.relu(conv + biases, name="relu_out")
        else:
            out = tf.add(conv, biases, name="linear_out")
        return out
