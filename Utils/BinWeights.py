import tensorflow as tf
from tensorflow.python.ops import math_ops

from differentiable_abs import differentiable_abs
from differntiable_sign import differentiable_sign


def BinConv(inputs, shape, strides, name, padding="VALID"):
    with tf.variable_scope(name):
        rweights = tf.get_variable(name="kernel", shape=shape, initializer=None)
        filter_size = shape[0] * shape[1] * shape[2]
        # shape : CC
        alpha_weight = tf.divide(tf.reduce_sum(differentiable_abs(rweights), axis=[0, 1, 2]),
                                 tf.to_float(filter_size))
        sign_weight = differentiable_sign(rweights)

        biases = tf.Variable(tf.zeros([shape[-1]]), name="bias")
        conv = tf.nn.conv2d(inputs, sign_weight,
                            strides=strides, padding=padding)
        conv *= alpha_weight
        return tf.nn.relu(conv + biases, name="relu_out")


def BinDense(inputs, shape, name, relu=True):
    with tf.variable_scope(name):
        rweights = tf.get_variable(name="kernel", shape=shape, initializer=None)
        biases = tf.Variable(tf.zeros([shape[-1]]), name="bias")
        dense = math_ops.matmul(inputs, rweights)
        if relu:
            return tf.nn.relu(dense + biases, name="relu_out")
        else:
            return tf.add(dense, biases, name="linear_out")
