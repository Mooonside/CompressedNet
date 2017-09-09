import tensorflow as tf
from differentiable_abs import differentiable_abs
from differntiable_sign import differentiable_sign
from numpy import ones


def XNOR_Conv(inputs, shape, strides, name, padding="VALID"):
    with tf.variable_scope(name):
        rweights = tf.get_variable(name="real_kernel", shape=shape, initializer=None)
        sign_weight = differentiable_sign(rweights)
        sign_input = differentiable_sign(inputs)

        # shape : N HH WW CC
        sign_conv = tf.nn.conv2d(
            input=sign_input,
            filter=sign_weight,
            strides=strides,
            padding=padding
        )
        filter_size = shape[0] * shape[1] * shape[2]

        # shape : CC
        alpha_weight = tf.divide(tf.reduce_sum(differentiable_abs(rweights), axis=[0, 1, 2]),
                                 tf.to_float(filter_size))

        abs_input = differentiable_abs(inputs)
        # shape : N H W 1
        channel_sum = tf.expand_dims(tf.reduce_sum(abs_input, axis=3), -1)
        # shape: KH KW 1 1
        tmp_filter = tf.constant(value=ones(shape=[shape[0], shape[1], 1, 1]), dtype=tf.float32)
        # shape: N HH WW 1
        alpha_input = tf.nn.conv2d(
            input=channel_sum,
            filter=tmp_filter,
            strides=strides,
            padding=padding
        )
        alpha_input *= 1.0 / filter_size

        # N HH WW CC *= N HH WW 1
        sign_conv *= alpha_input
        # N HH WW CC *= CC
        sign_conv *= alpha_weight

        biases = tf.Variable(tf.zeros([shape[-1]]), name="bias")
        return tf.nn.relu(sign_conv + biases, name="relu_out")


def XNOR_Dense(inputs, shape, name, relu=True):
    with tf.variable_scope(name):
        rweights = tf.get_variable(name="real_kernel", shape=shape, initializer=None)
        biases = tf.Variable(tf.zeros([shape[-1]]), name="bias")
        out = tf.add(tf.matmul(inputs, rweights), biases, name="dirc_out")
        if relu:
            return tf.nn.relu(out, name="relu_out")
        else:
            return out

if __name__ == "__main__":
    sess = tf.Session()
    inputs = tf.placeholder(shape=[None, 28, 28, 3], dtype=tf.float32)
    conv1 = XNOR_Conv(
        inputs=inputs,
        shape=[5, 5, 1, 6],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv1',
    )
    sess.run(tf.global_variables_initializer())
    print tf.global_variables()