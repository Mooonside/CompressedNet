from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# from Utils.differentiable_abs import differentiable_abs
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import math_ops
# @ops.RegisterGradient("BinAprx")
# def _Bin_Trans_grad(op, grad):
#   orig_weight = op.inputs[0]
#   shape = array_ops.shape(orig_weight)
#   size = tf.to_float(math_ops.reduce_prod(shape))
#   alpha = math_ops.reduce_sum(math_ops.abs(orig_weight))
#   alpha = math_ops.divide(alpha,size)
#   grad_mask = tf.to_float(math_ops.less(math_ops.abs(orig_weight),1.0))
#   orig_grad = grad * grad_mask * alpha + math_ops.divide(1,size)
#   return [orig_grad]  # List of one Tensor, since we have one input
# 
# bin_aprx_mod = tf.load_op_library('./bin_aprx.so')
# bin_aprx = bin_aprx_mod.bin_aprx


def differentiable_sign(inputs):
    def _apply(inputs):
        return np.sign(inputs)

    def _gradient(op, grad):
        """ 
            remove the grad w.r.t to whom is greater than 1 
        """
        orig = op.inputs[0]
        mask = tf.to_float(tf.less_equal(tf.abs(orig), tf.constant(value=1, dtype=tf.float32)))
        return grad * mask

    # produce unique gradient name and register it
    grad_name = "differentiable_sign_grad_" + str(np.random.randint(0, 1E+8)*
                                                  np.random.randint(0, 1E+8))
    tf.RegisterGradient(grad_name)(_gradient)

    # override py_func gradient to grad_name
    with tf.get_default_graph().gradient_override_map({"PyFunc": grad_name}):
        return tf.py_func(_apply, [inputs], tf.float32)


# debug
# if __name__ == "__main__":
#     sess = tf.Session()
#
#     image = tf.Variable(tf.random_normal(shape=[1,3,3,3]))
#     a = tf.Variable(tf.random_normal(shape=[3,3,3,1]))
#
#     sign_a = differentiable_sign(a)
#     alpha_a = tf.divide(tf.reduce_sum(differentiable_abs(a)),tf.to_float(tf.size(a)))
#     bin_a = sign_a * alpha_a
#
#     with tf.device('/cpu:0'):
#          true_a =  bin_aprx(a)
#
#     c1 = tf.nn.conv2d(input=image,
#                  filter=true_a,
#                  strides=[1,1,1,1],
#                  padding="VALID")
#
#     c2 = tf.nn.conv2d(input=image,
#                 filter=bin_a,
#                 strides=[1,1,1,1],
#                 padding="VALID",
#     )
#
#     d1 = tf.gradients(ys=[c1],xs=a)
#     d2 = tf.gradients(ys=[c2],xs=a)
#
#     sess.run(tf.global_variables_initializer())
#
#     ba = sess.run([c1])
#     ta = sess.run([c2])
#
#     print( np.sum(np.asarray(ba)-np.asarray(ta)) )
