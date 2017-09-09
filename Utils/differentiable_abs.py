from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def differentiable_abs(inputs):
    def _apply(inputs):
        return np.abs(inputs)

    def _gradient(op, grad):
        """ 
            return sign of the input
        """
        orig = op.inputs[0]
        mask = tf.sign(orig)
        return grad * mask

    # produce unique gradient name and register it
    grad_name = "differentiable_abs_grad" + str(np.random.randint(0, 1E+8)*
                                                np.random.randint(0, 1E+8))
    tf.RegisterGradient(grad_name)(_gradient)

    # override py_func gradient to grad_name
    with tf.get_default_graph().gradient_override_map({"PyFunc": grad_name}):
        return tf.py_func(_apply, [inputs], tf.float32)

# if __name__ == "__main__":
#     sess = tf.Session()
#
#     a = tf.Variable(tf.random_normal(shape=[3,3]))
#     b = differentiable_abs(a)
#     c = tf.divide(tf.reduce_sum(b),tf.to_float(tf.size(a)))
#
#     dcdb = tf.gradients(ys=[c],xs=b)
#     dcda = tf.gradients(ys=[c],xs=a)
#
#     sess.run(tf.global_variables_initializer())
#
#     ncnb,ncna = sess.run([dcdb,dcda])
#
#     print(ncnb)
#     print(ncna)