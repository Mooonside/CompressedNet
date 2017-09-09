import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

@ops.RegisterGradient("BinAprx")
def _Bin_Trans_grad(op, grad):
  orig_weight = op.inputs[0]
  shape = array_ops.shape(orig_weight)
  size = tf.to_float(math_ops.reduce_prod(shape))
  alpha = math_ops.reduce_sum(math_ops.abs(orig_weight))
  alpha = math_ops.divide(alpha,size)
  grad_mask = tf.to_float(math_ops.less(math_ops.abs(orig_weight),1.0))
  orig_grad = grad * grad_mask * alpha + math_ops.divide(1,size)
  return [orig_grad]  # List of one Tensor, since we have one input


config = tf.ConfigProto(log_device_placement = True)
config.graph_options.optimizer_options.opt_level = -1
# print alpha * sign
module = tf.load_op_library('./bin_aprx.so')
with tf.Session(config=config) as sess:
    a = tf.Variable(tf.random_normal(shape=[3, 3]), name='a', dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    with tf.device('/gpu:0'):
        gpuout =  module.bin_aprx(a)

    input = a.eval(session=sess)
    gpu = gpuout.eval(session=sess)
    npu = np.sum(np.abs(input)) / 9.0

    tgrad = (np.abs(input) <= 1) * npu
    tgrad += 1 / 9.0

    npu = npu * np.sign(input)

    print np.sum(gpu-npu)

    grad = tf.gradients(ys=[gpuout], xs=a)
    # print grad
    print np.sum(grad[0].eval(session=sess) - tgrad)