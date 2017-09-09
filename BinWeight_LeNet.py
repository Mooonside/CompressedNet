from __future__ import division
from Utils.BinWeights import BinConv, BinDense
from LeNet import *

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


class BinLeNet(LeNet):
    def __init__(self, name='BinLeNet'):
        # fetch data
        self.model_name = name
        super(BinLeNet, self).__init__(self.model_name)

    # noinspection PyAttributeOutsideInit
    def build_graph(self, lr=1e-3, grad_clipping=10):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        input_layer = tf.placeholder(
            shape=[None, 28, 28, 1],
            dtype=tf.float32,
            name='inputs')

        labels = tf.placeholder(
            dtype=tf.int32,
            name='labels')

        dp_rate = tf.placeholder(
            shape=[],
            dtype=tf.float32,
            name='dp_rate'
        )

        conv1 = BinConv(
            inputs=input_layer,
            shape=[5, 5, 1, 6],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv1',
        )

        self.variable_summaries(conv1, name='conv1')

        # Pooling Layer #1
        pool1 = tf.nn.max_pool(
            value=conv1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID"
        )

        # Convolutional import tensorflow.Layer #2
        conv2 = BinConv(
            inputs=pool1,
            shape=[5, 5, 6, 16],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='conv2',
        )
        self.variable_summaries(conv2, name='conv2')

        # Pooling Layer #2
        pool2 = tf.nn.max_pool(
            value=conv2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID"
        )

        # Flatten tensor into a batch of vectors
        pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

        # Dense Layer
        dense1 = BinDense(
            inputs=pool2_flat,
            shape=[400, 120],
            name="dense1",
            relu=True,
        )

        self.variable_summaries(dense1, name='dense_1')

        dense2 = BinDense(
            inputs=dense1,
            shape=[120, 84],
            name='dense2',
            relu=True,
        )

        self.variable_summaries(dense2, name='dense_2')

        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(
            inputs=dense2, rate=dp_rate, training=True)

        # Logits layer
        dense3 = BinDense(
            inputs=dropout,
            shape=[84, 10],
            name='dense3',
            relu=False
        )

        self.prediction = tf.cast(tf.argmax(dense3, 1), tf.int32)

        self.corrected = tf.reduce_sum(
            tf.sign(
                tf.cast(tf.equal(self.prediction, labels), tf.int32)
            )
        )
        self.accuracy = tf.reduce_mean(
            tf.sign(
                tf.cast(tf.equal(self.prediction, labels), tf.float32)
            )
        )
        tf.summary.scalar('accuracy', self.accuracy)

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=dense3)
        tf.summary.scalar('loss', self.loss)

        self.merge_sum = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        grad_vars = optimizer.compute_gradients(self.loss)
        grad_vars = [
            (tf.clip_by_norm(grad, grad_clipping), var)
            for grad, var in grad_vars]

        self.train_op = optimizer.apply_gradients(grad_vars, self.global_step)
        self.train_writer = tf.summary.FileWriter(self.model_name + '_log/train', self.sess.graph)
        # self.test_writer = tf.summary.FileWriter('LeNet_log/test')
        self.saver = tf.train.Saver(max_to_keep=3)

        # print "-" * 50
        # for i in tf.global_variables():
        #     print i.name
        # print "-" * 50


if __name__ == '__main__':
    model = BinLeNet()
    model.build_graph(lr=2e-4, grad_clipping=1)
    model.init_weight()
    # model.load_weight()
    model.train(batch_size=100, epoch_num=50, dp_rate=0.4)
    # model.get_weights()
    # model.valid(mode="test", batch_size=100)
