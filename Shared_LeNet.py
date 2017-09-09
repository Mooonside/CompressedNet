from __future__ import division
from Pruned_LeNet import *
from Utils.ThinConv import *
from Utils.weights_cluster import *


class Shared_LeNet(Pruned_LeNet):
    def __init__(self, name="Shared_LeNet"):
        self.model_name = name
        super(Pruned_LeNet, self).__init__(self.model_name)

    # noinspection PyAttributeOutsideInit
    def build_sharegraph(self, sharedict, lr=1e-3, grad_clipping=10):
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

        conv1_mask = tf.placeholder(
            shape=[5, 5, 1, 6],
            dtype=tf.float32,
            name='conv1_mask',
        )

        conv2_mask = tf.placeholder(
            shape=[5, 5, 6, 16],
            dtype=tf.float32,
            name='conv2_mask'
        )

        dense1_mask = tf.placeholder(
            shape=[400, 120],
            name="dense1_mask",
            dtype=tf.float32
        )

        dense2_mask = tf.placeholder(
            shape=[120, 84],
            dtype=tf.float32,
            name="dense2_mask"
        )

        dense3_mask = tf.placeholder(
            shape=[84, 10],
            dtype=tf.float32,
            name="dense3_mask"
        )

        conv1_belonging = sharedict['conv1/kernel:0_belonging']
        conv1_centroid = sharedict['conv1/kernel:0_centroid']
        conv2_belonging = sharedict['conv2/kernel:0_belonging']
        conv2_centroid = sharedict['conv2/kernel:0_centroid']
        dense1_belonging = sharedict['dense1/kernel:0_belonging']
        dense1_centroid = sharedict['dense1/kernel:0_centroid']
        dense2_belonging = sharedict['dense2/kernel:0_belonging']
        dense2_centroid = sharedict['dense2/kernel:0_centroid']
        dense3_belonging = sharedict['dense3/kernel:0_belonging']
        dense3_centroid = sharedict['dense3/kernel:0_centroid']

        conv1 = thin_conv_share(
            inputs=input_layer,
            strides=[1, 1, 1, 1],
            padding='SAME',
            belongings=conv1_belonging,
            centroids=conv1_centroid,
            name='conv1',
            mask=conv1_mask
        )

        self.variable_summaries(conv1, name='conv1')

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2
        )

        # Convolution Layer #2
        conv2 = thin_conv_share(
            inputs=pool1,
            padding='VALID',
            strides=[1, 1, 1, 1],
            belongings=conv2_belonging,
            centroids=conv2_centroid,
            name='conv2',
            mask=conv2_mask
        )

        self.variable_summaries(conv2, name='conv2')
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

        # Dense Layer
        dense1 = thin_dense_share(
            inputs=pool2_flat,
            belongings=dense1_belonging,
            centroids=dense1_centroid,
            name="dense1",
            relu=True,
            mask=dense1_mask
        )
        self.variable_summaries(dense1, name='dense1')

        dense2 = thin_dense_share(
            inputs=dense1,
            belongings=dense2_belonging,
            centroids=dense2_centroid,
            name='dense2',
            relu=True,
            mask=dense2_mask
        )
        self.variable_summaries(dense2, name='dense2')

        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(
            inputs=dense2, rate=dp_rate, training=True)

        # Logits layer
        logits = thin_dense_share(
            inputs=dropout,
            belongings=dense3_belonging,
            centroids=dense3_centroid,
            name='dense3',
            relu=False,
            mask=dense3_mask
        )

        self.prediction = tf.cast(tf.argmax(logits, 1), tf.int32)

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
            onehot_labels=onehot_labels, logits=logits)
        tf.summary.scalar('loss', self.loss)

        self.merge_sum = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        grad_vars = optimizer.compute_gradients(self.loss)
        # for i in grad_vars:
        #     print i
        grads = []
        for grad, var in grad_vars:
            if grad is not None:
                grads.append((tf.clip_by_norm(grad, grad_clipping), var))

        self.train_op = optimizer.apply_gradients(grads, self.global_step)
        print tf.global_variables()

        self.train_writer = tf.summary.FileWriter(self.model_name + '_log/train', self.sess.graph)
        # self.test_writer = tf.summary.FileWriter('LeNet_log/test')
        self.saver = tf.train.Saver(max_to_keep=5)

    def build_graph(self, lr=1e-3, grad_clipping=10):
        sharedict = self.share_weights(cbits=5, dbits=8)
        self.build_sharegraph(sharedict=sharedict, lr=lr, grad_clipping=grad_clipping)

    def share_weights(self, cbits=5, dbits=8):
        var = ['conv1/', 'conv2/', 'dense1/', 'dense2/', 'dense3/']
        sharedict = {}
        path = 'Pruned_weights_np/'
        for i in var:
            weight = np.load(file=path + i + 'kernel:0.npy')
            if u'conv' in i:
                belongings, centroids = weights_cluster(weights=weight, bits=cbits)
                sharedict.update({i+"kernel:0_belonging": belongings})
                sharedict.update({i+"kernel:0_centroid": centroids})
            else:
                belongings, centroids = weights_cluster(weights=weight, bits=dbits)
                sharedict.update({i+"kernel:0_belonging": belongings})
                sharedict.update({i+"kernel:0_centroid": centroids})
        self.logger("fetch shared weights finished!")
        return sharedict

    def load_weight_from_np(self):
        var = [v for v in tf.trainable_variables()]
        path = 'Pruned_weights_np/'
        for i in var:
            if 'share' not in i.name:
                weight = np.load(file=path + i.name + '.npy')
                op = tf.assign(i, weight)
                self.sess.run(op)
        self.logger("successfully restored from numpy!")

    def load_masks_from_np(self):
        var = ['conv1', 'conv2', 'dense1', 'dense2', 'dense3']
        path = 'Pruned_weights_np/'
        mask = {}
        for i in var:
            name = i + "/kernel:0"
            mask.update({name + "_mask": np.load(path + name + '_mask.npy')})
        self.logger("successfully load masks")
        return mask


if __name__ == '__main__':
    model = Shared_LeNet()
    model.build_graph(lr=1e-3, grad_clipping=10)
    model.init_weight()
    masks = model.load_masks_from_np()
    model.train(pruned=True, masks=masks, dp_rate=0.4, epoch_num=20, batch_size=100)
    # model.valid(pruned=True,masks=masks)