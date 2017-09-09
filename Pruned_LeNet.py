from __future__ import division
from LeNet import *
from Utils.ThinConv import *
import numpy as np


class Pruned_LeNet(LeNet):
    def __init__(self, name="Pruned_LeNet"):
        self.model_name = name
        super(Pruned_LeNet, self).__init__(self.model_name)

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

        conv1_mask = tf.placeholder(
            shape=[5, 5, 1, 6],
            dtype=tf.float32,
            name='conv1_mask'
        )

        conv2_mask = tf.placeholder(
            shape=[5, 5, 6, 16],
            dtype=tf.float32,
            name='conv2_mask'
        )

        dense1_mask = tf.placeholder(
            shape=[400, 120],
            dtype=tf.float32,
            name="dense1_mask"
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

        conv1 = thin_conv2D(
            inputs=input_layer,
            shape=[5, 5, 1, 6],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv1',
            mask=conv1_mask)

        self.variable_summaries(conv1, name='conv1')

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2
        )

        # Convolution Layer #2
        conv2 = thin_conv2D(
            inputs=pool1,
            shape=[5, 5, 6, 16],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='conv2',
            mask=conv2_mask
        )

        self.variable_summaries(conv2, name='conv2')
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

        # Dense Layer
        dense1 = thin_dense(
            inputs=pool2_flat,
            shape=[400, 120],
            name="dense1",
            relu=True,
            mask=dense1_mask
        )
        self.variable_summaries(dense1, name='dense1')

        dense2 = thin_dense(
            inputs=dense1,
            shape=[120, 84],
            name='dense2',
            relu=True,
            mask=dense2_mask
        )
        self.variable_summaries(dense2, name='dense2')

        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(
            inputs=dense2, rate=dp_rate, training=True)

        # Logits layer
        logits = thin_dense(
            inputs=dropout,
            shape=[84, 10],
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
        grad_vars = [
            (tf.clip_by_norm(grad, grad_clipping), var)
            for grad, var in grad_vars]
        self.train_op = optimizer.apply_gradients(grad_vars, self.global_step)
        print tf.global_variables()

        self.train_writer = tf.summary.FileWriter(self.model_name + '_log/train', self.sess.graph)
        # self.test_writer = tf.summary.FileWriter('LeNet_log/test')
        self.saver = tf.train.Saver(max_to_keep=5)

    def get_train_batch(self, idx=0, batch_size=100):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        upper = self.train_data.shape[0]
        if end > upper:
            end = upper
        batch = {
            'inputs:0': self.train_data[start:end, :],
            'labels:0': self.train_labels[start:end],
        }
        return batch

    def get_test_batch(self, idx=0, pruned=False, batch_size=100):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        upper = self.test_data.shape[0]
        if end > upper:
            end = upper
        batch = {
            'inputs:0': self.test_data[start:end, :],
            'labels:0': self.test_labels[start:end],
        }
        return batch

    # noinspection PyTypeChecker
    def get_weight_masks(self, thre=1e-1):
        var = [v for v in tf.trainable_variables()]
        wdict = {}
        lcnt = 0
        tcnt = 0
        for i in var:
            if u'bias' not in i.name:
                weight = self.sess.run(i)
                mask = np.abs(weight) > thre
                lcnt += np.sum(mask)
                tcnt += weight.size
                print "%d %d" % (np.sum(mask), weight.size)
                wdict.update({i.name + "_mask": mask})
        self.logger("total ratio:{:.4f}".format(lcnt/tcnt))
        return wdict

    # noinspection PyAttributeOutsideInit
    def train(self, dp_rate=0.4, batch_size=100, epoch_num=100, pruned=True, thre=1e-3, masks=None):
        # initialize
        self.best_val_acc = 0.0
        iter_per_epoch = self.train_data.shape[0] // batch_size
        if not self.train_data.shape[0] % batch_size == 0:
            iter_per_epoch += 1

        if pruned and masks is None:
            masks = self.get_weight_masks(thre=thre)

        gcounter = 0
        for i in range(epoch_num):
            self.logger("-"*30 + "EPOCH {}".format(i) + "-"*30)
            self.shuffle()
            for j in range(iter_per_epoch):
                batch_data = self.get_train_batch(j)
                batch_data.update({"dp_rate:0": dp_rate})
                if pruned:
                    batch_data.update({"conv1_mask:0": masks[u'conv1/kernel:0_mask']})
                    batch_data.update({"conv2_mask:0": masks[u'conv2/kernel:0_mask']})
                    batch_data.update({"dense1_mask:0": masks[u'dense1/kernel:0_mask']})
                    batch_data.update({"dense2_mask:0": masks[u'dense2/kernel:0_mask']})
                    batch_data.update({"dense3_mask:0": masks[u'dense3/kernel:0_mask']})
                else:
                    batch_data.update({"conv1_mask:0": np.ones([5, 5, 1, 6])})
                    batch_data.update({"conv2_mask:0": np.ones([5, 5, 6, 16])})
                    batch_data.update({"dense1_mask:0": np.ones([400, 120])})
                    batch_data.update({"dense2_mask:0": np.ones([120, 84])})
                    batch_data.update({"dense3_mask:0": np.ones([84, 10])})

                if gcounter % (batch_size * 20) == 0 and not gcounter == 0:
                    loss, acc, _, summary = self.sess.run(
                        [self.loss, self.accuracy, self.train_op, self.merge_sum],
                        feed_dict=batch_data
                    )
                    self.train_writer.add_summary(summary, j)
                    self.logger("batch loss:{:.3f} batch acc:{:.3f}".format(loss, acc))
                else:
                    loss, acc, _ = self.sess.run([self.loss, self.accuracy, self.train_op], feed_dict=batch_data)
                    # self.logger("batch loss:{:.2f} batch acc:{:.2f}".format(loss,acc))

                if gcounter % (batch_size*400) == 0:
                    self.logger("-"*50)
                    val_loss, val_acc = self.valid(pruned=True, thre=thre, masks=masks)
                    self.logger("-"*50)
                    self.early_stopping(val_acc)
                gcounter += batch_data['labels:0'].shape[0]

    def valid(self, mode="test", batch_size=100, pruned=True, thre=1e-2, masks=None):
        if pruned and masks is None:
            masks = self.get_weight_masks(thre)

        if mode == "test":
            batch_num = self.test_data.shape[0] // batch_size
            if not self.test_data.shape[0] % batch_size == 0:
                batch_num += 1

            total_corrected, total_loss = 0, 0
            for i in range(batch_num):
                batch_data = self.get_test_batch(i, batch_size=batch_size)
                batch_data.update({"dp_rate:0": 0.0})
                if pruned:
                    batch_data.update({"conv1_mask:0": masks[u'conv1/kernel:0_mask']})
                    batch_data.update({"conv2_mask:0": masks[u'conv2/kernel:0_mask']})
                    batch_data.update({"dense1_mask:0": masks[u'dense1/kernel:0_mask']})
                    batch_data.update({"dense2_mask:0": masks[u'dense2/kernel:0_mask']})
                    batch_data.update({"dense3_mask:0": masks[u'dense3/kernel:0_mask']})
                else:
                    batch_data.update({"conv1_mask:0": np.ones([5, 5, 1, 6])})
                    batch_data.update({"conv2_mask:0": np.ones([5, 5, 6, 16])})
                    batch_data.update({"dense1_mask:0": np.ones([400, 120])})
                    batch_data.update({"dense2_mask:0": np.ones([120, 84])})
                    batch_data.update({"dense3_mask:0": np.ones([84, 10])})
                samples = batch_data['labels:0'].shape[0]
                a, loss, corrected = self.sess.run([self.prediction, self.loss, self.corrected], feed_dict=batch_data)
                total_corrected += corrected
                total_loss += loss * samples

            val_acc = total_corrected / self.test_data.shape[0]
            val_loss = total_loss / self.test_data.shape[0]
            # val_acc = total_corrected / self.train_data.shape[0]
            # val_loss = total_loss / self.train_data.shape[0]
            self.logger("Val acc : {:.4f}.\tVal Loss : {:.4f}".
                        format(val_acc, val_loss))
            return val_loss, val_acc
        else:
            batch_num = self.train_data.shape[0] // batch_size
            if not self.train_data.shape[0] % batch_size == 0:
                batch_num += 1

            total_corrected, total_loss = 0, 0
            for i in range(batch_num):
                batch_data = self.get_train_batch(i, batch_size=batch_size)
                batch_data.update({"dp_rate:0": 0.0})
                if pruned:
                    batch_data.update({"conv1_mask:0": masks[u'conv1/kernel:0_mask']})
                    batch_data.update({"conv2_mask:0": masks[u'conv2/kernel:0_mask']})
                    batch_data.update({"dense1_mask:0": masks[u'dense1/kernel:0_mask']})
                    batch_data.update({"dense2_mask:0": masks[u'dense2/kernel:0_mask']})
                    batch_data.update({"dense3_mask:0": masks[u'dense3/kernel:0_mask']})

                else:
                    batch_data.update({"conv1_mask:0": np.ones([5, 5, 1, 6])})
                    batch_data.update({"conv2_mask:0": np.ones([5, 5, 6, 16])})
                    batch_data.update({"dense1_mask:0": np.ones([400, 120])})
                    batch_data.update({"dense2_mask:0": np.ones([120, 84])})
                    batch_data.update({"dense3_mask:0": np.ones([84, 10])})

                samples = batch_data['labels:0'].shape[0]
                a, loss, corrected = self.sess.run([self.prediction, self.loss, self.corrected], feed_dict=batch_data)
                total_corrected += corrected
                total_loss += loss * samples

            val_acc = total_corrected / self.train_data.shape[0]
            val_loss = total_loss / self.train_data.shape[0]
            # val_acc = total_corrected / self.train_data.shape[0]
            # val_loss = total_loss / self.train_data.shape[0]
            self.logger("Train acc : {:.2f}.\tTrain Loss : {:.2f}".
                        format(val_acc, val_loss))
            return val_loss, val_acc

    def save_weight_to_np(self, mask):
        var = [v for v in tf.trainable_variables()]
        path = 'Pruned_weights_np/'
        for i in var:
            if u'bias' not in i.name:
                np.save(path + i.name, self.sess.run(i) * mask[i.name+"_mask"])
            else:
                np.save(path + i.name, self.sess.run(i))

    def load_weight_from_np(self):
        var = [v for v in tf.trainable_variables()]
        path = 'Pruned_weights_np/'
        for i in var:
            weight = np.load(file=path + i.name + '.npy')
            op = tf.assign(i, weight)
            self.sess.run(op)
        self.logger("successfully restored from numpy!")

    def save_masks_to_np(self, mask):
        var = [v for v in tf.trainable_variables()]
        path = 'Pruned_weights_np/'
        for i in var:
            if u'bias' not in i.name:
                np.save(path + i.name + '_mask', mask[i.name+"_mask"])
        self.logger("successfully saved masks")

    def load_masks_from_np(self):
        var = [v for v in tf.trainable_variables()]
        path = 'Pruned_weights_np/'
        mask = {}
        for i in var:
            if u'bias' not in i.name:
                mask.update({i.name+"_mask": np.load(path + i.name + '_mask.npy')})
        self.logger("successfully load masks")
        return mask

if __name__ == '__main__':
    model = Pruned_LeNet()
    model.build_graph(lr=1e-3, grad_clipping=10)
    model.init_weight()
    model.load_weight_from_np()
    # mask = model.load_masks_from_np()
    model.valid(pruned=False)
    # model.valid(pruned=True, masks=mask)

    # model.load_weight(path='LeNet_weights')
    # th = 1e-2
    # while th <= 2e-1:
    #     print "-"*30 + ("th=%f" % th) + "-"*30
    #     model.valid(pruned=True, thre=th)
    #     th += 1e-2

    # model.load_weight(path='LeNet_weights')
    # th = 13e-2
    # mask = model.get_weight_masks(th)
    # model.save_masks_to_np(mask)
    # model.load_weight()
    # model.valid(pruned=True, masks=mask)
    # model.train(pruned=True, thre=th,masks=mask)
    # model.save_weight_to_np(mask=mask)
