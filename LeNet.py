from __future__ import division
import tensorflow as tf
from Utils.read_mnist import *
import os


class LeNet(object):
    def __init__(self, name="LeNet"):
        # fetch data
        self.train_data = read_image('./mnist_dataset/train-images.idx3-ubyte')
        self.train_labels = read_label('./mnist_dataset/train-labels.idx1-ubyte')
        self.test_data = read_image('./mnist_dataset/t10k-images.idx3-ubyte')
        self.test_labels = read_label('./mnist_dataset/t10k-labels.idx1-ubyte')
        self.logger('Fetch Dataset Finished')

        # model setting
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.patience = 1000
        self.model_name = name
        self.sess = tf.Session(config=config)
        self.weight_path = self.model_name + '_weights'
        self.create_folder()

    @staticmethod
    def logger(mesg):
        print(mesg)

    def get_weights(self):
        var = [v for v in tf.trainable_variables()]
        wdict = {}
        for i in var:
            wdict.update({i.name: self.sess.run(i)})
        return wdict

    @staticmethod
    def variable_summaries(var, name):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def shuffle(self):
        sam_num = self.train_data.shape[0]
        choice = np.random.choice(sam_num, sam_num, replace=False)
        self.train_data = self.train_data[choice, :]
        self.train_labels = self.train_labels[choice]
        self.logger('Shuffle!')

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

    def get_test_batch(self, idx=0, batch_size=100):
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

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=6,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
            name='conv1')
        self.variable_summaries(conv1, name='conv1')

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)

        # Convolutional import tensorflow.Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=16,
            strides=1,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu,
            name='conv2')
        self.variable_summaries(conv2, name='conv2')

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

        # Dense Layer
        dense_1 = tf.layers.dense(
            inputs=pool2_flat,
            units=120,
            activation=tf.nn.relu,
            name='dense1')
        self.variable_summaries(dense_1, name='dense_1')

        dense_2 = tf.layers.dense(
            inputs=dense_1,
            units=84,
            activation=tf.nn.relu,
            name='dense2')
        self.variable_summaries(dense_2, name='dense_2')

        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(
            inputs=dense_2, rate=dp_rate, training=True)

        # Logits layer
        logits = tf.layers.dense(
            inputs=dropout,
            units=10,
            name='dense3'
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

        self.train_writer = tf.summary.FileWriter(self.model_name + '_log/train', self.sess.graph)
        # self.test_writer = tf.summary.FileWriter('LeNet_log/test')
        self.saver = tf.train.Saver(max_to_keep=3)

        print "-" * 50
        for i in tf.global_variables():
            print i.name
        print "-" * 50

    def load_weight(self, path=None):
        # self.sess.run(tf.global_variables_initializer())
        if path is None:
            path = self.weight_path

        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt is not None:
            self.logger('Loading from {}.'.format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.logger('No models found in {}!'.format(path))

    def init_weight(self):
        self.sess.run(tf.global_variables_initializer())

    def save_weight(self, val_acc):
        path = self.saver.save(
            self.sess,
            self.model_name + "_weights/val_acc-{:.3f}.models".format(val_acc),
            global_step=self.global_step)
        self.logger("Save models to {}.".format(path))

    # noinspection PyAttributeOutsideInit
    def early_stopping(self, val_acc):
        if val_acc > self.best_val_acc:
            self.patience = self.patience
            self.best_val_acc = val_acc
            self.save_weight(val_acc)
        elif self.patience == 1:
            self.logger("End Training")
            exit(0)
        else:
            self.patience -= 1
            self.logger("Remaining/Patience : {}/{} .".format(self.patience, self.patience))

    def valid(self, mode="test", batch_size=100):
        if mode == "test":
            batch_num = self.test_data.shape[0] // batch_size
            if not self.test_data.shape[0] % batch_size == 0:
                batch_num += 1

            total_corrected, total_loss = 0, 0

            for i in range(batch_num):
                data = self.get_test_batch(i, batch_size)
                data.update({"dp_rate:0": 0.0})
                samples = data['labels:0'].shape[0]
                a, loss, corrected = self.sess.run([self.prediction, self.loss, self.corrected], feed_dict=data)
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
                data = self.get_train_batch(i, batch_size)
                data.update({"dp_rate:0": 0.0})
                samples = data['labels:0'].shape[0]
                a, loss, corrected = self.sess.run([self.prediction, self.loss, self.corrected], feed_dict=data)
                total_corrected += corrected
                total_loss += loss * samples

            val_acc = total_corrected / self.train_data.shape[0]
            val_loss = total_loss / self.train_data.shape[0]
            # val_acc = total_corrected / self.train_data.shape[0]
            # val_loss = total_loss / self.train_data.shape[0]
            self.logger("Train acc : {:.2f}.\tTrain Loss : {:.2f}".
                        format(val_acc, val_loss))
            return val_loss, val_acc

    # noinspection PyAttributeOutsideInit
    def train(self, batch_size=100, epoch_num=10, dp_rate=0.4):
        # initialize
        self.best_val_acc = 0.0
        iter_per_epoch = self.train_data.shape[0] // batch_size
        if not self.train_data.shape[0] % batch_size == 0:
            iter_per_epoch += 1

        gcounter = 0
        for i in range(epoch_num):
            self.logger("-" * 30 + "EPOCH {}".format(i) + "-" * 30)
            self.shuffle()
            for j in range(iter_per_epoch):
                batch_data = self.get_train_batch(j, batch_size)
                batch_data.update({"dp_rate:0": dp_rate})
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

                if gcounter % (batch_size * 400) == 0:
                    self.logger("-" * 50)
                    val_loss, val_acc = self.valid()
                    self.logger("-" * 50)
                    self.early_stopping(val_acc)
                gcounter += batch_data['labels:0'].shape[0]

    def create_folder(self):
        if os.path.exists(self.model_name + '_log'):
            pass
        else:
            os.mkdir(self.model_name + '_log')

        if os.path.exists(self.model_name + '_log/train'):
            pass
        else:
            os.mkdir(self.model_name + '_log/train')

        if os.path.exists(self.model_name + "_weights"):
            pass
        else:
            os.mkdir(self.model_name + "_weights")


if __name__ == '__main__':
    model = LeNet()
    model.build_graph(lr=1e-3, grad_clipping=10)
    model.init_weight()
    model.load_weight()
    # model.train(batch_size=100, epoch_num=50, dp_rate=0.4)
    # model.get_weights()
    model.valid(mode="test", batch_size=100)
