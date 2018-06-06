import tensorflow as tf
from read_utils import *
import numpy as np


class LSTM(object):
    def __init__(self, sampling=False, n_classes=10, n_steps=28, n_inputs=256, lstm_size=128, n_layers=2,
                 batch_size=128, keep_prob=0.8, learning_rate=0.001):
        # 构造成员变量
        if sampling is True:
            batch_size, n_steps, keep_prob=1,1,1
        else:
            batch_size, n_steps, keep_prob =batch_size, n_steps, keep_prob

        self.n_classes = n_classes
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.lstm_size = lstm_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate

        # 成员方法
        self.build_inputs()
        self.build_lstm()
        self.build_outputs()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver(max_to_keep=2)

    def build_inputs(self):
        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_steps, self.n_inputs), name='inputs')
        self.target = tf.placeholder(tf.int32, shape=(self.batch_size, self.n_steps, self.n_classes), name='targets')

    def build_lstm(self):
        # 创建单个cell
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop
        # 堆叠多层神经元
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.n_layers)])
        # 初始化神经元状态
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)
        # lstm_outputs.shape=(batch_size, n_steps, lstm_szie)

    def build_outputs(self):
        self.logits = tf.layers.dense(self.lstm_outputs,self.n_classes)
        self.prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))

    def build_optimizer(self, grad_clips=1):
        # 使用cliping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clips)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, tvars))

    def train_model(self, batch_generator, epoches=20):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            epoch = 1
            batch = 0

            for x, y, n_batches in batch_generator:
                if batch == 0:
                    print('-----------------------------------------------')
                batch += 1
                loss, _, _ = sess.run([self.loss, self.optimizer, self.final_state], feed_dict={self.inputs: x, self.target: y})
                print('epoch: ' + str(epoch) + '    batch: ' + str(batch) + '/' + str(n_batches) + '  loss=', loss)
                self.saver.save(sess, 'model/model.ckpt', global_step=epoch)
                if batch % n_batches == 0:
                    batch = 0
                    epoch += 1
                if epoch == epoches + 1:
                    break


    def sample(self, n_samples, start_string_arr, converter):
        # 定义samples为输出列表[1,4,6,4]
        samples = [i for i in start_string_arr]
        sess = self.session
        new_state = sess.run(self.initial_state)

        for i in range(n_samples):
            x = converter.int_to_vector(samples[len(samples) - 1])
            x = np.reshape(x, (1, 1, 256))
            preds, new_state = sess.run([self.prediction, self.final_state],
                                        feed_dict={self.inputs: x, self.initial_state: new_state})

            p = np.squeeze(preds)
            samples.append(np.argmax(list(p)) + 1)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def pick_top_n(preds, top_n=5):
        # 去掉维度为1的维度，变成[vocab_size,]
        p = np.squeeze(preds)  # (11980,)
        # choice_index = np.argsort(list(p))[-top_n:] + 1  #选出概率最大的top_n的字的编号
        # c = np.random.choice(choice_index, 1)
        c = np.argmax(p) + 1
        return c