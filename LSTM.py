import tensorflow as tf
from read_utils import *
import numpy as np


class LSTM(object):
    def __init__(self, sampling=False,
                 n_classes=10,  # 分类类别
                 n_steps=28,  # time_step
                 n_inputs=28,  # input_demension
                 lstm_size=128,  # cell神经元
                 n_layers=3,  # lstm layer
                 batch_size=128,  # batch_size
                 keep_prob=0.8,  # Dropout
                 learning_rate=0.001,  # 学习率
                 epoches=10,
                 batch_generator=None):
        # 构造成员变量
        if sampling is True:
            batch_size = 1
            n_steps = 1
            keep_prob = 1

        else:
            batch_size = batch_size
            n_steps = n_steps
            keep_prob = keep_prob

        self.n_classes = n_classes
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.lstm_size = lstm_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.batch_generator = batch_generator
        self.weights = {'in': tf.Variable(tf.random_normal([self.n_inputs, self.lstm_size])),  # (28, 128)
                        'out': tf.Variable(tf.random_normal([self.lstm_size, self.n_classes]))  # (128, 10)
                        }
        self.biases = {'in': tf.Variable(tf.constant(0.1, shape=[1, self.lstm_size])),  # (1,128)
                       'out': tf.Variable(tf.constant(0.1, shape=[1, self.n_classes]))  # (1,10)
                       }
        # 成员方法
        self.build_inputs()
        self.build_lstm()
        self.build_outputs()
        self.build_loss()
        self.build_optimizer()
        self.accuracy_estimate()
        self.saver = tf.train.Saver(max_to_keep=2)

    def build_inputs(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.n_steps, self.n_inputs), name='inputs')
        self.target = tf.placeholder(tf.int32, shape=(None, self.n_classes), name='targets')
        x_in = tf.reshape(self.inputs,
                          (-1, self.n_inputs))  # x.shape=[None, n_steps, n_inputs]   mlp的输入必须为[-1,n_inputs]
        x_in = tf.matmul(x_in, self.weights['in']) + self.biases['in']  # shape=[128 batch * 256,128]
        self.lstm_inputs = tf.reshape(x_in, [-1, self.n_steps,
                                             self.lstm_size])  # 输出X_in.shape=[128 batch, 28 steps, 128 hidden]

    def build_lstm(self):
        # 创建单个cell
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)

        # 堆叠多层神经元
        cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(self.n_layers)])
        # 初始化神经元状态
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        # lstm_outputs.shape=(batch_size, n_steps, lstm_szie)
        self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs,
                                                                initial_state=self.initial_state)

    def build_outputs(self):
        output = self.lstm_outputs[:, self.lstm_outputs.shape[1] - 1, :]  # shape=[batch_size, lstm_size]
        self.logits = tf.matmul(output, self.weights['out']) + self.biases['out']  # shape = (batch_size, n_classes)
        self.prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))

    def build_optimizer(self, grad_clips=1):
        # 使用cliping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clips)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, tvars))

    def accuracy_estimate(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(tf.reshape(self.target, (-1, self.n_classes)), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def train_model(self, converter):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            epoch = 1
            batch = 0

            for x, y, n_batches in self.batch_generator:
                if batch == 0:
                    print('-----------------------------------------------')
                batch += 1
                # ---------------------构造输入shape---------------------------
                xx = np.zeros((self.batch_size, self.n_steps, self.n_inputs))
                for i in range(xx.shape[0]):
                    for j in range(xx.shape[1]):
                        xx[i, j, :] = converter.int_to_vector(x[i, j])

                yy = np.zeros((self.batch_size, self.n_classes))
                for i in range(yy.shape[0]):
                    yy[i, y[i] - 1] = 1

                # -----------------------------------------------------------------
                loss, _, _ = sess.run([self.loss, self.optimizer, self.final_state],
                                      feed_dict={self.inputs: xx, self.target: yy})
                print('epoch: ' + str(epoch) + '    batch: ' + str(batch) + '/' + str(n_batches) + '  loss=', loss)
                self.saver.save(sess, 'model/model.ckpt', global_step=epoch)
                if batch % n_batches == 0:
                    batch = 0
                    epoch += 1
                if epoch == self.epoches + 1:
                    break

                # #一个epoch验证一次accuracy
                # valid_n_batch=int(mnist.test.images.shape[0]/self.batch_size)
                # acc=[]
                # for j in range(valid_n_batch):
                #     x = mnist.test.images[j*self.batch_size:(j+1)*self.batch_size, :].reshape([-1, self.n_steps, self.n_inputs])
                #     y = mnist.test.labels[j*self.batch_size:(j+1)*self.batch_size, :]
                #     acc.append(sess.run(self.accuracy_estimate(), feed_dict={self.inputs:x, self.target:y}))

                # print("epoch:",epoch+1,"acc:",sum(acc)/len(acc))

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