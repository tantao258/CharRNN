import numpy as np
import copy
import pickle
import os
from gensim.models import Word2Vec


class TextConverter(object):
    def __init__(self, filename=None):
        # filename is not None，加载实现保存好的词典数据
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
                self.word_to_int_table = pickle.load(f)
                self.int_to_word_table = pickle.load(f)
                self.word_to_vector = pickle.load(f)



        # filename is None，生成词典并固化在磁盘
        else:
            model = Word2Vec.load('./word2vec/novel_word2vector.model')
            vocab = []
            for (i, j) in model.wv.vocab.items():
                if len(i) == 1:
                    vocab.append(i)
            self.vocab = vocab
            # 对词进行数字编码
            self.word_to_int_table = {c: i for i, c in enumerate(self.vocab, start=1)}  # {'大': 1, '好': 2, '人': 0}
            self.word_to_vector = {word: model[word] for word in self.vocab}  # {word:vector}
            self.int_to_word_table = dict(enumerate(self.vocab, start=1))  # {0: '人', 1: '大', 2: '好'}

    # @property
    def vocab_size(self):
        return len(self.vocab)

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return 0

    def int_to_word(self, index):
        return self.int_to_word_table[index]

    def int_to_vector(self, index):
        return self.word_to_vector[self.int_to_word(index)]

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))  # 通过词语去查找对应的数字
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))  # 通过数字去查找对应的词语
        return "".join(words)  # join拼接字符串

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
            pickle.dump(self.word_to_int_table, f)
            pickle.dump(self.int_to_word_table, f)
            pickle.dump(self.word_to_vector, f)


def data_loader(filePath, converter):
    text_arr = []  # 将所有文本读成一行，并转化为对应索引
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            for i in line:
                if converter.word_to_int(i) != 0:
                    text_arr.append(converter.word_to_int(i))
    return text_arr


def batch_generator(arr, batch_size, n_steps, window_size=1):
    train_arr = []
    target_arr = []
    i = 0
    while (i * window_size + n_steps + 1) < len(arr):
        train_arr.append(arr[i * window_size: i * window_size + n_steps])
        target_arr.append(arr[i * window_size + n_steps])
        i += 1
    train_arr = np.array(train_arr)
    target_arr = np.reshape(np.array(target_arr), (-1,))

    while True:
        for n in range(0, train_arr.shape[0], batch_size):
            n_batches = int(train_arr.shape[0] / batch_size)
            if (n + 1) * batch_size < train_arr.shape[0]:
                x = train_arr[n:n + batch_size, ]
                y = target_arr[n:n + batch_size, ]

            yield x, y, n_batches
