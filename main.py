import os
from LSTM import *
from read_utils import *

# 超参数设置
tf.app.flags.DEFINE_string("filePath", "trainData/时间简史.txt", "utf8 encoded text file")  # 训练语料文件
tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size")
tf.app.flags.DEFINE_integer("window_size", 1, "window_size")
tf.app.flags.DEFINE_integer("n_steps", 30, "n_steps")
tf.app.flags.DEFINE_integer("n_inputs", 256, "n_inputs")
tf.app.flags.DEFINE_integer("n_classes", 10, "n_classes")
tf.app.flags.DEFINE_integer("lstm_size", 128, "lstm_size")
tf.app.flags.DEFINE_integer("n_layers", 2, "n_layers")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "Dropout")
tf.app.flags.DEFINE_float("epoches", 20, "epoches")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
tf.app.flags.DEFINE_string('checkpoint_path', 'model/', 'checkpoint path')
tf.app.flags.DEFINE_integer('max_length', 30, 'max length to generate')
tf.app.flags.DEFINE_string('start_string', '我', 'use this string to start generating')
Flags = tf.app.flags.FLAGS


def train():
    model_path = os.path.join('model')
    # 如果文件夹不存在，创建文件夹
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    # 生成或者加载模型需要的词典、字典
    dictionary_path = os.path.join('Dictionary', 'converter.pkl')
    if os.path.exists(dictionary_path):
        converter = TextConverter(dictionary_path)  # 创建对象
        print('词典、词表加载完成！')
    else:
        converter = TextConverter()  # 创建对象
        converter.save_to_file(dictionary_path)  # 将词典保存到磁盘
        print('词典、词表创建完成！')

    # 打开训练集数据，将text文本转化为数组,形如[1,4,5,,67,3]
    trainData_path = os.path.join('trainData', '时间简史.pkl')
    if os.path.exists(trainData_path):
        with open(trainData_path, 'rb') as f:
            text_arr = pickle.load(f)
            print("训练数据加载完成！")
    else:
        text_arr = data_loader(Flags.filePath, converter)
        with open(trainData_path, 'wb') as f:
            pickle.dump(text_arr, f)
        print('训练数据读取并固化完成！')

    # batch_generator，输出x,y数据shape=[batch_size, n_steps, n_inputs]
    g = batch_generator(text_arr, Flags.batch_size, Flags.n_steps, Flags.n_inputs,
                        converter.vocab_size(), Flags.window_size, converter=converter)

    # 创建对象lstm
    lstm = LSTM(sampling=False,
                n_classes=converter.vocab_size(),
                n_steps=Flags.n_steps,
                n_inputs=Flags.n_inputs,
                lstm_size=Flags.lstm_size,
                n_layers=Flags.n_layers,
                keep_prob=Flags.keep_prob,
                batch_size=Flags.batch_size,
                learning_rate=Flags.learning_rate,
                )

    lstm.train_model(batch_generator= g, epoches=Flags.epoches)


def sample():
    # 加载模型需要的词典、字典
    dictionary_path = os.path.join('Dictionary', 'converter.pkl')
    if os.path.exists(dictionary_path):
        converter = TextConverter(dictionary_path)  # 创建对象
        print('词典、词表加载完成！')
    else:
        converter = TextConverter()  # 创建对象
        converter.save_to_file(dictionary_path)  # 将词典保存到磁盘
        print('词典、词表创建完成！')

    # 更新模型参数路径
    if os.path.isdir(Flags.checkpoint_path):
        Flags.checkpoint_path = tf.train.latest_checkpoint(Flags.checkpoint_path)

    # 创建对象lstm
    lstm = LSTM(sampling=True, n_classes=converter.vocab_size(), n_inputs=Flags.n_inputs,
                lstm_size=Flags.lstm_size, n_layers=Flags.n_layers)
    # 加载模型参数
    lstm.load(Flags.checkpoint_path)

    # 将起始输入转化为数组
    start_string_arr = []
    for i in Flags.start_string:
        start_string_arr.append(converter.word_to_int(i))
    # print(start_string_arr)#[3535, 3390]
    # 通过起始字符串返回模型预测的数组
    text_arr = lstm.sample(Flags.max_length, start_string_arr, converter)

    out_put = [converter.int_to_word_table[i] for i in text_arr]
    print(''.join(out_put))


if __name__ == "__main__":
    #train()
    sample()