# coding=utf-8
"""
用自己创建的 sin 曲线预测一条 cos 曲线
PS：深度学习中经常看到epoch、 iteration和batchsize，下面按自己的理解说说这三个的区别：
（1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
（2）iteration：1个iteration等于使用batchsize个样本训练一次；
（3）epoch：1个epoch等于使用训练集中的全部样本训练一次；
举个例子，训练集有1000个样本，batchsize=10，那么：
训练完整个样本集需要：
100次iteration，1次epoch。
"""
import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('PS')
import matplotlib.pyplot as plt

# 定义超参数
BATCH_START = 0  # 建立 batch data 时候的 index
TIME_STEPS = 20  # time_steps也就是n_steps, 等于序列的长度
BATCH_SIZE = 50  # 批次的大小
INPUT_SIZE = 1  # sin 数据输入 size
OUTPUT_SIZE = 1  # cos 数据输出 size
CELL_SIZE = 10  # 隐藏层规模
LR = 0.006  # 学习率


# 数据生成
# 生成一个批次大小的数据的 get_batch function:
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs的shape是（50batch, 20steps）
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (
            10 * np.pi)  # 定义x
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # 返回seq，res 的shape(batch, step, 1), xs的shape为(batch_size, time_steps)
    # 一般像这种[:, :, np.newaxis]叫做扩维技术，从2为变成3维, 扩的维数为1。
    # seq[:,  :,  np.newaxis].shape=（50, 20, 1）
    # plt.plot (xs,  seq,  'r-',  xs,  res,  'b-')
    # plt.show ()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


# 1. 使用tf.Variable()的时候，tf.name_scope()和tf.variable_scope() 都会给 Variable 和 op 的 name属性加上前缀。
# 2. 使用tf.get_variable()的时候，tf.name_scope()就不会给 tf.get_variable()创建出来的Variable加前缀。

# 定义 LSTM 的主体结构
class LSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.output_size = output_size
        # 初始化几个函数
        with tf.name_scope("inputs"):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, self.input_size], name="xs")
            self.ys = tf.placeholder(tf.float32, [None, n_steps, self.output_size], name="ys")
        with tf.variable_scope("in_hidden"):
            self.add_input_layer()
        with tf.variable_scope("LSTM"):
            self.add_cell()
        with tf.variable_scope("out_hidden"):
            self.add_output_layer()
        with tf.name_scope("cost"):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    # 定义三个变量
    def ms_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))
        # return tf.nn.l2_loss(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1]))

    def _weight_variable(self, shape, name="weights"):
        initializer = tf.random_normal_initializer(mean=0, stddev=1, )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name="biases"):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    # 接下来定义几个函数
    def add_input_layer(self):
        # 应该我们只能在二维数据上矩阵相乘，计算logits, 之后在reshape成3维。以下同理
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # 输入shape(batch_size*n_step, input_size)
        # l_in_x = tf.Print(l_in_x, [l_in_x], summarize=1000, name='l_in_x')  # shape=[1000, 1]
        Ws_in = self._weight_variable([self.input_size, self.cell_size])   # 权重的shape(in_size, cell_size)
        bs_in = self._bias_variable([self.cell_size, ])  # 参数是一维矩阵, # 偏置的shape (cell_size, )
        # l_in_y = (batch * n_steps , cell_size)
        with tf.name_scope("Wx_puls_b"):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y-->>=(batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):  # 如果l_in_y的shape是(n_steps, batch, cell_size)的话，则对应的time_major=True
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("initial_state"):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32, )
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                                     self.l_in_y, initial_state=self.cell_init_state,
                                                                     time_major=False)

    def add_output_layer(self):
        # shape=(batch * n_steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name="2_2D")
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        with tf.name_scope("Wx_plus_b"):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out  # shape=(batch*n_steps, output_size)

    def compute_cost(self):
        # 计算一个batch内每一样本的loss # 平铺一下维数
        # self.label = tf.reshape(self.ys, [-1])
        # self.prediction = tf.reshape(self.pred,[-1])
        # losses = tf.reduce_mean(tf.square(self.pred-self.label))
        # losses = tf.nn.l2_loss(self.pred - self.label)
        # train_op = tf.train.AdamOptimizer(1e-2).minimize(losses)
        # 正确的loss函数
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name="reshape_pred")],
            [tf.reshape(self.ys, [-1], name="reshape_target")],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],average_across_timesteps=True,
            softmax_loss_function=self.ms_error, name="losses")

        with tf.name_scope("average_cost"):
            # 计算每一个batch的平均loss，因为梯度更新是在计算一个batch的平均误差的基础上进行更新的
            self.cost = tf.div(tf.reduce_sum(losses, name="losses_sum"), self.batch_size, name="average_cost")
            tf.summary.scalar("cost", self.cost)


if __name__ == "__main__":
    model = LSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)

    if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on
    # 在terminal中输入$ tensorboard --logdir='logs'，让后在浏览器中Chrome (http://0.0.0.0:6006/)查看tensorboard
    plt.ion()
    plt.show()
    for i in range(200):  # 训练200次，训练一次一个batch
        seq, res, xs = get_batch()  # 此时的seq, res都是3维数组，shape=(batch, time_steps, 1), 这里的1就是input_size
        if i == 0:
            # 创建初始状态，这里就开始体现类的优势了，直接调用里面的xs, ys,
            feed_dict = {model.xs: seq, model.ys: res}
        else:
            # 用最后的state代替初始化的state
            feed_dict = {model.xs: seq, model.ys: res, model.cell_init_state: state}
        _, cost, state, pred = sess.run([model.train_op, model.cost, model.cell_final_state, model.pred],
                                        feed_dict=feed_dict)
        # 输出值和带入的参数顺序一一对应，cost对应model.cost, 等等

        # xs[0, :], 表示的是一个batch里面的第一个序列，因为xs是由np.arange()函数生成的，
        # 所以xs在对于每一个batch来说，同一个batch里面的每个序列都是一样的
        # 例如xs的batch_size=3, time_step=4, [[0, 1, 2, 3],
        #                                 [0, 1, 2, 3],
        #                                 [0, 1, 2, 4]], shape=(3, 4)
        # res[0].flatten()表示的是一个batch里面的第一个序列，序列长度为time_steps * 1
        plt.plot(xs[0, :], res[0].flatten(), "r", xs[0, :], pred.flatten()[:TIME_STEPS], "b--")
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:  # 每训练20个批次来打印一次当时的cost
            print("cost:", round(cost, 4))  # 输出每一个batch的平均cost，约到零后面4位小数点
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
            tf.convert_to_tensor




