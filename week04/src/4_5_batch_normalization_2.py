import os
import tensorflow as tf
"""
batch normalization (BN) 就是以batch为单位进行操作，
减去 batch 内样本均值，除以 batch 内样本的标准差，(normalize)
最后进行平移和缩放，其中缩放参数 r 和平移参数 beta 都是可学习的参数 (scale and shift)
"""
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class NeuralNetWork():
    def __init__(self, initial_weights, activation_fn, use_batch_norm):
        """
        初始化网络对象
        :param initial_weights: 权重初始化值，是一个list，list中每一个元素是一个权重矩阵
        :param activation_fn: 隐层激活函数
        :param user_batch_norm: 是否使用batch normalization
        """
        self.use_batch_norm = use_batch_norm
        self.name = "With Batch Norm" if use_batch_norm else "Without Batch Norm"

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # 存储训练准确率
        self.training_accuracies = []

        self.build_network(initial_weights, activation_fn)

    def build_network(self, initial_weights, activation_fn):
        """
        构建网络图
        :param initial_weights: 权重初始化，是一个list
        :param activation_fn: 隐层激活函数
        """
        self.input_layer = tf.placeholder(tf.float32, [None, initial_weights[0].shape[0]])
        layer_in = self.input_layer

        # 前向计算（不计算最后输出层）
        for layer_weights in initial_weights[:-1]:
            layer_in = self.fully_connected(layer_in, layer_weights, activation_fn)

        # 输出层
        self.output_layer = self.fully_connected(layer_in, initial_weights[-1])

    def fully_connected(self, layer_in, layer_weights, activation_fn=None):
        """
        抽象出的全连接层计算
        """
        # 如果使用BN与激活函数
        if self.use_batch_norm and activation_fn:
            weights = tf.Variable(layer_weights)
            linear_output = tf.matmul(layer_in, weights)

            # 调用BN接口
            batch_normalized_output = tf.layers.batch_normalization(linear_output, training=self.is_training)

            return activation_fn(batch_normalized_output)
        # 如果不使用BN或激活函数（即普通隐层）
        else:
            weights = tf.Variable(layer_weights)
            bias = tf.Variable(tf.zeros([layer_weights.shape[-1]]))
            linear_output = tf.add(tf.matmul(layer_in, weights), bias)

            return activation_fn(linear_output) if activation_fn else linear_output

    def train(self, sess, learning_rate, training_batches, batches_per_validate_data, save_model=None):
        """
        训练模型
        :param sess: TensorFlow Session
        :param learning_rate: 学习率
        :param training_batches: 用于训练的batch数
        :param batches_per_validate_data: 训练多少个batch对validation数据进行一次验证
        :param save_model: 存储模型
        """

        # 定义输出label
        labels = tf.placeholder(tf.float32, [None, 10])

        # 定义损失函数
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                                  logits=self.output_layer))

        # 准确率
        correct_prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #
        if self.use_batch_norm:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

        else:
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

        # 显示进度条
        for i in tqdm.tqdm(range(training_batches)):
            batch_x, batch_y = mnist.train.next_batch(60)
            sess.run(train_step, feed_dict={self.input_layer: batch_x, labels: batch_y, self.is_training: True})
            if i % batches_per_validate_data == 0:
                val_accuracy = sess.run(accuracy, feed_dict={self.input_layer: mnist.validation.images,
                                                             labels: mnist.validation.labels,
                                                             self.is_training: False})
                self.training_accuracies.append(val_accuracy)
        print("{}: The final accuracy on validation data is {}".format(self.name, val_accuracy))

        # 存储模型
        if save_model:
            tf.train.Saver().save(sess, save_model)

    def test(self, sess, test_training_accuracy=False, restore=None):
        # 定义label
        labels = tf.placeholder(tf.float32, [None, 10])

        # 准确率
        correct_prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 是否加载模型
        if restore:
            tf.train.Saver().restore(sess, restore)

        test_accuracy = sess.run(accuracy, feed_dict={self.input_layer: mnist.test.images,
                                                      labels: mnist.test.labels,
                                                      self.is_training: False})

        print("{}: The final accuracy on test data is {}".format(self.name, test_accuracy))


def plot_training_accuracies(*args, batches_per_validate_data):
    """
    绘制模型在训练过程中的准确率曲线

    :param args: 一个或多个NeuralNetWork对象
    :param batches_per_validate_data: 训练多少个batch进行一次数据验证
    """
    fig, ax = plt.subplots()

    for nn in args:
        ax.plot(range(0, len(nn.training_accuracies) * batches_per_validate_data, batches_per_validate_data),
                nn.training_accuracies, label=nn.name)
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy During Training')
    ax.legend(loc=4)
    ax.set_ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.show()


def train_and_test(use_larger_weights, learning_rate, activation_fn, training_batches=50000,
                   batches_per_validate_data=500):
    """
    使用相同的权重初始化生成两个网络对象，其中一个使用BN，另一个不使用BN

    :param use_larger_weights: 是否使用更大的权重
    :param learning_rate: 学习率
    :param activation_fn: 激活函数
    :param training_batches: 训练阶段使用的batch数（默认为50000）
    :param batches_per_validate_data: 训练多少个batch后在validation数据上进行测试
    """
    if use_larger_weights:
        weights = [np.random.normal(size=(784, 128), scale=10.0).astype(np.float32),
                   np.random.normal(size=(128, 128), scale=10.0).astype(np.float32),
                   np.random.normal(size=(128, 128), scale=10.0).astype(np.float32),
                   np.random.normal(size=(128, 10), scale=10.0).astype(np.float32)
                   ]
    else:
        weights = [np.random.normal(size=(784, 128), scale=0.05).astype(np.float32),
                   np.random.normal(size=(128, 128), scale=0.05).astype(np.float32),
                   np.random.normal(size=(128, 128), scale=0.05).astype(np.float32),
                   np.random.normal(size=(128, 10), scale=0.05).astype(np.float32)
                   ]

    tf.reset_default_graph()

    nn = NeuralNetWork(weights, activation_fn, use_batch_norm=False)  # Without BN
    bn = NeuralNetWork(weights, activation_fn, use_batch_norm=True)  # With BN

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        print("【Training Result:】\n")

        nn.train(sess, learning_rate, training_batches, batches_per_validate_data)
        bn.train(sess, learning_rate, training_batches, batches_per_validate_data)

        print("\n【Testing Result:】\n")
        nn.test(sess)
        bn.test(sess)

    plot_training_accuracies(nn, bn, batches_per_validate_data=batches_per_validate_data)


# train_and_test(use_larger_weights=False, learning_rate=0.01, activation_fn=tf.nn.relu)

train_and_test(use_larger_weights=False, learning_rate=0.01, activation_fn=tf.nn.relu, training_batches=3000,
               batches_per_validate_data=50)
