#  coding: utf-8
import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
os.environ['CUDA_VISIBLE_DEVICES']='0'
# 载入数据集
mnist = input_data.read_data_sets("/raid/bruce/MNIST_data", one_hot=True)

"""
模型加载方法 1，使用saver.restore(sess, MODELNAME.ckpt)重新构建网络结构来加载模型
模型加载方法 2，使用加载网络图的模式，先加载网络图结构，然后开始加载网络结构op名称。
"""
# # 每个批次100张照片
# batch_size = 100
# # 计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
# # 定义两个placeholder
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
#
# # 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(x, W)+b)
#
# # 二次代价函数
# #  loss = tf.reduce_mean(tf.square(y-prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# # 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# # 结果存放在一个布尔型列表中
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# # 求准确率
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
"""
方法 1，重写网络图结构，然后开始加载网络模型，使用restore方法。
"""
#     sess.run(init)
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
#     saver.restore(sess, 'net/my_net.ckpt')
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

"""
方法 2，使用加载网络图结构，然后开始做inference
关键语句：
saver = tf.train.import_meta_graph(./modelpath/MODELNAME.meta) 不再是tf.train.Saver()
saver.restore()
graph = tf.get_default_graph()
a = graph.get_tensor_by_name("input_img:0")
"""
print(mnist.test.images.shape)
x = np.random.random((32, 784))
y = np.random.randint(10, size=(32, 10))
saver = tf.train.import_meta_graph('./net/myModel_10001.meta')
path = ""
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./net')
    if ckpt is None:
        print('Model not found, please train your model first')
    else:
        path = ckpt.model_checkpoint_path
        print(path)
    saver.restore(sess, path)
    graph = tf.get_default_graph()
    a = graph.get_tensor_by_name("input_img:0")
    b = graph.get_tensor_by_name("input_label:0")
    feed_dict = {a: x, b: y}
    # for x, y in enumerate(range(9)):
    #     feed_dict['a'] = x
    #     feed_dict['b'] = y
    add_op = graph.get_tensor_by_name("prediction:0")
    result = tf.argmax(add_op, 1)
    _, result_ = sess.run([add_op, result], feed_dict=feed_dict)
    print(result_)


