'''
Author: your name
Date: 2021-03-17 23:40:34
LastEditTime: 2021-03-17 23:40:35
LastEditors: your name
Description: In User Settings Edit
FilePath: /Tensorflow_learning/week03/src/3_0_basic_linerRegress.py
'''
# __author__ = 'Bruce Cui'
"""
线性回归代码, 展示回归的曲线
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 Error
np.random.seed(1)  # 随机采样一个随机值
# create data
x_data = np.random.rand(100).astype(np.float32)  # 0~1之间的随机的100个数
y_data = x_data * 0.1 + 0.3 + np.random.normal(0.0, 0.03)

# create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # -1 到 1 之间的均匀分布
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)  # 优化目标是损失函数loss

init = tf.initialize_all_variables()  # 初始化所有的变量

# create tensorflow structure end ###
sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for step in range(201):
    sess.run(train)

    plt.cla()  # 这个函数和plt.pause()配对使用。
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(Weights) * x_data + sess.run(biases))
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.draw()
    plt.pause(0.01)
    if step == 200:
        plt.ioff()
        plt.show()

        # if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
        print('y = %f*x+%f' % (sess.run(Weights), sess.run(biases)))


# example 2

numPuntos = 1000
conJunto = []
# 这个程序简短是生成 n 个随机数的程序片段
for i in range(numPuntos):
    x1 = np.random.normal(0.0, 0.55)
    # y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    y1 = x1 * 0.1 + 0.3
    conJunto.append([x1, y1])
x_data = [v[0] for v in conJunto]
y_data = [v[1] for v in conJunto]
# 这个程序简短是生成 n 个随机数的程序片段
#
# plt.plot(x_data, y_data, 'ro', label='Original data')
# plt.legend()
# plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
# -画图的部分-
for j in range(40):
    sess.run(train)
    plt.cla()  # 这个函数和plt.pause()配对使用。
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.draw()
    plt.pause(1.5)
    if j == 39:
        plt.ioff()
        plt.show()
    print('经过第%d次训练，预测的方程的结果是： y = %f * X + %f'%(j,sess.run(W),sess.run(b)))
