import numpy as np
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
不用tf.gradients 这个函数来做,而是用原本就有的计算梯度的优化算法来做
"""
# x_pure = np.random.randint(-10, 100, 40)
# x_train = x_pure + np.random.randn(40) / 10  # 为x加噪声
# y_train = 3 * x_pure + 2 + np.random.randn(40) / 10  # 为y加噪声  y = 3x + 2
#
# learning_rate = 0.001
# x_input = tf.placeholder(tf.float32, name='x_input')
# y_input = tf.placeholder(tf.float32, name='y_input')
# w = tf.Variable(2.0, name='weight')
# b = tf.Variable(1.0, name='biases')
# y = tf.add(tf.multiply(x_input, w), b)
# loss_op = tf.reduce_sum(tf.pow(y_input - y, 2)) / (2 * 32)
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op)  # 直接使用梯度优化器来计算
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(20):
#         _, loss = sess.run([train_op, loss_op], feed_dict={x_input: x_train[i], y_input: y_train[i]})
#         print("epoch: {} \t loss: {} ".format(i, loss))
#     print("使用梯度下降更新参数之后的系数w 和 b 分别是:", sess.run([w, b]))


"""
使用tf.gradients 这个函数来做,同时更新参数 w 和 b
"""
x_pure = np.random.randint(-10, 100, 40)
x_train = x_pure + np.random.randn(40) / 10  # 为x加噪声
y_train = 3 * x_pure + 2 + np.random.randn(40) / 10  # 为y加噪声  y = 3x + 2

learning_rate = 0.001
x_input = tf.placeholder(tf.float32, name='x_input')
y_input = tf.placeholder(tf.float32, name='y_input')
w = tf.Variable(2.0, name='weight')
b = tf.Variable(1.0, name='biases')
y = tf.add(tf.multiply(x_input, w), b)
loss_op = tf.reduce_sum(tf.pow(y_input - y, 2)) / (2 * 32)

gradients_node = tf.gradients(loss_op, [w, b])  # 使用的是梯度计算公式来计算
new_W = w.assign(w - learning_rate * gradients_node[0])  # 每次更新的值赋值给w,同时再赋值给new_W
new_b = b.assign(b - learning_rate * gradients_node[1])
print('the gradients is: ', gradients_node)
# tf.summary.scalar('norm_grads', gradients_node)
# tf.summary.histogram('norm_grads', gradients_node)
# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter('log')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(40):
        w_, b_, gradients, loss = sess.run([new_W, new_b, gradients_node, loss_op],
                                           feed_dict={x_input: x_train[i], y_input: y_train[i]})
        print("epoch: {} \t loss: {} \t gradients: {}".format(i, loss, gradients))
        print(w_, b_)




