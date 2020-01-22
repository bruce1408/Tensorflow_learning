#!/usr/bin/python
# coding = utf-8
import tensorflow as tf
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter1 = tf.Variable(tf.random_normal([1, 1, 5, 1]))
op1 = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='SAME')

input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([2, 2, 5, 1]))
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 3
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([1, 1, 5, 1]))
op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# case 4
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# case 5

input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 6

input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 7

input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

# case 8

input = tf.Variable(tf.random_normal([10, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print('*' * 20 + ' op1 ' + '*' * 20)
    print(sess.run(input))
    print(sess.run(filter1))
    print(sess.run(op1))
    print('*' * 20 + ' op2 ' + '*' * 20)
    print(sess.run(op2))
    print('*' * 20 + ' op3 ' + '*' * 20)
    print(sess.run(op3))
    print('*' * 20 + ' op4 ' + '*' * 20)
    print(sess.run(op4))
    print('*' * 20 + ' op5 ' + '*' * 20)
    print(sess.run(op5))
    print('*' * 20 + ' op6 ' + '*' * 20)
    print(sess.run(op6))
    print('*' * 20 + ' op7 ' + '*' * 20)
    print(sess.run(op7))
    print('*' * 20 + ' op8 ' + '*' * 20)
    print(sess.run(op8))
