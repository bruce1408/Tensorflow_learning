#!/usr/bin/env/python
# coding=utf-8
import tensorflow as tf
import random
import numpy as np

random.seed(0)
np.random.seed(0)
# 定义一个未知变量input_ids用于存储索引
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])

# 定义一个已知变量embedding，是一个5*5的对角矩阵
# embedding = tf.Variable(np.identity(5, dtype=np.int32))

# 或者随机一个矩阵
embedding = a = np.random.random([5, 3])

"""
[[0.5488135  0.71518937 0.60276338]
 [0.54488318 0.4236548  0.64589411]
 [0.43758721 0.891773   0.96366276]
 [0.38344152 0.79172504 0.52889492]
 [0.56804456 0.92559664 0.07103606]]
"""

# 根据input_ids中的id，查找embedding中对应的元素
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# print(embedding.eval())

print('\n',sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))

