import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='0'
"""
tensorflow 模型保存的几个文件，meta表示保存的是图解钩，包括变量还有集合等。
checkpoint 保存的是checkpoint文件的目录，记录了保存的最新的checkpoint文件以及checkp文件列表
meta表示保存图结构，而ckpt.data变量的值
这里默认保存的是所有的变量
"""
w1 = tf.Variable(tf.constant(2.0, shape=[1]), name='w1')
w2 = tf.Variable(tf.constant(3.0, shape=[1]), name='w2')

a = tf.placeholder('float', name='a')
b = tf.placeholder('float', name='b')
r1 = tf.multiply(w1, a)
r2 = tf.multiply(w2, b)
y = tf.add(r1, r2, name='final_op_add')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print(a)
    print(b)
    print(y)
    print(sess.run(y, feed_dict={a: 10.0, b: 10.0}))
    saver.save(sess, './8_5basic/model.ckpt')