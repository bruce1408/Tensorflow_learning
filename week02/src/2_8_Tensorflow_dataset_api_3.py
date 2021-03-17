import os
import numpy as np
import tensorflow as tf

np.random.seed(42)
data = np.random.random([4, 4])
X = tf.placeholder(dtype=tf.float32, shape=[4, 4], name='X')
dataset = tf.data.Dataset.from_tensor_slices(X)
iterator = tf.data.Iterator.from_structure(
    dataset.output_types, dataset.output_shapes)
dataset_next_op = iterator.get_next()

# name the operation
dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')

w = np.random.random([1, 4])
W = tf.Variable(w, name='W', dtype=tf.float32)
output = tf.multiply(W, dataset_next_op, name='output')
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
sess.run(dataset_init_op, feed_dict={X: data})
# print(sess.run(output))
# saver.save(sess, './model3/model3.ckpt')
while True:
    try:
        print(sess.run(output))
    except tf.errors.OutOfRangeError:
        saver.save(sess, './model3/', global_step=1002)
    break
