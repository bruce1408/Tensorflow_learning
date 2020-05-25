import os
import numpy as np
import tensorflow as tf

np.random.seed(42)
data = np.random.random([4, 4])
X = tf.placeholder(dtype=tf.float32, shape=[4, 4], name='X')
dataset = tf.data.Dataset.from_tensor_slices(X)
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
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

# np.random.seed(42)
# data = np.random.random([4, 4])
# tf.reset_default_graph()
# sess = tf.Session()
# saver = tf.train.import_meta_graph('model3/model3.ckpt.meta')
# ckpt = tf.train.get_checkpoint_state(os.path.dirname('model3/checkpoint'))
# saver.restore(sess, ckpt.model_checkpoint_path)
# graph = tf.get_default_graph()
#
# # Restore the init operation
# dataset_init_op = graph.get_operation_by_name('dataset_init')
#
# X = graph.get_tensor_by_name('X:0')
# output = graph.get_tensor_by_name('output:0')
# sess.run(dataset_init_op, feed_dict={X: data})
# while True:
#     try:
#         print(sess.run(output))
#     except tf.errors.OutOfRangeError:
#         break
