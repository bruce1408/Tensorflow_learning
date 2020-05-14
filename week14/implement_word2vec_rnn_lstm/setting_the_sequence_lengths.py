# # coding=utf-8
# import tensorflow as tf
# import numpy as np
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
# # tf.logging.set_verbosity(tf.logging.INFO)
# # tf.set_random_seed(1)
#
# n_steps = 2
# n_inputs = 3
# n_neurons = 5
#
# # batch_max = tf.placeholder(tf.float32, [None])
# X = tf.placeholder(tf.float32, [None, None, n_inputs])
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#
# seq_length = tf.placeholder(tf.int32, [None])
# outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
#
#
# init = tf.global_variables_initializer()
#
#
# X_batch = np.array([
#         # step 0     step 1
#         [[0, 1, 2], [9, 8, 7]],
#         [[3, 4, 5], [0, 0, 0]],
#         [[6, 7, 8], [6, 5, 4]],
#         [[9, 0, 1], [3, 2, 1]]
#     ])
#
# Y_batch = np.array(
#     [
#         [[3, 4, 6], [1, 3, 6], [3, 4, 7]],
#         [[4, 0, 1], [0, 3, 6], [2, 9, 1]],
#         [[6, 4, 6], [1, 3, 6], [3, 4, 9]],
#         [[3, 4, 7], [3, 0, 6], [2, 2, 1]]
#     ])
#
# import pdb;pdb.set_trace()
# def get_batch(sources, batch_size):
#
#     for batch_i in range(0, X_batch.shape[0]//batch_size):
#         start_i = batch_i * batch_size
#         sources_batch = sources[start_i:start_i + batch_size]
#         # print(sources_batch)
#         # 补全序列
#         # pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
#         # pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
#
#         # 记录每条记录的长度
#
#         source_lengths = []
#         for source in sources_batch:
#             source_lengths.append(source.shape[0])
#
#         yield sources_batch, source_lengths
#
#
# # for bat_i, (data, lenth) in enumerate(get_batch(Y_batch, 4)):
#     # print(data)
#     # print(lenth)
#
#
# # seq_length_batch = np.array([2, 1, 2, 2])
# #
# with tf.Session() as sess:
#     init.run()
#     for index, i in enumerate([X_batch, Y_batch]):
#
#         for batch_i, (X_batch, seq_length_batch) in enumerate(get_batch(i, 4)):
#             seq_length_batch = np.array(seq_length_batch)
#             outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch,
#                                                             seq_length: seq_length_batch})
#             print(index)
#             print(outputs_val)





# coding=utf-8
# import tensorflow as tf
# import numpy as np
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
# # tf.logging.set_verbosity(tf.logging.INFO)
# # tf.set_random_seed(1)
#
# n_steps = 2
# n_inputs = 3
# n_neurons = 5
#
# # batch_max = tf.placeholder(tf.float32, [None])
# X = tf.placeholder(tf.float32, [None, None, n_inputs])
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#
# seq_length = tf.placeholder(tf.int32, [None])
# outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
#
# # x = tf.contrib.layers.embed_sequence(X, 30, 15)
# init = tf.global_variables_initializer()
#
#
# X_batch = np.array([
#         # step 0     step 1
#         [[0, 1, 2], [9, 8, 7]],
#         [[3, 4, 5], [0, 0, 0]],
#         [[6, 7, 8], [6, 5, 4]],
#         [[9, 0, 1], [3, 2, 1]]
#     ])
#
# Y_batch = np.array(
#     [
#         [[3, 4, 6], [1, 3, 6], [3, 4, 7]],
#         [[4, 0, 1], [0, 3, 6], [2, 9, 1]],
#         [[6, 4, 6], [1, 3, 6], [3, 4, 9]],
#         [[3, 4, 7], [3, 0, 6], [2, 2, 1]]
#     ])
#
# # import pdb;pdb.set_trace()
# def get_batch(sources, batch_size):
#
#     for batch_i in range(0, X_batch.shape[0]//batch_size):
#         start_i = batch_i * batch_size
#         sources_batch = sources[start_i:start_i + batch_size]
#         # print(sources_batch)
#         # 补全序列
#         # pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
#         # pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
#
#         # 记录每条记录的长度
#
#         source_lengths = []
#         for source in sources_batch:
#             source_lengths.append(source.shape[0])
#
#         yield sources_batch, source_lengths
#
#
# # for bat_i, (data, lenth) in enumerate(get_batch(Y_batch, 4)):
#     # print(data)
#     # print(lenth)
# # seq_length_batch = np.array([2, 1, 2, 2])
#
#
#
# with tf.Session() as sess:
#     init.run()
#     for index, i in enumerate([X_batch, Y_batch]):
#
#         for batch_i, (X_batch, seq_length_batch) in enumerate(get_batch(i, 4)):
#             seq_length_batch = np.array(seq_length_batch)
#             outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch,
#                                                             seq_length: seq_length_batch})
#             print(index)
#             print(outputs_val)
#             # print(sess.run(x))


import tensorflow as tf
import numpy as np
from collections import defaultdict
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


features = [[1, 2, 3], [4, 5, 6]]
input_data = tf.placeholder(tf.int32, [None, 3])

# vocab_size 一定要大于这个id里面的最大的数
outputs = tf.contrib.layers.embed_sequence(input_data, vocab_size=10, embed_dim=4)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(sess.run(outputs, feed_dict={input_data: features}))

