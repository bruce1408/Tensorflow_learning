# coding=utf-8
import tensorflow as tf

import numpy as np
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1)

batch_size = 3
max_time = 7
input_depth = 8
num_units = 20
vocab_size = 10

inputs = tf.placeholder(shape=(batch_size, max_time, input_depth), dtype=tf.float32)
inputdata = tf.transpose(inputs, perm=[1, 0, 2])
sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
inputs_ta = inputs_ta.unstack(inputdata)

cell = tf.contrib.rnn.LSTMCell(num_units)

encoderoutput, encoderfinalstate = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
decoderInput = tf.transpose(encoderoutput, perm=[1, 0, 2])


def loop_fn(time, cell_output, cell_state, loop_state):
    emit_output = cell_output  # == None for time == 0
    if cell_output is None:  # time == 0
        next_cell_state = cell.zero_state(batch_size, tf.float32)
    else:
        next_cell_state = cell_state
    elements_finished = (time >= sequence_length)
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(finished,
                         lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
                         lambda: inputs_ta.read(time))
    next_loop_state = None
    return elements_finished, next_input, next_cell_state, emit_output, next_loop_state


outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
# dynamicoutput, dynamicsttae = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
outputs = outputs_ta.stack()

init = tf.global_variables_initializer()
data = np.arange(168).reshape([3, 7, 8])
seq = np.array([7, 7, 7])
with tf.Session() as sess:
    sess.run(init)
    # rawOutput, rawStates, dyns, dyno = sess.run([outputs, final_state, dynamicsttae, dynamicoutput],
    #                                                   feed_dict={inputs: data, sequence_length: seq})
    rawOutput, rawStates = sess.run([outputs, final_state],
                                                feed_dict={inputs: data, sequence_length: seq})
    print('raw_rnn_outputs:', rawOutput.shape)
    print('raw_rnn_state', rawStates[0].shape)
    # print('dynamic_state', dyns[0].shape)
    # print('dynamic_output', dyno.shape)

# W = tf.Variable(tf.random_uniform([num_units, vocab_size], -1, 1), dtype=tf.float32)  # shape is [40, 10]
# b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)  # shape is [10, ]
#
# def loop_fn_initial():  # 初始状态
#     initial_elements_finished = (0 >= sequence_length)  # all False at the initial step
#     initial_input = tf.zeros([batch_size, num_units], dtype=tf.float32)
#     # initial_input = None
#     initial_cell_state = encoderfinalstate
#     initial_cell_output = None
#     initial_loop_state = None  # we don't need to pass any additional information
#     return initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state
#
#
# def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
#     def get_next_input():
#         output_logits = tf.add(tf.matmul(previous_output, W), b)
#         prediction = tf.argmax(output_logits, axis=1)
#         next_input = tf.random_uniform([batch_size, num_units])
#         return next_input
#
#     elements_finished = (time >= sequence_length)  # this operation produces boolean tensor of [batch_size]
#     # defining if corresponding sequence has ended
#
#     finished = tf.reduce_all(elements_finished)  # -> boolean scalar
#     input = tf.cond(finished, lambda: tf.zeros([batch_size, num_units], dtype=tf.float32), get_next_input)  #
#     state = previous_state
#     output = previous_output
#     loop_state = None
#     return elements_finished, input, state, output, loop_state
#
#
# def loop_fn(time, previous_output, previous_state, previous_loop_state):
#     if previous_state is None:  # time == 0
#         assert previous_output is None and previous_state is None
#         return loop_fn_initial()  # 初始值状态
#     else:
#         return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
#
#
#     # decoder_final_state shape is: [batch x decoder_hidden_size]
#
# decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
#
#
# init = tf.global_variables_initializer()
# data = np.arange(168).reshape([3, 7, 8])
# seq = np.array([7, 7, 7])
# with tf.Session() as sess:
#     sess.run(init)
#     dyns, dyno = sess.run([decoder_outputs_ta, encoderfinalstate],
#                                                 feed_dict={inputs: data, sequence_length: seq})
#
#     print(dyns.shape)
#     print(dyno[0].shape)