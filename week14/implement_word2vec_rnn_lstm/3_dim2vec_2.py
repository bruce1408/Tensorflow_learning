# coding=utf-8
import tensorflow as tf
import numpy as np
np.random.seed(1)
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.control_flow_ops import division
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1)

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2
# def batch_input(inputs, max_sequence_length=None):
#     """
#     Args:
#         inputs:
#             list of sentences (integer lists)
#         max_sequence_length:
#             integer specifying how large should `max_time` dimension be.
#             If None, maximum sequence length would be used
#
#     Outputs:
#         inputs_time_major:
#             input sentences transformed into time-major matrix
#             (shape [max_time, batch_size]) padded with 0s
#         sequence_lengths:
#             batch-sized list of integers specifying amount of active
#             time steps in each input sequence
#     """
#
#     sequence_lengths = [len(seq) for seq in inputs]
#     batch_size = len(inputs)
#
#     if max_sequence_length is None:
#         max_sequence_length = max(sequence_lengths)
#
#     inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
#
#     for i, seq in enumerate(inputs):
#         for j, element in enumerate(seq):
#             inputs_batch_major[i, j] = element
#
#     # [batch_size, max_time] -> [max_time, batch_size]
#     # inputs_time_major = inputs_batch_major.swapaxes(0, 1)
#     return inputs_batch_major, sequence_lengths
#     # return inputs_time_major, sequence_lengths
#
#
# def random_sequences(length_from, length_to,
#                      vocab_lower, vocab_upper,
#                      batch_size):
#     """ Generates batches of random integer sequences,
#         sequence length in [length_from, length_to],
#         vocabulary in [vocab_lower, vocab_upper]
#     """
#     if length_from > length_to:
#         raise ValueError('length_from > length_to')
#
#     def random_length():
#         if length_from == length_to:
#             return length_from
#         return np.random.randint(length_from, length_to + 1)
#
#     while True:
#         yield [
#             np.random.randint(low=vocab_lower,
#                               high=vocab_upper,
#                               size=random_length()).tolist()
#             for _ in range(batch_size)
#         ]
#
#
# inputdata, lengseq = batch_input([[2, 7, 7, 6, 5, 9], [2, 3, 4, 5], [4, 5, 6, 8]])
# print(inputdata)
# print(lengseq)
#
#
# encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
# encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
# decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
#
#
# # embeddings modul
# embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)  # [10x20]
# encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
# encoder_cell = LSTMCell(encoder_hidden_units)
#
# ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) =\
#     (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
#                                     cell_bw=encoder_cell,
#                                     inputs=encoder_inputs_embedded,
#                                     sequence_length=encoder_inputs_length,
#                                     dtype=tf.float32, time_major=False)
#     )
#
#
# encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)  # [batch_size x maxstep x 40]
# encoder_outputs = tf.Print(encoder_outputs, [tf.shape(encoder_outputs)], message='encoder_outputs')
# encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
# encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
# encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)  # tuple c & h[batch x 40]
#
# # decoder
# decoder_cell = LSTMCell(decoder_hidden_units)
# batch_size, encoder_max_time = tf.unstack(tf.shape(encoder_inputs))  # encoder_input shape as params
#
#
# decoder_lengths = encoder_inputs_length + 3
# W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)  # shape is [40, 10]
# b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)  # shape is [10, ]
#
#
# assert EOS == 1 and PAD == 0
#
# eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')  # [1 x batch_size]
# pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')  # [1 x batch_size]
#
# eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)  # [batch x 20]
# pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)  # [batch x 20]
#
#
# def loop_fn_initial():  # 初始状态
#     initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
#     initial_input = eos_step_embedded
#     # initial_input = None
#     initial_cell_state = encoder_final_state
#     initial_cell_output = None
#     initial_loop_state = None  # we don't need to pass any additional information
#     return initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state
#
#
# def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
#     def get_next_input():
#         output_logits = tf.add(tf.matmul(previous_output, W), b)
#         prediction = tf.argmax(output_logits, axis=1)
#         next_input = tf.nn.embedding_lookup(embeddings, prediction)
#         print('done')
#         return next_input
#
#     elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
#     # defining if corresponding sequence has ended
#
#     finished = tf.reduce_all(elements_finished)  # -> boolean scalar
#     input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)  #
#     state = previous_state
#     output = previous_output
#     loop_state = None
#     return elements_finished, input, state, output, loop_state
#
#
# def loop_fn(time, previous_output, previous_state, previous_loop_state):
#     if previous_state is None:    # time == 0
#         assert previous_output is None and previous_state is None
#         return loop_fn_initial()  # 初始值状态
#     else:
#         return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
#
#
# # decoder_final_state shape is: [batch x decoder_hidden_size]
# decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
# decoder_outputs = decoder_outputs_ta.stack()
#
# # 这里才开始得到 decoder outputs
# # decoder_outputs shape [batch, None, decoder_hidden_size]
# decoder_outputs = tf.transpose(decoder_outputs, perm=[1, 0, 2])
# decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
# decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
# decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
# # decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
# decoder_logits = tf.reshape(decoder_logits_flat, (decoder_batch_size, decoder_max_steps, vocab_size))
#
# # 定义predicttion
# decoder_prediction = tf.argmax(decoder_logits, 2)
# # 定义损失函数
# stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#     labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
#     logits=decoder_logits,
# )
# loss = tf.reduce_mean(stepwise_cross_entropy)
# train_op = tf.train.AdamOptimizer().minimize(loss)
# # session 运行
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     batch_size = 100
#
#     batches = random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=batch_size)
#
#     print('head of the batch:')
#     for seq in next(batches)[:10]:
#         print(seq)
#
#
#     def next_feed():
#         batch = next(batches)
#         # print(batch)
#         encoder_inputs_, encoder_input_lengths_ = batch_input(batch)
#         decoder_targets_, _ = batch_input(
#             [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
#         )
#         return {
#             encoder_inputs: encoder_inputs_,
#             encoder_inputs_length: encoder_input_lengths_,
#             decoder_targets: decoder_targets_,
#         }
#
#
#     loss_track = []
#     max_batches = 3001
#     batches_in_epoch = 100
#
#     try:
#         for batch in range(max_batches):
#             fd = next_feed()
#             _, l = sess.run([train_op, loss], fd)
#             loss_track.append(l)
#
#             if batch == 0 or batch % batches_in_epoch == 0:
#                 print('batch {}'.format(batch))
#                 print('  minibatch loss: {}'.format(sess.run(loss, fd)))
#                 predict_ = sess.run(decoder_prediction, fd)
#                 for i, (inp, pred) in enumerate(zip(fd[encoder_inputs], predict_)):
#                     print('  sample {}:'.format(i + 1))
#                     print('    input     > {}'.format(inp))
#                     print('    predicted > {}'.format(pred))
#                     if i >= 2:
#                         break
#                 print()
#
#     except KeyboardInterrupt:
#         print('training interrupted')
#     import matplotlib.pyplot as plt
#
#     plt.plot(loss_track)
#     print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track) * batch_size,
#                                                                  batch_size))







feature_size = 9

def get_data(batch_size):
    train_data = np.random.normal(-1, 1, size=[batch_size, 7, feature_size])
    target_data = np.random.normal(-0.1, 0.1, size=[batch_size, 7, feature_size])
    return {encoder_inputs: train_data,
            decoder_targets: target_data,
            decoder_inputs: target_data}


# encoder_decoder_input
encoder_inputs = tf.placeholder(tf.float32, [None, None, feature_size], name='encoder_inputs')
decoder_targets = tf.placeholder(tf.float32, [None, None, feature_size], name='decoder_targets')
decoder_inputs = tf.placeholder(tf.float32, [None, None, feature_size], name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

# encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
# encoder_inputs_embedded = tf.Print(encoder_inputs_embedded, [tf.shape(encoder_inputs_embedded)],
# message='encoder_inputs_embed')
# decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
# decoder_inputs_embedded = tf.Print(decoder_inputs_embedded, [tf.shape(decoder_inputs_embedded)],
# message='decoder_inputs_embed')
# define encoder layer
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, dtype=tf.float32,
                                                         time_major=False)
del encoder_outputs

# define decoder layer
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell,
    decoder_inputs,
    initial_state=encoder_final_state,
    dtype=tf.float32,
    time_major=False,
    scope='plain_decoder',)
# decoder_outputs shape = [None, None, 20]

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)  # shape [batch, max_step, 10]
# decoder_logits = tf.Print(decoder_logits, [tf.shape(decoder_logits)], message='decoder_logits')
# decoder_prediction = tf.argmax(decoder_logits, 2)  # shape[None, None] = [batch, max_step]
decoder_prediction = tf.layers.dense(decoder_logits, 8)
# optimizer
loss = tf.reduce_mean(tf.square(decoder_prediction - encoder_inputs))
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()
# stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#     labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32), logits=decoder_logits)
# loss = tf.reduce_mean(stepwise_cross_entropy)
# train_op = tf.train.AdamOptimizer().minimize(loss)
# init = tf.global_variables_initializer()

# toy data and train the modle
# batch = [[6], [3, 4], [9, 8, 7]]
# batch, batch_length = helpers.batch(batch)
# print('batch_encoder:\n'+str(batch))
#
# din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32))
# print('decoder inputs:\n'+str(din_))
#
# with tf.Session() as sess:
#     sess.run(init)
#     pred = sess.run(decoder_prediction, feed_dict={encoder_inputs: batch, decoder_inputs: din_})
#     print('decoder_pred:\n'+str(pred))


# normal training the data
loss_track = []
epochs = 30001
batches_in_epoch = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    try:
        fd = get_data(100)
        for batch in range(epochs):

            _, l, enfs = sess.run([train_op, loss, encoder_final_state], fd)
            loss_track.append(l)
            if batch == 0 or batch % batches_in_epoch == 0:
                print('epoch {}'.format(batch))
                print('  minibatch loss:\n {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs], predict_)):
                    print('\033[1;32m  sample {}:\033[0m'.format(i + 1))
                    print('\033[1;36m    input     =================>\033[0m\n{}'.format(inp))
                    print('\033[1;36m    predicted =================>\n{}'.format(pred))
                    if i >= 2:
                        break
                # print()
    except KeyboardInterrupt:
        print('training interrupted')
    print(sess.run(embeddings))