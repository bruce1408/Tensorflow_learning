# # coding=utf-8
# import tensorflow as tf
# import numpy as np
#
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.set_random_seed(1)
#
# # param
# hidden_dim = 100
# attention_hidden_dim = 100
# num_steps = 7
# attention_W = tf.Variable(tf.random_uniform([hidden_dim * 4, attention_hidden_dim], 0.0, 1.0), name="attention_W")
# attention_U = tf.Variable(tf.random_uniform([hidden_dim * 2, attention_hidden_dim], 0.0, 1.0), name="attention_U")
# attention_V = tf.Variable(tf.random_uniform([attention_hidden_dim, 1], 0.0, 1.0), name="attention_V")
#
#
# def attention(prev_state, enc_outputs):
#     """
#     Attention model for Neural Machine Translation
#     :param prev_state: the decoder hidden state at time i-1
#     :param enc_outputs: the encoder outputs, a length 'T' list.
#     """
#     e_i = []
#     c_i = []
#     x_unpacked = tf.unstack(enc_outputs)
#     for outputs in x_unpacked:
#         atten_hidden = tf.tanh(tf.add(tf.matmul(prev_state, attention_W), tf.matmul(outputs, attention_U)))
#         e_i_j = tf.matmul(atten_hidden, attention_V)
#         e_i.append(e_i_j)
#     e_i = tf.concat(e_i, axis=1)
#     # e_i = tf.exp(e_i)
#     alpha_i = tf.nn.softmax(e_i)
#     alpha_i = tf.split(alpha_i, num_steps, 1)
#     for alpha_i_j, output in zip(alpha_i, enc_outputs):
#         c_i_j = tf.multiply(alpha_i_j, output)
#         c_i.append(c_i_j)
#     c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, num_steps, hidden_dim * 2])
#     c_i = tf.reduce_sum(c_i, 1)
#     return c_i
#
#
# a = np.arange(28000).reshape(20, 7, 200)
# encoder_input = tf.convert_to_tensor(a, dtype=tf.float32)
# prev_state = tf.convert_to_tensor(np.arange(8000).reshape(20, 400), dtype=tf.float32)
# encoder_input = tf.transpose(encoder_input, perm=[1, 0, 2])
# init = tf.global_variables_initializer()
# result = attention(prev_state=prev_state, enc_outputs=encoder_input)
# with tf.Session() as sess:
#     sess.run(init)
#     x = sess.run(result)
#     print(x.shape)


# coding=utf-8

# import tensorflow as tf
# from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
# import numpy as np
# import attention
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.set_random_seed(1)
#
# PAD = 0
# EOS = 1
#
# vocab_size = 10
# input_embedding_size = 20
# encoder_hidden_units = 20
# decoder_hidden_units = encoder_hidden_units * 2
#
#
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
# encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
# # encoder_inputs = tf.Print(encoder_inputs, [tf.shape(encoder_inputs)])
# encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
# decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
#
#
# # embeddings modul
# embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
# encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
# encoder_cell = LSTMCell(encoder_hidden_units)
#
# ((encoder_fw_outputs, encoder_bw_outputs),
#  (encoder_fw_final_state, encoder_bw_final_state)) =\
#     (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
#                                     cell_bw=encoder_cell,
#                                     inputs=encoder_inputs_embedded,
#                                     sequence_length=encoder_inputs_length,
#                                     dtype=tf.float32, time_major=False)
#     )
#
# attention_size = 50
# encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
#
# hidden_size = encoder_outputs.shape[2].value  # D value - hidden size of the RNN layer
#
# # Trainable parameters
# w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
# b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
# u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#
# with tf.name_scope('v'):
#     v = tf.tanh(tf.tensordot(encoder_outputs, w_omega, axes=1) + b_omega)
#
# # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
# vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
# alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
#
# # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
# output = tf.reduce_sum(encoder_outputs * tf.expand_dims(alphas, -1), 1)
#
#
#
# # sequence_length = encoder_outputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
# # hidden_size = encoder_outputs.get_shape()[2].value  # hidden size of the RNN layer
# #
# # # Attention mechanism
# # W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
# # b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
# # u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
# # v = tf.tanh(tf.matmul(tf.reshape(encoder_outputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
# # vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
# # exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
# # alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
#
# # Output of Bi-RNN is reduced with attention vector
# # output = tf.reduce_sum(encoder_outputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
#
# init = tf.global_variables_initializer
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     batch_size = 100
#     batches = random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=batch_size)
#     # print('head of the batch:')
#     # for seq in next(batches)[:10]:
#     #     print(seq)
#
#
#     def next_feed():
#         batch = next(batches)
#         # print(batch)
#         encoder_inputs_, encoder_input_lengths_ = batch_input(batch)
#         decoder_targets_, _ = batch_input([(sequence) + [EOS] + [PAD] * 2 for sequence in batch])
#         return {
#             encoder_inputs: encoder_inputs_,
#             encoder_inputs_length: encoder_input_lengths_,
#             decoder_targets: decoder_targets_,
#         }
#
#
#     loss_track = []
#     max_batches = 2
#     # batches_in_epoch = 100
#
#     try:
#         for math in range(max_batches):
#
#             fd = next_feed()
#             # ec = sess.run([encoder_outputs], fd)
#             atout, enoutputs = sess.run([output, encoder_outputs], fd)
#             print(atout.shape)
#             print(enoutputs.shape)
#
#     except KeyboardInterrupt:
#         print('training interrupted')

# ============================== example ====================
# import tensorflow as tf
# import numpy as np
# np.random.seed(1)
# from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.set_random_seed(1)
#
# PAD = 0
# EOS = 1
#
# vocab_size = 10
# input_embedding_size = 20
# encoder_hidden_units = 20
# decoder_hidden_units = encoder_hidden_units * 2
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
# # encoder_outputs = tf.Print(encoder_outputs, [tf.shape(encoder_outputs)], message='encoder_outputs')
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
#
#
# init = tf.global_variables_initializer
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     batch_size = 100
#     batches = random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=batch_size)
#     # print('head of the batch:')
#     # for seq in next(batches)[:10]:
#     #     print(seq)
#
#
#     def next_feed():
#         batch = next(batches)
#         # print(batch)
#         encoder_inputs_, encoder_input_lengths_ = batch_input(batch)
#         decoder_targets_, _ = batch_input([(sequence) + [EOS] + [PAD] * 2 for sequence in batch])
#         return {
#             encoder_inputs: encoder_inputs_,
#             encoder_inputs_length: encoder_input_lengths_,
#             decoder_targets: decoder_targets_,
#         }
#
#
#     loss_track = []
#     max_batches = 1
#     # batches_in_epoch = 100
#
#     try:
#         for math in range(max_batches):
#
#             fd = next_feed()
#             # ec = sess.run([encoder_outputs], fd)
#             atout, enoutputs = sess.run([decoder_outputs, encoder_outputs], fd)
#             print('encoder outputs', enoutputs.shape)
#             print('attention outputs', atout.shape)
#
#
#     except KeyboardInterrupt:
#         print('training interrupted')

# ============================ example ==================


# ========= new code
# import tensorflow as tf
#
# import numpy as np
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.set_random_seed(1)
#
# batch_size = 3
# max_time = 7
# input_depth = 8
# encoder_hidden_units = decoder_hidden_units = num_units = 20
# vocab_size = 10
# attention_size = 15
# inputs = tf.placeholder(shape=(batch_size, max_time, input_depth), dtype=tf.float32)
# decoder_input = tf.placeholder(shape=(batch_size, max_time, input_depth), dtype=tf.float32)
# # inputdata = tf.transpose(inputs, perm=[1, 0, 2])
# sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
# # inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
# # inputs_ta = inputs_ta.unstack(inputdata)
# decoder_lengths = sequence_length
# eos_step_embedded = tf.random_uniform([batch_size, num_units])  # 终止就是[batch x 20]
# pad_step_embedded = tf.random_normal([batch_size, num_units], stddev=0.01, dtype=tf.float32)
# cell = tf.contrib.rnn.LSTMCell(num_units)
#
# encoderoutput, encoderfinalstate = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
# decoderInput = tf.transpose(encoderoutput, perm=[1, 0, 2])  # 输出结果转置
# # shape = [20, 10]
# W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
# # shape = [10,]
# b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
#
#
# # attention layer
# def loop_fn_initial():  # 初始状态
#     initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
#     initial_input = eos_step_embedded
#     # initial_input = None
#     initial_cell_state = encoderfinalstate  # [batch, 20]
#     initial_cell_output = None
#     initial_loop_state = None  # we don't need to pass any additional information
#     return initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state
#
# # ========================== 每个时间步的循环，非常重要 ===========================
# def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
#     # def get_next_input():
#     #     output_logits = tf.add(tf.matmul(previous_output, W), b) # [batch, 10]
#     #     prediction = tf.argmax(output_logits, axis=1)  # [batch, ]
#     #     next_input = tf.nn.embedding_lookup(embeddings, prediction) # [batch, 20]
#     #     return next_input
#
#     def get_next_input_data():
#         # sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
#         # hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer
#
#         # Attention mechanism
#         sequence_length = encoderoutput.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
#         hidden_size = encoderoutput.get_shape()[2].value  # hidden size of the RNN layer
#
#         # Attention mechanism
#         e_i = list()
#         W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
#         W_prestate = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
#         # b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#         u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#         inputs_data = tf.transpose(encoderoutput, perm=[1, 0, 2])
#         x_unpacked = tf.unstack(inputs_data)
#         for outputs in x_unpacked:
#             atten_hidden = tf.tanh(tf.add(tf.matmul(previous_output, W_prestate), tf.matmul(outputs, W_omega)))
#             e_i_j = tf.matmul(atten_hidden, tf.reshape(u_omega, [-1, 1]))
#             e_i.append(e_i_j)
#         e_i = tf.concat(e_i, axis=1)
#
#         exps = tf.reshape(tf.exp(e_i), [-1, sequence_length])
#         print('done')
#         alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
#
#         # Output of Bi-RNN is reduced with attention vector
#         output = tf.reduce_sum(encoderoutput * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
#         return output
#
#
#     elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
#     # defining if corresponding sequence has ended
#
#     finished = tf.reduce_all(elements_finished)  # -> boolean scalar
#     input = tf.cond(finished, lambda: pad_step_embedded, get_next_input_data)  #
#     state = previous_state
#     output = previous_output
#     loop_state = None
#     return elements_finished, input, state, output, loop_state
# # =============================================================================
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
# decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
# decoder_output = decoder_outputs_ta.stack()
#
#
# init = tf.global_variables_initializer()
# encoder_data = np.arange(168).reshape([3, 7, 8])
# decoder_target = (np.arange(168)+10).reshape([3, 7, 8])
# seq = np.array([7, 7, 7])
# with tf.Session() as sess:
#     sess.run(init)
#     # rawOutput, rawStates, dyns, dyno = sess.run([outputs, final_state, dynamicsttae, dynamicoutput],
#     #                                                   feed_dict={inputs: data, sequence_length: seq})
#     encoderstate = sess.run(encoderfinalstate, feed_dict={inputs: encoder_data,
#                                                                  decoder_input: decoder_target,
#                                                                  sequence_length: seq})
#     print('encoder outputs', encoderstate[0].shape)
# print('raw_rnn_outputs:', rawOutput.shape)
# print('raw_rnn_state', rawStates[0].shape)
# print('dynamic_state', dyns[0].shape)
# print('dynamic_output', dyno.shape)
# ===== new code ================
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


# coding=utf-8
import tensorflow as tf
import numpy as np

np.random.seed(1)
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1)

PAD = 0
EOS = 1

vocab_size = 10
attention_size = 50
batch_size = 100
maxstep = 7
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


# inputdata, lengseq = batch_input([[2, 7, 7, 6, 5, 9], [2, 3, 4, 5], [4, 5, 6, 8]])
# print(inputdata)
# print(lengseq)

generate_data_encoder = np.arange(14000).reshape([100, 7, 20])
generate_data_decoder = np.random.normal(size=[100, 7, 40])
encoder_length = np.ones(100) * 7

encoder_inputs = tf.placeholder(shape=(batch_size, maxstep, 20), dtype=tf.float32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(batch_size, maxstep, 40), dtype=tf.float32, name='decoder_targets')

# embeddings modul
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)  # [10x20]
# encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = \
    (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                     cell_bw=encoder_cell,
                                     inputs=encoder_inputs,
                                     sequence_length=encoder_inputs_length,
                                     dtype=tf.float32, time_major=False)
     )

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)  # [batch_size x maxstep x 40]
encoder_outputs = tf.Print(encoder_outputs, [tf.shape(encoder_outputs)], message='encoder_outputs')
encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)  # tuple c & h[batch x 40]

# decoder
decoder_cell = LSTMCell(decoder_hidden_units)
batch_size, encoder_max_time, _ = tf.unstack(tf.shape(encoder_inputs))  # encoder_input shape as params

decoder_lengths = encoder_inputs_length
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)  # shape is [40, 10]
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)  # shape is [10, ]

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')  # [1 x batch_size]
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')  # [1 x batch_size]

eos_step_embedded = tf.random_normal([batch_size, 20], dtype=tf.float32)  # [batch x 20]
pad_step_embedded = tf.random_normal([batch_size, 20], dtype=tf.float32)  # [batch x 20]
W_omega = tf.Variable(tf.random_uniform([decoder_hidden_units, attention_size]), dtype=tf.float32)
W_prestate = tf.Variable(tf.random_normal([decoder_hidden_units, attention_size], stddev=0.1), dtype=tf.float32)
u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), dtype=tf.float32)
result = tf.Variable(tf.random_uniform([40, 20]), dtype=tf.float32)
inputs_data = tf.transpose(encoder_outputs, perm=[1, 0, 2])
# inputs_ta = tf.TensorArray(dtype=tf.float32, size=maxstep, dynamic_size=True, infer_shape=False)
# inputs_ta = inputs_ta.unstack(inputs_data)
x_unpacked = tf.unstack(inputs_data)


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

# code by my function

def loop_fn_initial():  # 初始状态
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state  # [batch, 40]
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state


# ========================== 每个时间步的循环，非常重要 ===========================
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        sequence_length = encoder_outputs.get_shape()[
            1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = encoder_outputs.get_shape()[2].value  # hidden size of the RNN layer

        # Attention mechanism
        e_i = list()

        inputs_data = tf.transpose(encoder_outputs, perm=[1, 0, 2])
        aixs_0 = inputs_data.get_shape()[0].value
        x_unpacked = tf.unstack(inputs_data, num=aixs_0)
        for outputs in x_unpacked:
            atten_hidden = tf.tanh(tf.add(tf.matmul(previous_output, W_prestate), tf.matmul(outputs, W_omega)))
            e_i_j = tf.matmul(atten_hidden, tf.reshape(u_omega, [-1, 1]))
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)

        exps = tf.reshape(tf.exp(e_i), [-1, sequence_length])
        print('done')
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # Output of Bi-RNN is reduced with attention vector
        next_input = tf.reduce_sum(encoder_outputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        next_input_ = tf.matmul(next_input, result)
        return next_input_

    elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    input = tf.cond(finished, get_next_input, get_next_input)
    # input = tf.while_loop(finished, lambda : pad_step_embedded, get_next_input)
    # input = tf.cond(finished, get_pa_input, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None
    return elements_finished, input, state, output, loop_state


# =============================================================================


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:  # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()  # 初始值状态
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()
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
# session 运行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 100


    def next_feed():
        return {
            encoder_inputs: generate_data_encoder,
            encoder_inputs_length: encoder_length,
            decoder_targets: generate_data_decoder,
        }


    max_batches = 1
    batches_in_epoch = 100

    try:
        for batch in range(max_batches):
            fd = next_feed()
            decoderstate, decoderOutput = sess.run([decoder_final_state, decoder_outputs], fd)
            print('decoder output shape', decoderOutput.shape)

    except KeyboardInterrupt:
        print('training interrupted')
