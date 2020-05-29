# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops.tensor_array_ops import TensorArray
import numpy as np
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
np.random.seed(1)
tf.set_random_seed(1)
tf.reset_default_graph()

PAD = 0
EOS = 1
vocab_size = 10
batch_size = 10
feature_size = 8
num_steps = 7
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

def get_data(batch_size):
    train_data = np.random.normal(-1, 1, size=[batch_size, 7, feature_size])
    target_data = np.random.normal(-0.1, 0.1, size=[batch_size, 7, feature_size])
    return {encoder_inputs: train_data, decoder_targets: target_data,   decoder_inputs: target_data}


# encoder_decoder_input
encoder_inputs = tf.placeholder(tf.float32, [None, None, feature_size], name='encoder_inputs')
decoder_targets = tf.placeholder(tf.float32, [None, None, feature_size], name='decoder_targets')
decoder_inputs = tf.placeholder(tf.float32, [None, None, feature_size], name='decoder_inputs')

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, dtype=tf.float32)


cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

# del encoder_outputs
# def attention(encoder_outputs, prev_state):
#
#     attention_size = 50
#     hidden_size = encoder_outputs.shape[2].value  # D value - hidden size of the RNN layer
#
#     # Trainable parameters
#     w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
#     b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#     u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#
#     with tf.name_scope('v'):
#         v = tf.tanh(tf.tensordot(encoder_outputs, w_omega, axes=1) + b_omega)
#
#     # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
#     vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
#     alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
#
#     # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
#     output = tf.reduce_sum(encoder_outputs * tf.expand_dims(alphas, -1), 1)
#     return output

def attention(prev_state, enc_outputs):
    """
    Attention model for Neural Machine Translation
    :param prev_state: the decoder hidden state at time i-1
    :param enc_outputs: the encoder outputs, a length 'T' list.
    """

    e_i = []
    c_i = []
    attention_hidden_dim = 50
    attention_W = tf.Variable(tf.random_uniform([encoder_hidden_units, attention_hidden_dim], 0.0, 1.0), name="attention_W")
    attention_U = tf.Variable(tf.random_uniform([encoder_hidden_units, attention_hidden_dim], 0.0, 1.0), name="attention_U")
    attention_V = tf.Variable(tf.random_uniform([attention_hidden_dim, 1], 0.0, 1.0), name="attention_V")

    for output in enc_outputs:
        atten_hidden = tf.tanh(tf.add(tf.matmul(prev_state, attention_W), tf.matmul(output, attention_U)))
        e_i_j = tf.matmul(atten_hidden, attention_V)
        e_i.append(e_i_j)
    e_i = tf.concat(e_i, axis=1)
    alpha_i = tf.nn.softmax(e_i)
    alpha_i = tf.split(alpha_i, num_steps, 1)
    for alpha_i_j, output in zip(alpha_i, enc_outputs):
        c_i_j = tf.multiply(alpha_i_j, output)
        c_i.append(c_i_j)
    c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, num_steps, decoder_hidden_units * 2])
    c_i = tf.reduce_sum(c_i, 1)
    return c_i


def decode(cell, init_state, enc_outputs, loop_function=None):
    outputs = []
    prev = None
    state = init_state
    for i, inp in enumerate(tf.map_fn(lambda x: x, decoder_inputs)):

        if loop_function is not None and prev is not None:
            with tf.variable_scope("loop_function", reuse=True):
                inp = loop_function(prev, i)
        if i > 0:
            tf.get_variable_scope().reuse_variables()
        c_i = attention(state, enc_outputs)
        inp = tf.concat([inp, c_i], axis=1)
        output, state = cell(inp, state)
        # print output.eval()
        outputs.append(output)
        if loop_function is not None:
            prev = output
    return outputs

# def loop_function(prev, _):
#     """
#     :param prev: the output of t-1 time
#     :param _:
#     :return: the embedding of t-1 output
#     """
#     prev = tf.add(tf.matmul(prev, softmax_w), softmax_b)
#     prev_sympol = tf.arg_max(prev, 1)
#
#     emb_prev = tf.nn.embedding_lookup(target_embedding, prev_sympol)
#     return emb_prev

init_state = tf.zeros([3, encoder_hidden_units], tf.float32)
decoderRe = decode(cell, init_state, encoder_outputs)

# def loop_fn(time, cell_output, cell_state, loop_state):
#
#     def get_step_input():
#         # global time
#         return encoder_time_batch[time]
#
#     emit_output = cell_output  # == None if time = 0
#     if cell_output is None:  # time = 0
#         next_cell_state = cell.zero_state([batch_size, input_size], tf.float32)
#         _initial_state = next_cell_state
#     else:
#         next_cell_state = cell_state
#
#     elements_finished = (time >= num_steps-1)
#     finished = tf.reduce_all(elements_finished)
#     next_input = tf.cond(finished,
#                          lambda: tf.zeros([batch_size, input_size], dtype=tf.float32), get_step_input)
#     time = time + 1
#
#     # apply linear + sig transform here
#     # print("before lin+sig", next_input)
#     # next_input = _linear_transform(next_input)  # [32, 200] --> [32, 1]
#     # print("after lin+sig", next_input)
#     # next_input = tf.contrib.layers.linear(next_input, input_size)
#
#     next_loop_state = None
#     return elements_finished, next_input, next_cell_state, emit_output, next_loop_state
# outputs_ta, final_state = tf.nn.raw_rnn(cell, loop_fn)
# outputs = outputs_ta.stack()


loss_track = []
epochs = 30001
batches_in_epoch = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    try:
        fd = get_data(3)
        for batch in range(2):

            enoutputs = sess.run(decoderRe, fd)
            # print(_in.shape)
            print(enoutputs.shape)

    except KeyboardInterrupt:
        print('training interrupted')
    # print(sess.run(embeddings))
