# coding=utf-8
import tensorflow as tf
import numpy as np
import generate_3dim_data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
np.random.seed(1)
tf.set_random_seed(1)


PAD = 0
EOS = 1
vocab_size = 10
batch_size = 10
feature_size = 13
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

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

# embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
# encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
# encoder_inputs_embedded = tf.Print(encoder_inputs_embedded, [tf.shape(encoder_inputs_embedded)],
# message='encoder_inputs_embed')
# decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
# decoder_inputs_embedded = tf.Print(decoder_inputs_embedded, [tf.shape(decoder_inputs_embedded)],
# message='decoder_inputs_embed')
# define encoder layer

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, dtype=tf.float32)
del encoder_outputs  # 非attention机制，不需要encoder_output

# define decoder layer
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell,
    decoder_inputs,  # 输入
    initial_state=encoder_final_state,
    dtype=tf.float32,
    time_major=False,
    scope='plain_decoder',)
# decoder_outputs shape = [None, None, 20]

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)  # shape [batch, max_step, 10]
# decoder_logits = tf.Print(decoder_logits, [tf.shape(decoder_logits)], message='decoder_logits')
# decoder_prediction = tf.argmax(decoder_logits, 2)  # shape[None, None] = [batch, max_step]
decoder_prediction = tf.layers.dense(decoder_logits, 8)  # [batch_size, max_step, dim_size]
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
            # print('enfs\n', enfs.__len__())
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
    # print(sess.run(embeddings))
