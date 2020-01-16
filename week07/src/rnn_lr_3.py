import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
# tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(1)
#
# mnist = input_data.read_data_sets('/Users/bruce/programme/Python/datasets/MNIST_data', one_hot=True)
#
# # hyperparameters
# lr = 0.001
# training_iters = 100000
# batch_size = 128
#
# n_inputs = 28  # shape 28*28
# n_steps = 28  # time steps
# n_hidden_unis = 128  # neurons in hidden layer
# n_classes = 10  # classes 0-9
#
#
# # tf Graph input
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# y = tf.placeholder(tf.float32, [None, n_classes])
#
#
# # Define weights  in # (28,128)  out # (128,10)
# weights = {'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
#            'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))}
#
# # (128,)  # (10,)
# biases = {'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])),
#           'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}
#
#
# def RNN(X, weights, biases):
#     # hidden layer for input to cell
#
#     # X(128 batch, 28 steps, 28 inputs) => (128*28, 28)
#     X = tf.reshape(X, [-1, n_inputs])
#
#     # ==>(128 batch * 28 steps, 28 hidden)
#     X_in = tf.matmul(X, weights['in'])+biases['in']
#     # ==>(128 batch , 28 steps, 28 hidden)
#     X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unis])
#     # cell
#     lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
#     # lstm cell is divided into two parts(c_state, m_state)
#     _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
#     outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
#     # hidden layer for output as the final results
#     results = tf.matmul(states[1], weights['out']) + biases['out']  # states[1]->m_state states[1]=output[-1]
#     # outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
#     # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
#     return results, states
#
#
# pred, state = RNN(x, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# train_op = tf.train.AdamOptimizer(lr).minimize(cost)
#
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     step = 0
#     while step * batch_size < training_iters:
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
#         sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
#         if step % 20 == 0:
#             print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
#         step += 1


# 单层RNN 神经网络训练mnist
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# n_steps = 28
# n_inputs = 28
# n_neurons = 150
# n_outputs = 10
mnist = input_data.read_data_sets("/home/bruce/bigVolumn/Datasets/fashion_mnist")
# X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
# y_test = mnist.test.labels
#
#
# learning_rate = 0.001
#
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# y = tf.placeholder(tf.int32, [None])
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
#
# logits = tf.layers.dense(states, n_outputs)  # 这个函数超级牛逼，不需要你自己再定义全链接函数，只需要给输出的位数就可以
#
# xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
# loss = tf.reduce_mean(xentropy)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# correct = tf.nn.in_top_k(logits, y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1), tf.float32)))
#
# init = tf.global_variables_initializer()
#
#
# n_epochs = 100
# batch_size = 150
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for iteration in range(mnist.train.num_examples // batch_size):
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             X_batch = X_batch.reshape((-1, n_steps, n_inputs))
#             sess.run(optimizer, feed_dict={X: X_batch, y: y_batch})
#         acc_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
#         acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#         print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
#         print(outputs.get_shape())
        # print(' : ', logits.shape)
        # print()

# 多层RNN神经网络训练 mnist
n_steps = 28
n_inputs = 28
n_outputs = 10
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels
n_neurons = 100
n_layers = 3

layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

states_concat = tf.concat(axis=1, values=states)  # 将三层的状态全部叠加起来
logits = tf.layers.dense(states_concat, n_outputs)  # 这一步太关键了。
# import pdb
# pdb.set_trace()
# logits = tf.Print(logits, [logits], name='logits', summarize=100)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)  # 这个用sparse softmax 正好
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()


n_epochs = 10
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(optimizer, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        print(logits.shape)
        print(outputs.shape)
