import tensorflow as tf
import numpy as np
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
<<<<<<< HEAD
mnist = input_data.read_data_sets('Users/bruce/program/Python/datasets/fashion_mnist', one_hot=True)
=======
mnist = input_data.read_data_sets('/home/bruce/bigVolumn/Datasets/fashion_mnist', one_hot=True)
>>>>>>> 054737601996624b90ae40bb35d43875d014de0b

# set params
n_inputs = 28
n_steps = 28
rnn_size = 256
n_output_layer = 10
batch_size = 100

# define placeholder
X = tf.placeholder('float', [None, n_steps, n_inputs])
Y = tf.placeholder('float', [None])


# define RNN function
def recurrent_neural_network(data):
    layer = {'w_': tf.Variable(tf.random_normal([rnn_size, n_output_layer])),
             'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, n_inputs])
    data = tf.split(data, n_steps)
    outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)
    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
    return ouput


def train_neural_network(X, Y):
    predict = recurrent_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predict))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost_func)
    epochs = 10
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples / batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([batch_size, n_steps, n_inputs])
                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy: ', session.run({X: mnist.test.images.reshape(-1, n_steps, n_inputs), Y: mnist.test.labels}))


train_neural_network(X, Y)