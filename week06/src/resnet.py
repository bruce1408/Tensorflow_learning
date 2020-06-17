import os
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def idBlock(input, filters, kernels, training):
    x1 = tf.layers.conv2d(input, filters[0], kernels[0], padding='same')
    x2 = tf.layers.batch_normalization(x1)


def convBlock():
    pass


"""
7-layer fully connected neural network
"""

__author__ = "lizhongding"


def one_hot_encoding(x, depth=10):
    length = len(x)
    coder = np.zeros([length, depth])
    for i in range(length):
        coder[i, x[i]] = 1
    return coder


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255
y_train = one_hot_encoding(y_train)
y_test = one_hot_encoding(y_test)

BATCH_SIZE = 64
EPOCHS = 50
NUM_BATCHES = x_train.shape[0] // BATCH_SIZE

x = tf.placeholder(tf.float32, [None, 784], 'input_x')
y = tf.placeholder(tf.int32, [None, 10], 'input_y')

w1 = tf.Variable(tf.truncated_normal([784, 1024]))
b1 = tf.Variable(tf.truncated_normal([1, 1024]))

w2 = tf.Variable(tf.truncated_normal([1024, 512]))
b2 = tf.Variable(tf.truncated_normal([1, 512]))

w3 = tf.Variable(tf.truncated_normal([512, 512]))
b3 = tf.Variable(tf.truncated_normal([1, 512]))

w4 = tf.Variable(tf.truncated_normal([512, 512]))
b4 = tf.Variable(tf.truncated_normal([1, 512]))

w5 = tf.Variable(tf.truncated_normal([512, 256]))
b5 = tf.Variable(tf.truncated_normal([1, 256]))

w6 = tf.Variable(tf.truncated_normal([256, 64]))
b6 = tf.Variable(tf.truncated_normal([1, 64]))

w7 = tf.Variable(tf.truncated_normal([64, 10]))
b7 = tf.Variable(tf.truncated_normal([1, 10]))

is_train = tf.placeholder_with_default(False, (), 'is_train')

h1 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.add(tf.matmul(x, w1), b1), training=is_train))
h2 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.add(tf.matmul(h1, w2), b2), training=is_train))
h3 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.add(tf.matmul(h2, w3), b3), training=is_train))
h4 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.add(tf.matmul(h3, w4), b4), training=is_train))
h5 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.add(tf.matmul(h4, w5), b5), training=is_train))
h6 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.add(tf.matmul(h5, w6), b6), training=is_train))
h7 = tf.nn.leaky_relu(tf.add(tf.matmul(h6, w7), b7))

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=h7))

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer = tf.train.AdamOptimizer().minimize(loss)

accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y, 1), tf.argmax(h7, 1))))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        for i in range(NUM_BATCHES):
            sess.run(optimizer, feed_dict={
                x: x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE - 1, :],
                y: y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE - 1, :], is_train: True})  # 可通过修改该参数打开或关闭 BN。
        print("After Epoch {0:d}, the test accuracy is {1:.4f} ".
              format(epoch + 1, sess.run(accuracy, feed_dict={x: x_test, y: y_test})))
    print("Finished!")
