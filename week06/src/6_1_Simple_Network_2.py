# encoding=utf-8
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
# input data
mnist_data = input_data.read_data_sets(
    '../datasets/fashion_mnist', one_hot=True)
batch_size = 100
n_batch = mnist_data.train.num_examples // batch_size
epoch = 21


# define the structure of network
def weight_initial(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_initial(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# define two placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一个卷积层
w1 = weight_initial([5, 5, 1, 32])
b1 = bias_initial([32])

conv1 = tf.nn.relu(conv2d(x_image, w1) + b1)
pool1 = pooling(conv1)

# 第二个卷积层
w2 = weight_initial([5, 5, 32, 64])
b2 = bias_initial([64])

conv2 = tf.nn.relu(conv2d(pool1, w2) + b2)
pool2 = pooling(conv2)

# 初始化全连接层
w3 = weight_initial([7 * 7 * 64, 1024])
b3 = bias_initial([1024])

pool2_flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])
funnlink = tf.nn.relu(tf.matmul(pool2_flatten, w3) + b3)

keep_prob = tf.placeholder(tf.float32)
fulllink_prob = tf.nn.dropout(funnlink, keep_prob)

# 第二个全连接层
w4 = weight_initial([1024, 10])
b4 = bias_initial([10])


prediction = tf.nn.softmax(tf.matmul(fulllink_prob, w4) + b4)


# define cross entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 优化步骤
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
# 结果保存在布尔型
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    count = 0
    for eachEpoch in range(epoch):
        for i in range(n_batch):
            count += 1
            batch_x, batch_y = mnist_data.train.next_batch(batch_size)
            acctrain, _ = sess.run([accuracy, train_step], feed_dict={
                                   x: batch_x, y: batch_y, keep_prob: 0.7})
            if i % 200 == 0:
                print("Step " + str(int(count/100)) +
                      ", training Accuracy= " + str(acctrain))
        acc = sess.run(accuracy, feed_dict={
                       x: mnist_data.test.images, y: mnist_data.test.labels, keep_prob: 1.0})
        print("Epoch " + str(eachEpoch) + ", Testing Accuracy= " + str(acc))
