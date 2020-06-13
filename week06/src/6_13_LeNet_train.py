import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Image Parameters
N_CLASSES = 10  # CHANGE HERE, total number of classes
IMG_HEIGHT = 32  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 32  # CHANGE HERE, the image width to be resized to
CHANNELS = 1  # The 3 color channels, change to 1 if grayscale


# Parameters
epoch = 5
learning_rate = 0.0001
num_steps = 10000
batch_size = 32
display_step = 100
val_display = 300
save_check = 1000
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)
batch_count = int(mnist.train.num_examples / batch_size)

X = tf.placeholder(tf.float32, [None, 784], name="x-input")
Y = tf.placeholder(tf.float32, [None, 10], name="y-input")
X_ = tf.reshape(X, [-1, 28, 28, 1])


# -----------------------------------------------
# LENET NETWORK
# -----------------------------------------------
def conv_net(x, n_classes, reuse, is_training):
    """
    Create model padding有两种类型，一种是valid，还有一种是same:
    valid表示不够卷积核大小就丢弃，
    same表示不够的话就补0
    max_pooling2d 默认的padding是valid，就是说不够的话丢弃，否则same补充0；

    LeNet 网络结构实现:
    输入为32*32单通单图片,
    卷积层: filter 大小是 5 * 5 步长为1, 输出通道是 6
    池化层: filter 大小是 2 * 2 步长是2
    卷积层: filter 大小是 5 * 5 步长是1, 输出通道是16
    池化层: filter 大小是 2 * 2 步长是2
    全连接层: 120 个神经元
    全连接层: 84 个神经元
    全连接层: 2个神经元
    总共是 5 个卷积层(含全连接3个)

    :param x: input size = [batch_size, heght, width, channels]
    :param n_classes: the num of class
    :param reuse: resure the conver weights
    :param is_training: training or validation
    :return: the prob of the multiclass task
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('LeNet', reuse=reuse):
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, activation=tf.nn.sigmoid)
        # average Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(pool1, 16, 5, activation=tf.nn.sigmoid)
        # average Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(pool2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 120)
        fc1 = tf.layers.dense(fc1, 84)
        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X_, N_CLASSES, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights, but has
logits_test = conv_net(X_, N_CLASSES, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('./model_lenet')
    if ckpt is None:
        print("Model is not found, please train your model first...")
    else:
        path = ckpt.model_checkpoint_path
        print("loading the pre-trained model from %s " % path)
        saver.restore(sess, path)
    sess.run(init)
    for i in range(epoch):
        for j in range(batch_count):
            x_train, y_train = mnist.train.next_batch(batch_size)
            loss_train, _, acc = sess.run([loss_op, train_op, accuracy], feed_dict={X: x_train, Y: y_train})
            if j % 200 == 0:  # print out the train result every j times
                print("MiniBatch Loss is: %f, the Training acc is: %f" % (loss_train, acc))

        acc, loss_val = sess.run([accuracy, loss_op], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        print("="*58)
        print("Epoch " + str(i) + ", Testing Accuracy= " + str(acc) + " loss = " + str(loss_val))
        print("="*58)
        saver.save(sess, "./model_lenet/model.ckpt")

print("Optimization Finished!")
