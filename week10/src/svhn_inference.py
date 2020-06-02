import os
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

# Image Parameters
N_CLASSES = 11  # CHANGE HERE, total number of classes
IMG_HEIGHT = 224  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 224  # CHANGE HERE, the image width to be resized to
CHANNELS = 1  # The 3 color channels, change to 1 if grayscale
n_classes = N_CLASSES  # MNIST total classes (0-9 digits)
dropout = 0.35
num_steps = 10000
train_display = 100
val_display = 300
learning_rate = 0.0001
BATCHSIZE = 64


def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # Convolution Layer with 32 filters and a kernel size of 3
        x = tf.reshape(x, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 3])
        # Convolution Layer with 32 filters and a kernel size of 3
        conv1 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
        conv1_1 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool1 = tf.layers.max_pooling2d(conv1_1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)

        conv3_1 = tf.layers.conv2d(pool2, 256, 3, activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, activation=tf.nn.relu)
        conv3_3 = tf.layers.conv2d(conv3_2, 256, 3, activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3_3, 2, 2)

        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu)
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu)
        conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4_3, 2, 2)

        conv5_1 = tf.layers.conv2d(pool4, 512, 3, activation=tf.nn.relu)
        conv5_2 = tf.layers.conv2d(conv5_1, 512, 3, activation=tf.nn.relu)
        conv5_3 = tf.layers.conv2d(conv5_2, 512, 3, activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv5_3, 2, 2)
        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(pool4)

        # Fully connected layer (in contrib folder for now)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dense(fc1, 4096)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 4096)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
        # Output layer, class prediction
        digit1 = tf.layers.dense(fc2, n_classes)
        digit2 = tf.layers.dense(fc2, n_classes)
        digit3 = tf.layers.dense(fc2, n_classes)
        digit4 = tf.layers.dense(fc2, n_classes)
        digit5 = tf.layers.dense(fc2, 6)

        digit1 = tf.nn.softmax(digit1) if not is_training else digit1
        digit2 = tf.nn.softmax(digit2) if not is_training else digit2
        digit3 = tf.nn.softmax(digit3) if not is_training else digit3
        digit4 = tf.nn.softmax(digit4) if not is_training else digit4
        digit5 = tf.nn.softmax(digit5) if not is_training else digit5

        # we only apply softmax to testing network
        # out = tf.nn.softmax(out) if not is_training else out
    return digit1, digit2, digit3, digit4, digit5

