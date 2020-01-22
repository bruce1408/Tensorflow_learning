""" Build an Image Dataset in TensorFlow.

For this example, you need to make your own set of images (JPEG).
We will show 2 different ways to build that dataset:

- From a root folder, that will have a sub-folder containing images for each class
    ```
    ROOT_FOLDER
       |-------- SUBFOLDER (CLASS 0)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
       |
       |-------- SUBFOLDER (CLASS 1)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
    ```

- From a plain text file, that will list all images with their class ID:
    ```
    /path/to/image/1.jpg CLASS_ID
    /path/to/image/2.jpg CLASS_ID
    /path/to/image/3.jpg CLASS_ID
    /path/to/image/4.jpg CLASS_ID
    etc...
    ```

Below, there are some parameters that you need to change (Marked 'CHANGE HERE'),
such as the dataset path.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
# from __future__ import print_function

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset Parameters - CHANGE HERE
MODE = 'folder'  # or 'file', if you choose a plain text file (see above).
DATASET_PATH = '/home/bruce/bigVolumn/Datasets/aptos/train_images'  # the dataset file or root folder path.
pathNew = "/home/bruce/bigVolumn/Datasets/aptos/train_data/"
path = "/home/bruce/bigVolumn/Datasets/aptos/train_images/"

# Image Parameters
N_CLASSES = 5  # CHANGE HERE, total number of classes
IMG_HEIGHT = 128  # CHANGE HERE, the image width to be resized to
IMG_WIDTH = 128  # CHANGE HERE, the image height to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale
num_epoch = 5
batch_size = 16


# Reading the dataset
# def read_images(dataset_path, batch_size):
#     df = pd.read_csv("/home/bruce/bigVolumn/Datasets/aptos/train.csv")
#     df.id_code = pathNew+df.id_code.apply(str)+'.png'
#     imagepaths = df.id_code.tolist()
#     labels = df['diagnosis'].tolist()
#     print(imagepaths)
#
#     assert len(imagepaths) == len(labels)
#     writer = tf.python_io.TFRecordWriter("train_data.tfrecords")
#
#     # # Convert to Tensor
#     imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
#     labels = tf.convert_to_tensor(labels, dtype=tf.int32)
#     # Build a TF Queue, shuffle data
#     image, label = tf.train.slice_input_producer([imagepaths, labels],
#                                                  shuffle=True)
#
#     # Read images from disk
#     image = tf.read_file(image)
#     image = tf.image.decode_jpeg(image, channels=CHANNELS)
#
#     # Resize images to a common size
#     image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
#
#     # Normalize
#     image = image * 1.0/127.5 - 1.0
#
#     # Create batches
#     X, Y = tf.train.batch([image, label], batch_size=batch_size,
#                           capacity=batch_size * 8,
#                           num_threads=4)
#
#     return X, Y


def rawImageData():
    """
    对原始图片存储的位置放到列表中
    :return:
    """
    df = pd.read_csv("/home/bruce/bigVolumn/Datasets/aptos/train.csv")
    df.id_code = pathNew + df.id_code.apply(str) + '.png'
    imagepaths = df.id_code.tolist()
    labels = df['diagnosis'].tolist()
    print(imagepaths)
    print(labels)

    assert len(imagepaths) == len(labels)
    trainData, testData, trainLabel, testLabel = train_test_split(imagepaths, labels, test_size=0.25, random_state=0)

    print('the num train is:', len(trainData))
    print("the num test is ", len(testData))

    return trainData, trainLabel, testData, testLabel


def createTFrecord(imagepaths, labels, batch_size):
    # # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)
    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    # Normalize
    # image = image * 1.0 / 127.5 - 1.0
    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 8, num_threads=4)
    return X, Y


trainData, trainLabel, testData, testLabel = rawImageData()
X_train, Y_train = createTFrecord(trainData, trainLabel, batch_size)
X_test, Y_test = createTFrecord(testData, testLabel, batch_size)

# Parameters
learning_rate = 0.1
num_steps = 1000
display_step = 100

# Network Parameters
dropout = 0.5  # Dropout, probability to keep units


# ---- random num of conv layer
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[batch_size, 128, 128, 3])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        # out = tf.nn.softmax(out) if not is_training else out
        out = tf.nn.softmax(out)
    return out


# def conv_net(x, n_classes, dropout, reuse, is_training):
#     # Define a scope for reusing the variables
#     with tf.variable_scope('ConvNet', reuse=reuse):
#         # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
#         # Reshape to match picture format [Height x Width x Channel]
#         # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
#         x = tf.reshape(x, shape=[batch_size, 224, 224, 3])
#
#         # Convolution Layer with 32 filters and a kernel size of 5
#         conv1 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
#         conv1_1 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
#         # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
#         pool1 = tf.layers.max_pooling2d(conv1_1, 2, 2)
#
#         # Convolution Layer with 32 filters and a kernel size of 5
#         conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu)
#         conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu)
#         # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
#         pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)
#
#         conv3_1 = tf.layers.conv2d(pool2, 512, 3, activation=tf.nn.relu)
#         conv3_2 = tf.layers.conv2d(conv3_1, 512, 3, activation=tf.nn.relu)
#         conv3_3 = tf.layers.conv2d(conv3_2, 512, 3, activation=tf.nn.relu)
#         conv3_4 = tf.layers.conv2d(conv3_3, 512, 3, activation=tf.nn.relu)
#         pool3 = tf.layers.max_pooling2d(conv3_4, 2, 2)
#
#         conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu)
#         conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu)
#         conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, activation=tf.nn.relu)
#         conv4_4 = tf.layers.conv2d(conv4_3, 512, 3, activation=tf.nn.relu)
#         pool4 = tf.layers.max_pooling2d(conv4_4, 2, 2)
#
#         # Flatten the data to a 1-D vector for the fully connected layer
#         fc1 = tf.contrib.layers.flatten(pool4)
#
#         # Fully connected layer (in contrib folder for now)
#         fc1 = tf.layers.dense(fc1, 4096)
#         # Apply Dropout (if is_training is False, dropout is not applied)
#         fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
#
#         fc2 = tf.layers.dense(fc1, 4096)
#         fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
#         # Output layer, class prediction
#         out = tf.layers.dense(fc2, n_classes)
#         # Because 'softmax_cross_entropy_with_logits' already apply softmax,
#         # we only apply softmax to testing network
#         out = tf.nn.softmax(out) if not is_training else out
#     return out


# # Because Dropout have different behavior at training and prediction time, we
# # need to create 2 distinct computation graphs that share the same weights.
# # Create a graph for training
logits_train = conv_net(X_train, 5, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights, but has
# different behavior for 'dropout' (not applied).
logits_test = conv_net(X_test, 5, 1.0, reuse=True, is_training=False)
# Define loss and optimizer (with train logits, for dropout to take effect)
# loss_op = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(logits_train), reduction_indices=[1]))
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=Y_train))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# logits_train = conv_net(X_train, 5, dropout, reuse=False, is_training=True)
# logits_test = conv_net(X_train, 5, dropout, reuse=True, is_training=False)
# # Define loss and optimizer (with train logits, for dropout to take effect)
# # loss_op = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(logits_train), reduction_indices=[1]))
# loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=Y_train))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op, global_step=tf.train.get_or_create_global_step())

# Evaluate model (with test logits, for dropout to be disabled)
# Y_test = tf.one_hot(Y_test, 5)

correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y_test, tf.int64))
# correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# # Saver object
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allocator_type = "BFC"
config.gpu_options.allow_growth = True

# Start training
with tf.Session(config=config) as sess:
    batch_num = int(2880 / batch_size)  # batch_num 180
    print("batch_num is:", batch_num)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    """
    在创建文件名队列之后整个系统处于停滞状态,文件名没有真正的加入到队列中去,如果此时开始计算,那么内存队列什么都没有,计算等待
    系统阻塞,但是tf.train.start_queue_runners函数填充队列,可以后续用于计算
    """
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    ckpt = tf.train.get_checkpoint_state("./model1")
    if ckpt is None:
        print("please train the model first!")
    else:
        path = ckpt.model_checkpoint_path
        print("loading pre-trained model from the %s..." % path)
        saver.restore(sess, path)
    # Training cycle
    accbest = 0.0
    for epoch in range(num_epoch):
        for step in range(batch_num):
            # Run optimization and calculate batch loss and accuracy
            _, loss = sess.run([train_op, loss_op])

            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))

        acc, loss = sess.run([accuracy, loss_op])
        print("epoch " + str(epoch) + " accuracy= " + "{:.4f}".format(acc) + " loss= " + "{:.4f}".format(loss))
        # path_name = "./model1/model" + str(epoch) + ".ckpt"
        # if acc > accbest:
        #     accbest = max(accbest, acc)
        # saver.save(sess, path_name)
        # print("model has been saved!")
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)

# max_step = 50000
# batch_size = 4
# data_dir = './Data/train.tfrecords'
#
# def read_and_decode(filename_queue, batchsize, random_crop=False, random_clip=False, shuffle_batch=True):
#     reader = tf.TFRecordReader()
#
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(
#         serialized_example, features={
#             'height': tf.FixedLenFeature([], tf.int64),
#             'width': tf.FixedLenFeature([], tf.int64),
#             'label': tf.FixedLenFeature([], tf.int64),
#             'image_raw': tf.FixedLenFeature([], tf.string)
#         }
#     )
#
#     image = tf.decode_raw(features['image_raw'], tf.float32)
#     height = tf.cast(features['height'], tf.int32)
#     # width = features['width']
#     width = tf.cast(features['width'], tf.int32)
#     label = tf.cast(features['label'], tf.float32)
#     image = tf.reshape(image, [32, 32, 3])
#
#     # image = tf.image.resize_images(image, [28, 28])
#
#     if shuffle_batch:
#         images, labels, widths, heights = tf.train.shuffle_batch(
#             [image, label, width, height],
#             batch_size=batchsize,
#             num_threads=10,
#             capacity=1000,
#             min_after_dequeue=10)
#     else:
#         images, labels, widths, heights = tf.train.batch(
#             [image, label, width, height],
#             batch_size=batchsize,
#             num_threads=10,
#             capacity=1000)
#     return images, labels, widths, heights
#
#
# def variable_with_wight_loss(shape, stddev, wl):
#     var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
#     if wl is not None:
#         weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
#         tf.add_to_collection('losses', weight_loss)
#     return var
#
#
# def build_graph(images_holder, labels):
#     weight1 = variable_with_wight_loss([5, 5, 3, 64], stddev=1e-4, wl=0.0)
#     kernel1 = tf.nn.conv2d(images_holder, weight1, [1, 1, 1, 1], padding='SAME')
#     bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
#     conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
#     pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#     weight2 = variable_with_wight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
#     kernel2 = tf.nn.conv2d(pool1, weight2, [1, 1, 1, 1], padding='SAME')
#     bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
#     conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
#     pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#     # full connection
#     reshape = tf.reshape(pool2, [batch_size, -1])
#     dim = reshape.get_shape()[1].value
#     weight3 = variable_with_wight_loss(shape=[dim, 384], stddev=0.04, wl=0.04)
#     bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
#     local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
#
#     weight4 = variable_with_wight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
#     bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
#     local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
#
#     weight5 = variable_with_wight_loss(shape=[192, 5], stddev=1 / 192.0, wl=0.0)
#     bias5 = tf.Variable(tf.constant(0.0, shape=[5]))
#     logits = tf.add(tf.matmul(local4, weight5), bias5)
#
#     # loss
#     labels = tf.cast(labels, tf.int64)
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=logits, labels=labels, name='cross_entropy_per_example')
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#     tf.add_to_collection('losses', cross_entropy_mean)
#
#     loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
#     train_op = tf.train.AdagradOptimizer(1e-3).minimize(loss)
#     top_K_op = tf.nn.in_top_k(logits, labels, 1)
#
#     return train_op, loss, top_K_op
#
#
# train_tfrecord_filename = './data/train.tfrecords'
# train_filename_queue = tf.train.string_input_producer([train_tfrecord_filename], num_epochs=10)
#
# images, labels, widths, heights = read_and_decode(train_filename_queue, batch_size, shuffle_batch=True)
#
# with tf.Session() as sess:
#
#     train_op, loss, top_K_op = build_graph(images, labels)
#
#     tf.local_variables_initializer().run()
#     # local_variables like epoch_num, batch_size
#     tf.global_variables_initializer().run()
#
#     coord = tf.train.Coordinator()
#     tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     for step in range(max_step):
#         start_time = time.time()
#         _, loss_value = sess.run([train_op, loss])
#         # print(sess.run(labels))
#
#         duration = time.time() - start_time
#         if step % 10 == 0:
#             sec_per_batch = float(duration)
#             format_str = ('%s: step %d,loss=%.4f (%.3f sec/batch)')
#             print(format_str % (datetime.now(), step, loss_value, sec_per_batch))
#         if step % 1000 == 0:
#             true_count = 0
#             for i in range(10):
#                 print(str(i))
#                 print(top_K_op)
#                 predictions = sess.run([top_K_op])
#
#                 true_count += np.sum(predictions)
#             precision = true_count / (batch_size * 10)
#             print('precision @ 1 = %.3f' % precision)
#
#     coord.request_stop()
#     coord.join()
