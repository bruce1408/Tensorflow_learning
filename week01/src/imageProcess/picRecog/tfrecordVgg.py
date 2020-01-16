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
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

# Dataset Parameters
pathNew = "/home/bruce/bigVolumn/Datasets/aptos/train_data/"
# # use cpu to train the model
# path = "/home/bruce/bigVolumn/Datasets/aptos/train_images/"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Image Parameters
N_CLASSES = 5  # CHANGE HERE, total number of classes
IMG_HEIGHT = 1736  # CHANGE HERE, the image width to be resized to
IMG_WIDTH = 1736  # CHANGE HERE, the image height to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 16
display_step = 100
num_epoch = 30
dropout = 0.5  # Dropout, probability to keep units


def rawImageData():
    """
    对原始图片存储的位置放到list中
    :return:the list of traindata, trainlabel, testdata, testlabel, each list contain the file abs_path
    """
    df = pd.read_csv("/home/bruce/bigVolumn/Datasets/aptos/train.csv")
    df.id_code = pathNew + df.id_code.apply(str) + '.png'
    imagepaths = df.id_code.tolist()
    labels = df['diagnosis'].tolist()
    print(imagepaths)
    print(labels)

    assert len(imagepaths) == len(labels)
    num_examples = len(imagepaths)
    n_truct = 3600  # 人为设置的
    num_train = int(0.8 * 3600)  # 2880

    print('the num train is:', num_train)
    trainData = imagepaths[:num_train]  # 0-2880
    trainLabel = labels[:num_train]  # 0-2880
    testData = imagepaths[num_train:3600]  #
    testLabel = labels[num_train:3600]
    return trainData, trainLabel, testData, testLabel


def createTFrecord(data, labels, tfrecordName):

    writer = tf.python_io.TFRecordWriter(tfrecordName)  # 打开tfrecord文件

    for index, image_name in enumerate(data):
        img = Image.open(image_name)
        img_raw = img.tobytes()
        # 所有feature送入字典,且转成tf_example
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化且写入该样本
    writer.close()  # 关闭tfrecord file


def buildData():
    """
    Build the data input, Once this is executed, it will not be executed next time.
    :return: generate trainData, testData tfrecord files
    """
    traindata, trainlabel, testdata, testlabel = rawImageData()
    createTFrecord(traindata, trainlabel, "trainData_v2.tfrecords")
    createTFrecord(testdata, testlabel, "testData_v2.tfrecords")


def readTFrecord(TFRecord_file, batch_size, num_epoch):
    """
    read the Tfrecord file, and return the num of batch_size train,test data
    :param TFRecord_file:
    :param batch_size:
    :param num_epoch: the num of epoch,the value must be same with session part train.
    :return:
    """
    with tf.name_scope('input'):
        # tf.train.string_input_producer
        # 将文件名列表交给tf.train.string_input_producer 函数.
        # 生成一个先入先出的队列，文件阅读器会需要它来读取数据。
        # Returns: A QueueRunner for the Queue is added to the current Graph's
        # QUEUE_RUNNER collection
        filename_queue = tf.train.string_input_producer([TFRecord_file], num_epochs=num_epoch)

        # init TFRecordReader class
        reader = tf.TFRecordReader()
        key, values = reader.read(filename_queue)  # filename_queue

        # parse_single_example将Example协议内存块(protocol buffer)解析为张量
        features = tf.parse_single_example(values,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })
        # decode to tf.uint8
        image = tf.decode_raw(features['img_raw'], tf.uint8)  # tf.uint8
        # image cast
        image = tf.cast(image, tf.float32)  # tf.uint8 to tf.float32

        # reshape
        image = tf.reshape(image, [1736, 1736, 3])
        image = tf.image.per_image_standardization(image)  # 图片归一化
        label = tf.cast(features['label'], tf.int32)  # label tf.int32

        # create batchs of tensors
        # This function is implemented using a queue.A QueueRunner for the queue
        # is added to the current Graph's QUEUE_RUNNER collection.
        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=128,
                                                               target_width=128)
        image_batch, label_batch = tf.train.batch([resized_image, label],
                                                  batch_size=batch_size,
                                                  num_threads=4,
                                                  capacity=16)
        label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[batch_size, 128, 128, 3])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
        conv1_1 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool1 = tf.layers.max_pooling2d(conv1_1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)

        conv3_1 = tf.layers.conv2d(pool2, 512, 3, activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(conv3_1, 512, 3, activation=tf.nn.relu)
        conv3_3 = tf.layers.conv2d(conv3_2, 512, 3, activation=tf.nn.relu)
        conv3_4 = tf.layers.conv2d(conv3_3, 512, 3, activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3_4, 2, 2)

        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu)
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu)
        conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, activation=tf.nn.relu)
        conv4_4 = tf.layers.conv2d(conv4_3, 512, 3, activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4_4, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(pool4)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 4096)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 2048)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out
        # out = tf.nn.softmax(out)
    return out


X_train, Y_train = readTFrecord("./tfData/trainData_v2.tfrecords", batch_size, num_epoch)
X_test, Y_test = readTFrecord("./tfData/testData_v2.tfrecords", batch_size, num_epoch)
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

# Evaluate model (with test logits, for dropout to be disabled)
Y_test = tf.one_hot(Y_test, 5)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# # Saver object
saver = tf.train.Saver(max_to_keep=1)
config = tf.ConfigProto()
config.gpu_options.allocator_type = "BFC"
config.gpu_options.allow_growth = True

# Start training
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    batch_num = int(2880/batch_size)  # batch_num 180
    print("batch_num is:", batch_num)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    """
    在创建文件名队列之后整个系统处于停滞状态,文件名没有真正的加入到队列中去,如果此时开始计算,那么内存队列什么都没有,计算等待
    系统阻塞,但是tf.train.start_queue_runners函数填充队列,可以后续用于计算
    """
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    ckpt = tf.train.get_checkpoint_state("./model2")


    bestAcc = 0.0
    if ckpt is None:
        print("please train the model first!")
    else:
        path = ckpt.model_checkpoint_path
        print("loading pre-trained model from the %s..." % path)
        saver.restore(sess, path)
    for epoch in range(num_epoch):
        for step in range(batch_num):
            _, loss = sess.run([train_op, loss_op])
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))
        acc, loss_ = sess.run([accuracy, loss_op])
        # if acc > bestAcc:
        #     bestAcc = max(acc, bestAcc)
        #     path_name = "./model2/model" + str(epoch) + ".ckpt"
        #     saver.save(sess, path_name)
        #     print("model has been saved!")
        print("epoch " + str(epoch) + " accuracy= " + "{:.4f}".format(acc)+" ,minBatchloss= "+"{:.4f}".format(loss_))
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)
