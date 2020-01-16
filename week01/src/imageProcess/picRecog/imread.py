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
from skimage import io
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from datetime import datetime
import time

# Dataset Parameters - CHANGE HERE
MODE = 'folder'  # or 'file', if you choose a plain text file (see above).
DATASET_PATH = '/home/bruce/bigVolumn/Datasets/aptos/train_images'  # the dataset file or root folder path.
pathNew = "/home/bruce/bigVolumn/Datasets/aptos/train_data/"
path = "/home/bruce/bigVolumn/Datasets/aptos/train_images/"

# Image Parameters
N_CLASSES = 5  # CHANGE HERE, total number of classes
IMG_HEIGHT = 1736  # CHANGE HERE, the image width to be resized to
IMG_WIDTH = 1736  # CHANGE HERE, the image height to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale

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

def read_images(batch_size):
    df = pd.read_csv("/home/bruce/bigVolumn/Datasets/aptos/train.csv")
    df.id_code = pathNew+df.id_code.apply(str)+'.png'
    imagepaths = df.id_code.tolist()
    labels = df['diagnosis'].tolist()
    print(imagepaths)
    print(labels)

    assert len(imagepaths) == len(labels)
    writer = tf.python_io.TFRecordWriter("train_data.tfrecords")

    for image_name, label in zip(imagepaths, labels):
        img = io.imread(image_name)
        img_str = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print("the tfrecord data is done!")


# read_images(16)


# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# Note that a few elements have changed (usage of queues).

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 16
display_step = 100

# Network Parameters
dropout = 0.5  # Dropout, probability to keep units

# Build the data input
# read_images(batch_size)


def readTFrecord(TFRecord_file, batch_size, standardized=True):
    with tf.name_scope('input'):

        # tf.train.string_input_producer
        # 将文件名列表交给tf.train.string_input_producer 函数.
        # 生成一个先入先出的队列，文件阅读器会需要它来读取数据。
        # Returns: A QueueRunner for the Queue is added to the current Graph's
        # QUEUE_RUNNER collection
        filename_queue = tf.train.string_input_producer([TFRecord_file], num_epochs=10)

        # init TFRecordReader class
        reader = tf.TFRecordReader()
        key, values = reader.read(filename_queue)  # filename_queue

        # parse_single_example将Example协议内存块(protocol buffer)解析为张量
        features = tf.parse_single_example(values,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })
        image = tf.decode_raw(features['img_raw'], tf.uint8)  # tf.uint8

        image = tf.cast(image, tf.float32)  # tf.uint8 to tf.float32

        image = tf.reshape(image, [1736, 1736, 3])

        label = tf.cast(features['label'], tf.int32)  # label tf.int32

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=128,
                                                               target_width=128)

        image_batch, label_batch = tf.train.batch([resized_image, label],
                                                  batch_size=batch_size,
                                                  num_threads=4,
                                                  capacity=100)

        label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


X, Y = readTFrecord("./trainData_v2.tfrecords", batch_size)

config = tf.ConfigProto()
config.gpu_options.allocator_type = "BFC"
config.gpu_options.allow_growth = True

# Start training
with tf.Session(config=config) as sess:

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    # Start the data queue
    coord = tf.train.Coordinator()

    # tf.train.start_queue_runners()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for step in range(10):
        for i in range(180):
            img, lab = sess.run([X, Y])
            print(img.shape)
    coord.request_stop()
    coord.join(threads)

# 验证图片是否正确
def validPic():
    filename = os.listdir(pathNew)
    # print(filename.__len__())
    # for i in filename:
    #     with open('./pic.txt', 'a') as fr:
    #         filePath = os.path.join(pathNew, i)
    #         fr.write(filePath)
    #         fr.write('\n')
    #     img = Image.open(filePath)
    #     print(img.size)

    df = pd.read_csv("/home/bruce/bigVolumn/Datasets/aptos/train.csv")
    df.id_code = pathNew + df.id_code.apply(str) + '.png'
    imagepaths = df.id_code.tolist()
    labels = df['diagnosis'].tolist()
    print(imagepaths)
    print(labels)
    file1 = [os.path.join(pathNew, i) for i in filename]
    file1.sort()
    imagepaths.sort()
    for index, file in zip(file1, imagepaths):
        if index != file:
            print(index, file)
# validPic()




