import os
import tensorflow as tf
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Dataset Parameters - CHANGE HERE
# DATASET_PATH = '101_ObjectCategories'  # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 2  # CHANGE HERE, total number of classes
IMG_HEIGHT = 128  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 128  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale
n_classes = N_CLASSES  # MNIST total classes (0-9 digits)
dropout = 0.75
num_steps = 20000
display_step = 100
learning_rate = 0.01
BATCHSIZE=32


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path):
    """
    imagepaths 保存的是所有的图片的路径
    :param dataset_path:
    :param mode:
    :param batch_size:
    :return:
    """
    path = os.getcwd()
    dirPath = os.path.join(path, dataset_path)
    print(dirPath)
    imagePaths = list()
    labels = list()
    label = 0
    for parent, _, filenames in os.walk(dirPath):
        for img in filenames:
            if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith(".JPEG"):
                imagePaths.append(os.path.join(parent, img))
                labels.append(label)
        label += 1
    for i in range(len(labels)):
        labels[i] = labels[i]-1
    return imagePaths, labels


def _parse_function1(imagepaths, labels):
    """
    数据预处理环节
    :param imagepaths:
    :param labels:
    :return:
    """
    image_string = tf.read_file(imagepaths)
    image_decode = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    image_resized = tf.image.resize_images(image_decode, [IMG_HEIGHT, IMG_WIDTH])
    return image_resized, labels


def _parse_function(record):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['img_raw'], tf.uint8)
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


dataset = tf.data.TFRecordDataset(
    "../datasets/train_dogs_cat.tfrecord")
dataset = dataset.map(_parse_function)
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=BATCHSIZE)
dataset = dataset.prefetch(BATCHSIZE)

# Create an iterator over the dataset
iterator = dataset.make_one_shot_iterator()
with tf.Session() as sess:
    while True:
        try:
            image, label = sess.run(iterator.get_next())
            # print(sess.run(iterator.get_next()))
            print(image.shape)
            print(label.shape)
        except tf.errors.OutOfRangeError:
            break
