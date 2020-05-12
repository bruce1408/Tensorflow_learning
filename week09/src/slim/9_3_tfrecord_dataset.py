import os
import math
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True, threshold=np.inf)

"""
tensorflow 有三种数据的输入方式,分别是:
1, placehold feed_dict：从内存中读取数据，占位符填充数据
2, queue队列：从硬盘读取数据
3, Dataset：同时支持内存和硬盘读取数据

官方推荐用tf.data.Dateset!!!
Tensorflow中之前主要用的数据读取方式主要有：
1、建立placeholder，然后使用feed_dict将数据feed进placeholder进行使用。
使用这种方法十分灵活，可以一下子将所有数据读入内存，然后分batch进行feed；也可以建立一个Python的generator，
一个batch一个batch的将数据读入，并将其feed进placeholder。这种方法很直观，用起来也比较方便灵活，但是这种方法的效率较低，
难以满足高速计算的需求。

2、使用TensorFlow的QueueRunner，通过一系列的Tensor操作，将磁盘上的数据分批次读入并送入模型进行使用。这种方法效率很高，
但因为其牵涉到Tensor操作，不够直观，也不方便调试，所有有时候会显得比较困难。使用这种方法时，
常用的一些操作包括tf.TextLineReader，tf.FixedLengthRecordReader以及tf.decode_raw等等。
如果需要循环，条件操作，还需要使用TensorFlow的tf.while_loop，tf.case等操作。

3、上面的方法我觉得已经要被tensorflow放弃了，现在官方推荐用tf.data.Dataset模块，使其数据读入的操作变得更为方便，
而支持多线程（进程）的操作，也在效率上获得了一定程度的提高
tfrecord支持写入三种格式的数据：string，int64，float32，以列表的形式分别通过 
tf.train.BytesList、tf.train.Int64List、tf.train.FloatList写入tf.train.Feature,并且以dict的形式
把数据汇总,最后构建tf.train.Features：本文主要通过tensorflow的最新形式来读取数据.
"""
np.random.seed(0)
pathDir = "/home/bruce/dataSets/101_ObjectCategories"
classes = os.walk(pathDir).__next__()[1]
imgPath = list()
labels = list()
label = 0
for folder in classes:
    folderPath = os.path.join(pathDir, folder)
    for img in os.listdir(folderPath):
        if img.endswith(".jpeg") or img.endswith('.JPEG') or img.endswith(".jpg"):
            imgPath.append(os.path.join(folderPath, img))
            labels.append(label)
    label += 1
# data = np.array([imgPath, labels]).transpose()

img_train, img_test, label_train, label_test = train_test_split(imgPath, labels, test_size=0.25)
img_train, img_val, label_train, label_val = train_test_split(img_train, label_train, test_size=0.25)
print(img_train.__len__())
print(img_test.__len__())
print(img_val.__len__())


def imgToTfrecord(filepath, labelpath, tfrecordName):

    writer = tf.python_io.TFRecordWriter("mydata.tfrecords")
    for path, label in zip(img_train, label_train):
        img = Image.open(path)
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            # value=[index]决定了图片数据的类型label
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


# writer= tf.python_io.TFRecordWriter("mydata.tfrecords")
# for path, label in zip(img_train, label_train):
#     img=Image.open(img_path)
#     img_raw=img.tobytes()#将图片转化为二进制格式
#     example = tf.train.Example(features=tf.train.Features(feature={
#         #value=[index]决定了图片数据的类型label
#         "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#     })) #example对象对label和image数据进行封装
#     writer.write(example.SerializeToString())  #序列化为字符串
# writer.close()

def _int64_feature(value):
    return tf.train.Feature(int64List=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(floatList=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytesList=tf.train.BytesList(value=[value]))
