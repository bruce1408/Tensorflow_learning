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
BATCH_SIZE = 20
HIGHTSIZE = 128
WIDTHSIZE = 128
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


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def imageToTfrecord(filepath, labelpath, tfrecordName):
    writer = tf.python_io.TFRecordWriter(tfrecordName)
    for path, label in zip(filepath, labelpath):
        image = Image.open(path)
        # print(image.size)
        image = image.resize((HIGHTSIZE, WIDTHSIZE), Image.ANTIALIAS)
        # print(image.size)
        image = np.array(image)
        print(image.shape)
        img_raw = image.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            # value=[index]决定了图片数据的类型label
            "label": _int64_feature(label),
            "image": _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


# 生成tfrecord
# imageToTfrecord(img_train, label_train, './train.tfrecord')
def convert_to_tfrecord(images, labels):
    errorCount = 0
    filename = './train_1.tfrecord'
    n_samples = len(labels)
    if np.shape(images)[0] != n_samples:
        raise ValueError('Image size %d does not match label size %d.' % (images.shape, n_samples.shape))
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start....')
    for i in np.arange(n_samples):
        try:
            image = Image.open(images[i])
            image = image.resize((HIGHTSIZE, WIDTHSIZE))
            image = np.array(image)
            if image.shape == (HIGHTSIZE, WIDTHSIZE, 3):
                errorCount += 1
                print(images[i])
                print(labels[i])
                # os.remove(images[i])  # 去除掉所有的非rgb图
                image_raw = image.tostring()
                label = int(labels[i])
                example = tf.train.Example(features=tf.train.Features(
                    feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                             "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
                ))
                writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error : %s' % e)
            print('Skip it! \n')
    writer.close()
    print("Transform done!")
    print(errorCount)

convert_to_tfrecord(img_train, label_train)
# 读取tfrecord文件
# def read_and_decode(filename):  # 读入tfrecords
#     filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw': tf.FixedLenFeature([], tf.string),
#                                        })  # 将image数据和label取出来
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [128, 128, 3])  # reshape为128*128的3通道图片
#     img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
#     label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
#     with tf.Session() as sess:  # 开始一个会话
#         init_op = tf.initialize_all_variables()
#         sess.run(init_op)
#         coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
#         threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队
#         for i in range(20):
#             example, l = sess.run([img, label])  # 在会话中取出image和label
#             img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
#             print(i, l)
#             img.save('./datasets/' + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
#         coord.request_stop()
#         coord.join(threads)
#     # return img, label
#
#
# read_and_decode('./train_1.tfrecord')
# print(img)
# print(label)


def read_and_decode(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })
    image = tf.decode_raw(img_features['img_raw'], tf.uint8)

    image = tf.reshape(image, [HIGHTSIZE, WIDTHSIZE, 3])

    image = tf.cast(image, tf.float32) * (1. / 255)  # 在流中抛出img张量
    label = tf.cast(img_features['label'], tf.int32)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=2,
                                              capacity=32)
    label_batch = tf.reshape(label_batch, [batch_size])
    print("Read tfrecord doc done!")
    return image_batch, label_batch


def plot_images(images, labels):
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(4, 5, i + 1)
        plt.axis('off')
        plt.title(labels[i], fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()


image_batch, label_batch = read_and_decode('./train_1.tfrecord', batch_size=BATCH_SIZE)

x = image_batch
in_channels = x.get_shape()[-1]

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            # just plot one batch size
            image, label = sess.run([image_batch, label_batch])
            # print(label.shape)
            plot_images(image, label)
            i = i + 1

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
# filename_queue = tf.train.string_input_producer(["./train.tfrecord"])  # 读入流中
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
# features = tf.parse_single_example(serialized_example,
#                                    features={
#                                        'label': tf.FixedLenFeature([], tf.int64),
#                                        'image': tf.FixedLenFeature([], tf.string),
#                                    })  # 取出包含image和label的feature对象
# image = tf.decode_raw(features['image'], tf.uint8)
# image = tf.reshape(image, [64, 64, 3])
# label = tf.cast(features['label'], tf.int32)
# with tf.Session() as sess:  # 开始一个会话
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
#     threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队
#     for i in range(20):
#         example, l = sess.run([image, label])  # 在会话中取出image和label
#         img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
#         img.save('./datasets/'+str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
#         print(example, l)
#     coord.request_stop()
#     coord.join(threads)

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
