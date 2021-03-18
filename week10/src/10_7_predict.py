# example
import tensorflow as tf
from alexnet_inference import conv_net
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
BATCH_SIZE = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
MODEL_SAVE_PATH = "model_svhn9/"
imgPath = "../datasets/svhn/mchar_test_a"


x = tf.placeholder(tf.float32, [None, 224, 224])

# def read_and_decode(tfrecords_file, batch_size):
#     """
#     val tfrecord is right
#     :param tfrecords_file:
#     :param batch_size:
#     :return:
#     """
#     filename_queue = tf.train.string_input_producer([tfrecords_file])
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     img_features = tf.parse_single_example(serialized_example,
#                                            features={
#                                                'label0': tf.FixedLenFeature([], tf.int64),
#                                                'label1': tf.FixedLenFeature([], tf.int64),
#                                                'label2': tf.FixedLenFeature([], tf.int64),
#                                                'label3': tf.FixedLenFeature([], tf.int64),
#                                                'image': tf.FixedLenFeature([], tf.string),
#                                            })
#     image = tf.decode_raw(img_features['image'], tf.uint8)
#     image = tf.reshape(image, [224, 224])
#     image = tf.cast(image, tf.float32) * (1. / 255)  # 在流中抛出img张量
#     # length = tf.cast(img_features['length'], tf.int32)
#     label0 = tf.cast(img_features['label0'], tf.int32)
#     label1 = tf.cast(img_features['label1'], tf.int32)
#     label2 = tf.cast(img_features['label2'], tf.int32)
#     label3 = tf.cast(img_features['label3'], tf.int32)
#
#     image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = \
#         tf.train.batch([image, label0, label1, label2, label3],
#                                               batch_size=batch_size,
#                                               num_threads=2,
#                                               capacity=32)
#     # length_batch = tf.reshape(length_batch, [batch_size])
#     label_batch0 = tf.reshape(label_batch0, [batch_size])
#     label_batch1 = tf.reshape(label_batch1, [batch_size])
#     label_batch2 = tf.reshape(label_batch2, [batch_size])
#     label_batch3 = tf.reshape(label_batch3, [batch_size])
#
#     print("Read tfrecord doc done!")
#     return image_batch, label_batch0, label_batch1, label_batch2, label_batch3


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    # 获取图片数据
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    # 没有经过预处理的灰度图
    image_raw = tf.reshape(image, [224, 224])
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, image_raw, label0, label1, label2, label3


image, image_raw, label0, label1, label2, label3 = read_and_decode("valData_4_digit_nolen.tfrecord")

# 使用shuffle_batch可以随机打乱
image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
    [image, image_raw, label0, label1, label2, label3], batch_size=BATCH_SIZE,
    capacity=50000, min_after_dequeue=10000, num_threads=1)


def plot_images(images, label0, label1, label2, label3, label4):
    for i in np.arange(0, 16):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        title = str(label0[i]) + str(label1[i]) + str(label2[i]) + str(label3[i]) + str(label4[i])
        plt.title(title, fontsize=10)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.imshow(images[i])
    plt.savefig('./name.jpg')
    plt.show()


with tf.Session() as sess:
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值
    logits0, logits1, logits2, logits3 = conv_net(X, 11, 0.2, reuse=False, is_training=False)

    # 预测值
    predict0 = tf.reshape(logits0, [-1, 11])
    correct_prediction0 = tf.argmax(predict0, 1)

    predict1 = tf.reshape(logits1, [-1, 11])
    correct_prediction1 = tf.argmax(predict1, 1)

    predict2 = tf.reshape(logits2, [-1, 11])
    correct_prediction2 = tf.argmax(predict2, 1)

    predict3 = tf.reshape(logits3, [-1, 11])
    correct_prediction3 = tf.argmax(predict3, 1)

    # correct_prediction0 = tf.argmax(logits0, 1)
    # correct_prediction1 = tf.argmax(logits1, 1)
    # correct_prediction2 = tf.argmax(logits2, 1)
    # correct_prediction3 = tf.argmax(logits3, 1)

    # 初始化
    sess.run(tf.global_variables_initializer())
    # 载入训练好的模型
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

    saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.restore(sess, './model_svhn7/model10000.ckpt.data-00000-of-00001')

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(20):
        # 获取一个批次的数据和标签
        b_image, b_image_raw, b_label0, b_label1, b_label2, b_label3 = sess.run([image_batch,
                                                                                 image_raw_batch,
                                                                                 label_batch0,
                                                                                 label_batch1,
                                                                                 label_batch2,
                                                                                 label_batch3])
        # 显示图片
        img = Image.fromarray(b_image_raw[0], 'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # 打印标签
        print('label:', b_label0, b_label1, b_label2, b_label3)
        # 预测
        label0, label1, label2, label3 = sess.run([correct_prediction0,
                                                   correct_prediction1,
                                                   correct_prediction2,
                                                   correct_prediction3], feed_dict={x: b_image})
        # 打印预测值
        print('predict:', label0, label1, label2, label3)

        # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
