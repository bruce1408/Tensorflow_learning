# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 08:52:30 2018

@author: zy
"""

'''
导入flowers数据集
'''

# from datasets import download_and_convert_flowers
# from preprocessing import vgg_preprocessing
# from datasets import flowers
import tensorflow as tf

slim = tf.contrib.slim


def read_flower_image_and_label(dataset_dir, is_training=False):
    """
    下载flower_photos.tgz数据集
    切分训练集和验证集
    并将数据转换成TFRecord格式  5个训练数据文件(3320)，5个验证数据文件(350)，还有一个标签文件(存放每个数字标签对应的类名)

    args:
        dataset_dir:数据集所在的目录
        is_training：设置为TRue，表示加载训练数据集，否则加载验证集
    return:
        image,label:返回随机读取的一张图片，和对应的标签
    """
    download_and_convert_flowers.run(dataset_dir)
    '''
    利用slim读取TFRecord中的数据
    '''
    # 选择数据集train
    if is_training:
        dataset = flowers.get_split(split_name='train', dataset_dir=dataset_dir)
    else:
        dataset = flowers.get_split(split_name='validation', dataset_dir=dataset_dir)

    # 创建一个数据provider
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    # 通过provider的get随机获取一条样本数据 返回的是两个张量
    [image, label] = provider.get(['image', 'label'])

    return image, label


def get_batch_images_and_label(dataset_dir, batch_size, num_classes, is_training=False, output_height=224,
                               output_width=224, num_threads=10):
    """
    每次取出batch_size个样本

    注意：这里预处理调用的是slim库图片预处理的函数，例如：如果你使用的vgg网络，就调用vgg网络的图像预处理函数
          如果你使用的是自己定义的网络，则可以自己写适合自己图像的预处理函数，比如归一化处理也可以使用其他网络已经写好的预处理函数

    args:
         dataset_dir:数据集所在的目录
         batch_size:一次取出的样本数量
         num_classes：输出的类别 用于对标签one_hot编码
         is_training：设置为TRue，表示加载训练数据集，否则加载验证集
         output_height：输出图片高度
         output_width：输出图片宽

     return:
        images,labels:返回随机读取的batch_size张图片，和对应的标签one_hot编码
    """
    # 获取单张图像和标签
    image, label = read_flower_image_and_label(dataset_dir, is_training)
    # 图像预处理 这里要求图片数据是tf.float32类型的
    image = vgg_preprocessing.preprocess_image(image, output_height, output_width, is_training=is_training)

    # 缩放处理
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.image.resize_image_with_crop_or_pad(image, output_height, output_width)

    #  shuffle_batch 函数会将数据顺序打乱
    #  bacth 函数不会将数据顺序打乱
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        capacity=5 * batch_size,
        num_threads=num_threads)

    # one-hot编码
    labels = slim.one_hot_encoding(labels, num_classes)

    return images, labels
