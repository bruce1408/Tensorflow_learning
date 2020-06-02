import requests
import os, sys
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
"""
train data info :
5th is: 8
4th is: 1280
3th is: 7813
2th is: 16262

val data info:
5th is: 2
4th is: 118
3th is: 1569
2th is: 6393



"""
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
trainImg = "/raid/bruce/datasets/svhn/mchar_train"
trainLabel = "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json"
valImg = "/raid/bruce/datasets/svhn/mchar_val"
valLabel = "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json"
HIGHTSIZE = 224
WIDTHSIZE = 224
CHANNELS = 3
BATCH_SIZE = 16


def readImgPath(dataimg, datalabel):

    labels = requests.get(datalabel)
    json_response = labels.content.decode()
    dict_json = json.loads(json_response)
    imgLabelData = dict()
    for key, value in dict_json.items():
        length = len(value['label'])
        # print(length)
        key = os.path.join(dataimg, key)
        if value['label'].__len__() >= 5:  # 5还有5以上的标签不要
            continue
        while len(value['label']) < 5:
            value['label'].append(10)
        value['label'].insert(0, length-1)
        imgLabelData[key] = value['label']
    return imgLabelData


# imgLabelData = readImgPath(valImg, valLabel)

# print(imgLabelData)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(imageLabelData, filename):
    errorCount = 0
    num = imageLabelData.__len__()
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start....')
    for index, (imgpath, value) in enumerate(imageLabelData.items()):
        try:
            image = Image.open(imgpath)
            image = image.resize((HIGHTSIZE, WIDTHSIZE))
            image = np.array(image)
            # image = np.array(image.convert('L'))  # 转成灰度图即可 尺寸变成 224*224*1
            if image.shape == (HIGHTSIZE, WIDTHSIZE, 3):
                errorCount += 1
                image_raw = image.tostring()
                label = value
                example = tf.train.Example(features=tf.train.Features(
                    feature={"img_raw": _bytes_feature(image_raw),
                             'length': _int64_feature(label[0]),
                             "label0": _int64_feature(label[1]),
                             'label1': _int64_feature(label[2]),
                             'label2': _int64_feature(label[3]),
                             'label3': _int64_feature(label[4]),
                             # 'label4': _int64_feature(label[5]),
                             }
                ))
                writer.write(example.SerializeToString())
            sys.stdout.write('\r>> Creating image %d/%d' % (index + 1, num))
            sys.stdout.flush()
        except IOError as e:
            print('Could not read:', imgpath)
            print('error : %s' % e)
            print('Skip it! \n')
    sys.stdout.write('\n')
    sys.stdout.flush()
    writer.close()
    print("Transform done!")
    print(errorCount)


# 生成tfrecord
imgLabelData = readImgPath(valImg, valLabel)
print(imgLabelData)
convert_to_tfrecord(imgLabelData, "valData_4_len_3channles_label0.tfrecord")


def read_and_decode(tfrecords_file, batch_size):
    """
    val tfrecord is right
    :param tfrecords_file:
    :param batch_size:
    :return:
    """
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example,
                                           features={
                                               'length': tf.FixedLenFeature([], tf.int64),
                                               'label0': tf.FixedLenFeature([], tf.int64),
                                               'label1': tf.FixedLenFeature([], tf.int64),
                                               'label2': tf.FixedLenFeature([], tf.int64),
                                               'label3': tf.FixedLenFeature([], tf.int64),
                                               # 'label4': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })
    image = tf.decode_raw(img_features['img_raw'], tf.uint8)
    image = tf.reshape(image, [HIGHTSIZE, WIDTHSIZE, 3])
    image = tf.cast(image, tf.float32) * (1. / 255)  # 在流中抛出img张量
    length = tf.cast(img_features['length'], tf.int32)
    label0 = tf.cast(img_features['label0'], tf.int32)
    label1 = tf.cast(img_features['label1'], tf.int32)
    label2 = tf.cast(img_features['label2'], tf.int32)
    label3 = tf.cast(img_features['label3'], tf.int32)
    # label4 = tf.cast(img_features['label4'], tf.int32)

    image_batch, length_batch, label_batch0, label_batch1, label_batch2, label_batch3 = \
        tf.train.batch([image, length, label0, label1, label2, label3],
                                              batch_size=batch_size,
                                              num_threads=2,
                                              capacity=32)
    length_batch = tf.reshape(length_batch, [batch_size])
    label_batch0 = tf.reshape(label_batch0, [batch_size])
    label_batch1 = tf.reshape(label_batch1, [batch_size])
    label_batch2 = tf.reshape(label_batch2, [batch_size])
    label_batch3 = tf.reshape(label_batch3, [batch_size])
    # label_batch4 = tf.reshape(label_batch4, [batch_size])

    print("Read tfrecord doc done!")
    return image_batch, length_batch, label_batch0, label_batch1, label_batch2, label_batch3


def plot_images(images, label0, label1, label2, label3):
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        title = str(label0[i]) + str(label1[i]) + str(label2[i]) + str(label3[i])
        plt.title(title, fontsize=10)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.imshow(images[i])
    plt.savefig('./name.jpg')
    plt.show()


# 验证图片生成的tfrecord文件是否正确
# def valTfrecord(tfrecordName):
image_batch, length_batch, label_batch0, label_batch1, label_batch2, label_batch3 = \
    read_and_decode("trainData_4_len_3channles.tfrecord", batch_size=BATCH_SIZE)
x = image_batch
in_channels = x.get_shape()[-1]

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            # just plot one batch size
            image, label0, label1, label2, label3 = sess.run([image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
            print(image.shape)
            plot_images(image, label0, label1, label2, label3)
            i = i + 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)


# valTfrecord('valData_4_len_3channles.tfrecord')