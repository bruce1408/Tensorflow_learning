import requests
import os, sys
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['CUDA_ VISIBLE_DEVICES'] = '0'
trainImg = "/raid/bruce/datasets/svhn/mchar_train"
trainLabel = "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json"
valImg = "/raid/bruce/datasets/svhn/mchar_val"
valLabel = "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json"
HIGHTSIZE = 224
WIDTHSIZE = 224
CHANNELS = 1
BATCH_SIZE = 20

def readImgPath(dataimg, datalabel):
    labels = requests.get(datalabel)
    json_response = labels.content.decode()
    dict_json = json.loads(json_response)
    imgLabelData = dict()
    for key, value in dict_json.items():
        key = os.path.join(dataimg, key)
        if value['label'].__len__() == 6:
            continue
        while len(value['label']) < 5:
            value['label'].append(10)
        imgLabelData[key] = value['label']
    return imgLabelData


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
            sys.stdout.write('\r>> Creating image %d/%d' % (index + 1, num))
            sys.stdout.flush()
            image = Image.open(imgpath)
            image = image.resize((HIGHTSIZE, WIDTHSIZE))
            image = np.array(image.convert('L'))  # 转成灰度图即可 尺寸变成 224*224*1
            if image.shape == (HIGHTSIZE, WIDTHSIZE, CHANNELS):
                errorCount += 1
                image_raw = image.tostring()
                label = value
                example = tf.train.Example(features=tf.train.Features(
                    feature={"img_raw": _bytes_feature(image_raw),
                             "label0": _int64_feature(label[0]),
                             'label1': _int64_feature(label[1]),
                             'label2': _int64_feature(label[2]),
                             'label3': _int64_feature(label[3]),
                             'label4': _int64_feature(label[4]),
                             }
                ))
                writer.write(example.SerializeToString())
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
# imgLabelData = readImgPath(valImg, valLabel)
# convert_to_tfrecord(imgLabelData, "valData.tfrecord")

# 验证数据集是否正确

def read_and_decode(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label0': tf.FixedLenFeature([], tf.int64),
                                               'label1': tf.FixedLenFeature([], tf.int64),
                                               'label2': tf.FixedLenFeature([], tf.int64),
                                               'label3': tf.FixedLenFeature([], tf.int64),
                                               'label4': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })
    image = tf.decode_raw(img_features['img_raw'], tf.uint8)
    image = tf.reshape(image, [HIGHTSIZE, WIDTHSIZE, CHANNELS])
    image = tf.cast(image, tf.float32) * (1. / 255)  # 在流中抛出img张量
    label0 = tf.cast(img_features['label0'], tf.int32)
    label1 = tf.cast(img_features['label1'], tf.int32)
    label2 = tf.cast(img_features['label2'], tf.int32)
    label3 = tf.cast(img_features['label3'], tf.int32)
    label4 = tf.cast(img_features['label4'], tf.int32)

    image_batch, label_batch0, label_batch1, label_batch2, label_batch3, label_batch4 = \
        tf.train.batch([image, label0, label1, label2, label3, label4],
                                              batch_size=batch_size,
                                              num_threads=2,
                                              capacity=32)
    label_batch0 = tf.reshape(label_batch0, [batch_size])
    label_batch1 = tf.reshape(label_batch1, [batch_size])
    label_batch2 = tf.reshape(label_batch2, [batch_size])
    label_batch3 = tf.reshape(label_batch3, [batch_size])
    label_batch4 = tf.reshape(label_batch4, [batch_size])

    print("Read tfrecord doc done!")
    return image_batch, label_batch0, label_batch1, label_batch2, label_batch3, label_batch4


def plot_images(images, label0, label1, label2, label3, label4):
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(4, 5, i + 1)
        plt.axis('off')
        title = str(label0) + str(label1) + str(label2) + str(label3) + str(label4)
        plt.title(title, fontsize=14)
        plt.subplots_adjust(wspace=0.5, hspace=3)

        plt.imshow(images[i])
    plt.show()


# 验证图片生成的tfrecord文件是否正确
image_batch, label_batch0, label_batch1, label_batch2, label_batch3, label_batch4 = \
    read_and_decode('./train.tfrecord', batch_size=BATCH_SIZE)
x = image_batch
in_channels = x.get_shape()[-1]

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            # just plot one batch size
            image, label0, label1, label2, label3, label4 = \
                sess.run([image_batch, label_batch0, label_batch1, label_batch2, label_batch3, label_batch4])
            # print(label.shape)
            plot_images(image, label0, label1, label2, label3, label4)
            i = i + 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)