from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import pandas as pd
from PIL import Image
import numpy as np
from natsort import natsorted
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import csv as csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
imgPath = "/raid/bruce/dog_cat/test1"
CHECK_POINT_PATH = 'check_point/train_model.ckpt'
NUM_KAGGLE_TEST = 12500


with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    keep_prob = tf.placeholder(tf.float32)
    logits, _ = nets.vgg.vgg_19(inputs=images, num_classes=2, dropout_keep_prob=keep_prob, is_training=False)
    probabilities = tf.nn.softmax(logits)
    correct_prediction = tf.argmax(logits, 1)
    variables_to_restore = slim.get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)

    data = dict()
    cnt = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        labelList = list()
        imageList = list()
        # 通过文件名得到模型保存时迭代的轮数.格式：model.ckpt-6000.data-00000-of-00001

        num = len(os.listdir(imgPath))
        for i in natsorted(os.listdir(imgPath)):
            cnt += 1
            imgpath = os.path.join(imgPath, i)
            img = Image.open(imgpath)

            img = img.resize((224, 224))
            image_ = np.array(img)
            image_ = image_.reshape([1, 224, 224, 3])

            # 获取预测结果
            probabilities_, label = sess.run([probabilities, correct_prediction], feed_dict={images: image_})

            # 获取此标签的概率
            probability = probabilities_[0][label[0]]
            labelList.append(label[0])
            imageList.append(img)
            # probability = probability.clip(min=0.005, max=0.995)
            # data[i.split('.')[0]] = probability[0]
            data[i.split('.')[0]] = label[0].clip(min=0.05, max=0.995)
            # print(data)

            sys.stdout.write('\r>> Creating image %d/%d' % (cnt + 1, num))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

        # print("After %s training step(s),validation label = %d, has %g probability, the img path is %s" %
        #       (global_step, label, probability, imgpath))
        sorted(data.keys())
        print('the result is:', data)
        result = pd.DataFrame.from_dict(data, orient='index', columns=['label'])
        result = result.reset_index().rename(columns={'index': 'id'})
        result.to_csv('/raid/bruce/dog_cat/result.csv', index=False)
        print("predict is done!")

