from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import os
import os.path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
TRAIN_CHECK_POINT = 'check_point/train_model.ckpt'
VGG_19_MODEL_DIR = '../datasets/dog_cat/vgg_19.ckpt'
BATCHSIZE = 32
EPOCH = 30
IMG_HEIGHT = 224
IMG_WIDTH = 224


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


is_training = tf.placeholder(tf.bool)

traindata = tf.data.TFRecordDataset(
    "../week02/src/train_dog_cat_224.tfrecord"). \
    map(_parse_function).\
    repeat().batch(BATCHSIZE).\
    prefetch(BATCHSIZE)

valdata = tf.data.TFRecordDataset(
    "../week02/src/test_dog_cat_224.tfrecord"). \
    map(_parse_function).\
    repeat().batch(BATCHSIZE).\
    prefetch(BATCHSIZE)
# Create an iterator over the dataset

iterator = tf.data.Iterator.from_structure(traindata.output_types, traindata.output_shapes)
X, Y = iterator.get_next()

traindata_init = iterator.make_initializer(traindata)
valdata_init = iterator.make_initializer(valdata)


def get_accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
with slim.arg_scope(nets.vgg.vgg_arg_scope()):
    logits, _ = nets.vgg.vgg_19(inputs=X, num_classes=2, dropout_keep_prob=keep_prob, is_training=is_training)
# logits = tf.squeeze(logits, [1, 2])
variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_19/fc8'])
restorer = tf.train.Saver(variables_to_restore)

with tf.name_scope('cross_entropy'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# tf.summary.scalar('cross_entropy', loss)
learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.name_scope('accuracy'):
    accuracy = get_accuracy(logits, Y)
# tf.summary.scalar('accuracy', accuracy)
#
# merged = tf.summary.merge_all()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    restorer.restore(sess, VGG_19_MODEL_DIR)
    step = 0
    best_acc = 0.0
    for ep in range(EPOCH):
        all_accuracy = 0
        all_loss = 0
        sess.run(traindata_init)
        for i in range(200):
            _, accuracy_out, loss_out = sess.run([optimizer, accuracy, loss],
                                                 feed_dict={keep_prob: 0.5, is_training: True})
            # train_writer.add_summary(summary, step)
            step += 1
            all_accuracy += accuracy_out
            all_loss += loss_out
            if i % 10 == 0:
                print("Epoch %d: Batch %d accuracy is %.2f; Batch loss is %.5f" % (ep + 1, i, accuracy_out, loss_out))
        print("Epoch %d: Train accuracy is %.2f; Train loss is %.5f" % (ep + 1, all_accuracy / 200, all_loss / 200))

        all_accuracy = 0
        all_loss = 0
        for i in range(600):
            sess.run(valdata_init)
            accuracy_out, loss_out = sess.run([accuracy, loss], feed_dict={keep_prob: 1.0, is_training: False})
            all_accuracy += accuracy_out
            all_loss += loss_out

        print("Epoch %d: Validation accuracy is %.2f; Validation loss is %.5f" %
              (ep + 1, all_accuracy / 600, all_loss / 600))
        tempAcc = all_accuracy / 600.0
        if best_acc < tempAcc:
            best_acc = tempAcc
            saver.save(sess, TRAIN_CHECK_POINT)
