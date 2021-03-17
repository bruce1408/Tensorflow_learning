"""
实现inception v1 网络结构
"""
import os
import tensorflow as tf
from utils.logWriter import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

"""
train the dataset from scratch
https://www.zhihu.com/search?type=content&q=Momentum 
https://blog.csdn.net/yzy_1996/article/details/84618536
lr = lr * decay_rate ^ (global_step / decay_step)
"""
# Image Parameters
N_CLASSES = 2  # CHANGE HERE, total number of classes
IMG_HEIGHT = 224  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 224  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale
n_classes = N_CLASSES  # MNIST total classes (0-9 digits)
dropout = 0.45
num_steps = 10000
train_display = 10
val_display = 1000
BATCHSIZE = 32
save_check = 3000
learning_rate = 0.001
decay_rate = 0.96
decay_step = 500
log_path = './inception_train'


def _parse_function(record):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['img_raw'], tf.uint8)
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])
    image = tf.cast(image, tf.float32)
    image = image / 225.0
    image = image - 0.5
    image = image * 2.0
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


# train data pipline
# repeat -> shuffle 和 shuffle -> repeat不一样
traindata = tf.data.TFRecordDataset("/raid/bruce/dog_cat/train_dog_cat_224.tfrecord"). \
    map(_parse_function). \
    shuffle(buffer_size=2000, reshuffle_each_iteration=True). \
    batch(BATCHSIZE). \
    repeat(). \
    prefetch(BATCHSIZE)

# val data pipline
valdata = tf.data.TFRecordDataset("/raid/bruce/dog_cat/test_dog_cat_224.tfrecord"). \
    map(_parse_function). \
    batch(BATCHSIZE). \
    repeat(). \
    prefetch(BATCHSIZE)
# Create an iterator over the dataset

is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(tf.constant(0), name='global_step', trainable=False)

iterator = tf.data.Iterator.from_structure(traindata.output_types, traindata.output_shapes)
X, Y = iterator.get_next()

traindata_init = iterator.make_initializer(traindata)
valdata_init = iterator.make_initializer(valdata)


def check_accuracy(sess, correct_prediction, dataset_init_op, batches_to_check):
    # Initialize the validation dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    for i in range(batches_to_check):
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def inception_block(X_input, filters, stage, block):
    """
    Implementation of the inception block
    :param x: input img = [224, 224, 3]
    :param kernel_size:
    :param filters:
    :param stage:
    :param block:
    :return:
    """
    conv_name_base = 'inceptioin_' + str(stage) + "_" + block
    relu_name_base = 'relu_' + str(stage)
    f1, f2, f3, f4, f5, f6 = filters

    with tf.name_scope('inception_block' + str(stage)):

        conv1 = tf.layers.conv2d(X_input, f1, kernel_size=1, strides=1, padding='same',
                                 name=conv_name_base + 'conv1', activation=tf.nn.relu)
        conv3_1 = tf.layers.conv2d(X_input, f2, kernel_size=1, strides=1, padding='same',
                                   name=conv_name_base + 'conv3_1',
                                   activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(conv3_1, f3, kernel_size=3, strides=1, padding='same',
                                   name=conv_name_base + 'conv3_2',
                                   activation=tf.nn.relu)

        conv5_1 = tf.layers.conv2d(X_input, f4, kernel_size=1, strides=1, padding='same',
                                   name=conv_name_base + 'conv5_1',
                                   activation=tf.nn.relu)
        conv5_2 = tf.layers.conv2d(conv5_1, f5, kernel_size=5, strides=1, padding='same',
                                   name=conv_name_base + 'conv5_2',
                                   activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(X_input, pool_size=3, strides=1, padding='same')
        pool2 = tf.layers.conv2d(pool1, f6, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)

        # print(conv1.shape, conv3_2.shape, conv5_2.shape, pool2.shape)
        out = tf.concat([conv1, conv3_2, conv5_2, pool2], axis=3)
        print('outshape', out.shape)

    return out


def GoogleNet(X, n_classes):
    conv1 = tf.layers.conv2d(X, filters=64, kernel_size=7, strides=2, padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=2, padding='same')

    conv2 = tf.layers.conv2d(pool1, filters=192, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv2, filters=192, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=2, padding='same')
    print('layer1: ', pool2.shape)
    # inception 3
    incep3a = inception_block(pool2, [64, 96, 128, 16, 32, 32], '3a', block='a')
    incep3a = inception_block(incep3a, [128, 128, 192, 32, 96, 64], '3a', block='b')
    pool3 = tf.layers.max_pooling2d(incep3a, pool_size=3, strides=2, padding='same')
    print('pool3: ', pool3.shape)

    # inception 4
    incep4a = inception_block(pool3, [192, 96, 208, 16, 48, 64], '4a', block='a')
    incep4b = inception_block(incep4a, [160, 112, 224, 24, 64, 64], '4a', block='b')
    incep4c = inception_block(incep4b, [128, 128, 256, 24, 64, 64], '4a', block='c')
    incep4d = inception_block(incep4c, [112, 144, 288, 32, 64, 64], '4a', block='d')
    incep4e = inception_block(incep4d, [256, 160, 320, 32, 128, 128], '4a', block='e')
    pool4 = tf.layers.max_pooling2d(incep4e, pool_size=3, strides=2, padding='same')
    print('pool4 ', pool4.shape)
    # inceptioin 5a
    incep5a = inception_block(pool4, [256, 160, 320, 32, 128, 128], '5a', block='a')
    incep5a = inception_block(incep5a, [384, 192, 384, 48, 128, 128], '5a', block='b')
    pool5 = tf.layers.average_pooling2d(incep5a, pool_size=7, strides=1, padding='valid')
    print('pool5 ', pool5.shape)
    droplayer = tf.layers.dropout(pool5, rate=dropout, training=is_training)
    fc1 = tf.contrib.layers.flatten(droplayer)
    out = tf.layers.dense(fc1, n_classes)
    out = tf.nn.softmax(out)

    # 辅助分类器1
    ass1 = tf.layers.average_pooling2d(incep4a, 5, strides=3, padding='VALID')
    ass1 = tf.layers.conv2d(ass1, filters=128, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)
    ass1 = tf.contrib.layers.flatten(ass1)
    ass1 = tf.layers.dense(ass1, 1024)
    ass1 = tf.nn.relu(ass1)
    ass1 = tf.layers.dropout(ass1, 0.2, training=is_training)
    ass1 = tf.layers.dense(ass1, n_classes)
    out_1 = tf.nn.softmax(ass1)

    # 辅助分类器2
    ass2 = tf.layers.average_pooling2d(incep4d, 5, strides=3, padding='VALID')
    ass2 = tf.layers.conv2d(ass2, filters=128, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)
    ass2 = tf.contrib.layers.flatten(ass2)
    ass2 = tf.layers.dense(ass2, 1024)
    ass2 = tf.nn.relu(ass2)
    ass2 = tf.layers.dropout(ass2, 0.2, training=is_training)
    ass2 = tf.layers.dense(ass2, n_classes)
    out_2 = tf.nn.softmax(ass2)

    return out, out_1, out_2


out1, out2, out3 = GoogleNet(X, n_classes=2)

cost_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out1, labels=Y))
cost_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out2, labels=Y))
cost_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out3, labels=Y))
loss_op = cost_real + 0.3 * cost_1 + 0.3 * cost_2

learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_rate, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95).minimize(loss_op)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
correct_pred = tf.equal(tf.argmax(out1, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
# 添加日志文件
if not os.path.exists(log_path):
    print("====== The log folder was not found and is being generated !======")
    os.makedirs(log_path)
else:
    print('======= The log path folder already exists ! ======')

log = Logger('./inception_train/inception_train.log', level='info')
# Start training
# Initialize the iterator
with tf.Session() as sess:
    # sess.run(iterator.initializer)
    sess.run(init)
    sess.run(traindata_init)
    saver = tf.train.Saver(max_to_keep=3)
    ckpt = tf.train.get_checkpoint_state('./model_GoogleNet')
    if ckpt is None:
        print("Model not found, please train your model first...")
    else:
        path = ckpt.model_checkpoint_path
        print('loading pre-trained model from %s.....' % path)
        saver.restore(sess, path)
    # Training cycle
    for step in range(1, num_steps + 1):
        loss, acc, _ = sess.run([loss_op, accuracy, optimizer], {is_training: True})
        if step % train_display == 0 or step == 1:
            lr = sess.run(learning_rate, {global_step: step})

            log.logger.info("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss)
                            + ", train acc = " + "{:.2f}".format(acc) + ", lr = " + "{:.6f}".format(lr))

        if step % val_display == 0 and step is not 0:
            sess.run(valdata_init)
            avg_acc = 0
            acc = check_accuracy(sess, correct_pred, valdata_init, val_display)
            loss = sess.run(loss_op, {is_training: False})
            print("\033[1;36m=\033[0m" * 60)
            log.logger.info("Step %d, Minibatch Loss= %.4f, Test Accuracy= %.4f\033[0m" % (step, loss, acc))
            # print("\033[1;36mStep %d, Minibatch Loss= %.4f, Test Accuracy= %.4f\033[0m" % (step, loss, acc))
            print("\033[1;36m=\033[0m" * 60)

        if step % save_check == 0:
            path_name = "./model_GoogleNet/model" + str(step) + ".ckpt"
            saver.save(sess, path_name)
            print("model has been saved in %s" % path_name)

    print("Optimization Finished!")
