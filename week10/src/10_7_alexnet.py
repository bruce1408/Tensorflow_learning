import os
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Image Parameters
N_CLASSES = 10  # CHANGE HERE, total number of classes
IMG_HEIGHT = 224  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 224  # CHANGE HERE, the image width to be resized to
CHANNELS = 1  # The 3 color channels, change to 1 if grayscale
n_classes = N_CLASSES  # MNIST total classes (0-9 digits)
dropout = 0.1
num_steps = 10000
train_display = 100
val_display = 300
learning_rate = 0.0001
BATCHSIZE = 64


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
        'image': tf.FixedLenFeature((), tf.string),
        'label0': tf.FixedLenFeature((), tf.int64),
        'label1': tf.FixedLenFeature([], tf.int64),
        'label2': tf.FixedLenFeature([], tf.int64),
        'label3': tf.FixedLenFeature([], tf.int64)
        # 'length': tf.FixedLenFeature([], tf.int64)

    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, CHANNELS])

    image = tf.cast(image, tf.float32)
    label0 = tf.cast(parsed['label0'], tf.int32)
    label1 = tf.cast(parsed['label1'], tf.int32)
    label2 = tf.cast(parsed['label2'], tf.int32)
    label3 = tf.cast(parsed['label3'], tf.int32)
    # label4 = tf.cast(parsed['length'], tf.int32)

    return image, label0, label1, label2, label3


# 训练集
traindata = tf.data.TFRecordDataset("./captcha/train.tfrecords").\
    map(_parse_function).\
    repeat().shuffle(buffer_size=3000).batch(BATCHSIZE).prefetch(BATCHSIZE)

# 验证集
# valdata = tf.data.TFRecordDataset("./valData_4_digit_nolen.tfrecord").\
#     map(_parse_function).\
#     repeat().shuffle(buffer_size=3000).batch(BATCHSIZE).prefetch(BATCHSIZE)

# Create an iterator over the dataset
iterator = tf.data.Iterator.from_structure(traindata.output_types, traindata.output_shapes)
X, Y0, Y1, Y2, Y3 = iterator.get_next()

traindata_init = iterator.make_initializer(traindata)
# valdata_init = iterator.make_initializer(valdata)
print(X.shape)


def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # Convolution Layer with 32 filters and a kernel size of 3
        x = tf.reshape(x, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
        # Convolution Layer with 32 filters and a kernel size of 3
        conv1 = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=4, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool1 = tf.layers.max_pooling2d(conv1, 3, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2_1 = tf.layers.conv2d(pool1, filters=256, kernel_size=5, padding='SAME', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(conv2_1, 3, 2)

        conv3_1 = tf.layers.conv2d(pool2, 384, 3, padding='SAME', activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(conv3_1, 384, 3, padding='SAME', activation=tf.nn.relu)
        conv3_3 = tf.layers.conv2d(conv3_2, 256, 3, padding='SAME', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3_3, 3, 2)

        fc1 = tf.contrib.layers.flatten(pool3)

        fc2 = tf.layers.dense(fc1, 4096)
        fc3 = tf.layers.dense(fc2, 4096)

        # Output layer, class prediction
        digit1 = tf.layers.dense(fc3, n_classes)
        digit2 = tf.layers.dense(fc3, n_classes)
        digit3 = tf.layers.dense(fc3, n_classes)
        digit4 = tf.layers.dense(fc3, n_classes)
        # digit5 = tf.layers.dense(fc2, 6)

        digit1 = tf.nn.softmax(digit1) if not is_training else digit1
        digit2 = tf.nn.softmax(digit2) if not is_training else digit2
        digit3 = tf.nn.softmax(digit3) if not is_training else digit3
        digit4 = tf.nn.softmax(digit4) if not is_training else digit4
        # digit5 = tf.nn.softmax(digit5) if not is_training else digit5

        # we only apply softmax to testing network
        # out = tf.nn.softmax(out) if not is_training else out
    return digit1, digit2, digit3, digit4


# Create a graph for training
logits_train_digit0, logits_train_digit1, logits_train_digit2, logits_train_digit3 = \
    conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test_digit0, logits_test_digit1, logits_test_digit2, logits_test_digit3 = \
    conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train_digit0, labels=Y0))
loss_op1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train_digit1, labels=Y1))
loss_op2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train_digit2, labels=Y2))
loss_op3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train_digit3, labels=Y3))

# convert the label to one-hot label
# one_hot_labels0 = tf.one_hot(indices=tf.cast(Y0, tf.int32), depth=10)
# one_hot_labels1 = tf.one_hot(indices=tf.cast(Y1, tf.int32), depth=10)
# one_hot_labels2 = tf.one_hot(indices=tf.cast(Y2, tf.int32), depth=10)
# one_hot_labels3 = tf.one_hot(indices=tf.cast(Y3, tf.int32), depth=10)
#
# # calculate total loss
# loss_op0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train_digit0, labels=one_hot_labels0))
# loss_op1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train_digit1, labels=one_hot_labels1))
# loss_op2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train_digit2, labels=one_hot_labels2))
# loss_op3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train_digit3, labels=one_hot_labels3))

loss_op = (loss_op0 + loss_op1 + loss_op2 + loss_op3)/4.0
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# accuracy
# # label 0 is:
correct_pred0 = tf.equal(tf.argmax(logits_train_digit0, 1), tf.cast(Y0, tf.int64))
accuracy_train0 = tf.reduce_mean(tf.cast(correct_pred0, tf.float32))
# # label 1 is:
correct_pred1 = tf.equal(tf.argmax(logits_train_digit1, 1), tf.cast(Y1, tf.int64))
accuracy_train1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))
# # label 2 is:
correct_pred2 = tf.equal(tf.argmax(logits_train_digit2, 1), tf.cast(Y2, tf.int64))
accuracy_train2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))
# # label 3 is:
correct_pred3 = tf.equal(tf.argmax(logits_train_digit3, 1), tf.cast(Y3, tf.int64))
accuracy_train3 = tf.reduce_mean(tf.cast(correct_pred3, tf.float32))


# Evaluate model (with test logits, for dropout to be disabled)
# label 0 is:
# correct_pred0 = tf.equal(tf.argmax(logits_test_digit0, 1), tf.cast(Y0, tf.int64))
# accuracy_test0 = tf.reduce_mean(tf.cast(correct_pred0, tf.float32))
# # label 1 is:
# correct_pred1 = tf.equal(tf.argmax(logits_test_digit1, 1), tf.cast(Y1, tf.int64))
# accuracy_test1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))
# # label 2 is:
# correct_pred2 = tf.equal(tf.argmax(logits_test_digit2, 1), tf.cast(Y2, tf.int64))
# accuracy_test2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))
# # label 3 is:
# correct_pred3 = tf.equal(tf.argmax(logits_test_digit3, 1), tf.cast(Y3, tf.int64))
# accuracy_test3 = tf.reduce_mean(tf.cast(correct_pred3, tf.float32))


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training and Initialize the iterator
with tf.Session() as sess:
    # sess.run(iterator.initializer)
    sess.run(init)
    sess.run(traindata_init)
    # sess.run(valdata_init)
    saver = tf.train.Saver(max_to_keep=3)
    ckpt = tf.train.get_checkpoint_state('./model_svhn7')
    if ckpt is None:
        print("Model not found, please train your model first...")
    else:
        path = ckpt.model_checkpoint_path
        print('loading pre-trained model from %s.....' % path)
        saver.restore(sess, path)
    # Training cycle
    for step in range(1, num_steps + 1):
        sess.run(train_op)
        if step % train_display == 0 or step == 1:
            # Run optimization and calculate batch loss and accuracy
            loss, acc0, acc1, acc2, acc3 = sess.run([loss_op, accuracy_train0, accuracy_train1,
                                  accuracy_train2, accuracy_train3])
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc0) + ", {:.3f}".format(acc1) + ", {:.3f}".format(acc2) + ", {:.3f}".format(acc3))

        # if step % val_display == 0:
        #     loss, acct0, acct1, acct2, acct3 = sess.run([loss_op, accuracy_test0, accuracy_test1, accuracy_test2,
        #                                                    accuracy_test3])
        #     print("\033[1;36m=\033[0m"*60)
        #     print("\033[1;36mStep %d, Minibatch Loss= %.4f, Test Accuracy= %.4f, %.4f, %.4f, %.4f\033[0m" %
        #           (step, loss, acct0, acct1, acct2, acct3))
        #     print("\033[1;36m=\033[0m"*60)

        if step % 1000 == 0:
            path_name = "./model_svhn7/model" + str(step) + ".ckpt"
            print(path_name)
            saver.save(sess, path_name)
            print("model has been saved")

    print("Optimization Finished!")
