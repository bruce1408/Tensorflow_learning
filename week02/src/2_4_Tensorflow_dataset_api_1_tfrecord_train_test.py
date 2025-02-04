import os
import tensorflow as tf
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

"""
train the dataset from scratch
"""
# Image Parameters
N_CLASSES = 2  # CHANGE HERE, total number of classes
IMG_HEIGHT = 128  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 128  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale
n_classes = N_CLASSES  # MNIST total classes (0-9 digits)
# dropout = 0.5
num_steps = 10000
train_display = 100
val_display = 1000
learning_rate = 0.0001
BATCHSIZE = 64
save_check = 1000


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path):
    """
    imagepaths 保存的是所有的图片的路径
    :param dataset_path:
    :param mode:
    :param batch_size:
    :return:
    """
    path = os.getcwd()
    dirPath = os.path.join(path, dataset_path)
    imagePaths = list()
    labels = list()
    label = 0
    for parent, _, filenames in os.walk(dirPath):
        for img in filenames:
            if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith(".JPEG"):
                imagePaths.append(os.path.join(parent, img))
                labels.append(label)
        label += 1
    for i in range(len(labels)):
        labels[i] = labels[i] - 1
    return imagePaths, labels


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
        'img_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['img_raw'], tf.uint8)
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])
    image = tf.cast(image, tf.float32)
    image = image/225.0
    image = image - 0.5
    image = image * 2.0
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


# train data pipline
# repeat -> shuffle 和 shuffle -> repeat不一样
traindata = tf.data.TFRecordDataset("./train_dog_cat.tfrecord").\
    map(_parse_function).\
    shuffle(buffer_size=2000, reshuffle_each_iteration=True).\
    batch(BATCHSIZE).\
    repeat().\
    prefetch(BATCHSIZE)

# val data pipline
valdata = tf.data.TFRecordDataset("./test_dog_cat.tfrecord").\
    map(_parse_function).\
    batch(BATCHSIZE).\
    repeat().\
    prefetch(BATCHSIZE)
# Create an iterator over the dataset

iterator = tf.data.Iterator.from_structure(traindata.output_types, traindata.output_shapes)
X, Y = iterator.get_next()

traindata_init = iterator.make_initializer(traindata)
valdata_init = iterator.make_initializer(valdata)


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op, batches_to_check):
    # Initialize the validation dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    for i in range(batches_to_check):
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False, dropout: 0.0})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


# Define the newwork
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # Convolution Layer with 32 filters and a kernel size of 5
        # x = tf.reshape(x, shape=[-1, 64, 64, 3])
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
        conv1_1 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool1 = tf.layers.max_pooling2d(conv1_1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)

        conv3_1 = tf.layers.conv2d(pool2, 512, 3, activation=tf.nn.relu)
        # conv3_2 = tf.layers.conv2d(conv3_1, 512, 3, activation=tf.nn.relu)
        # conv3_3 = tf.layers.conv2d(conv3_2, 512, 3, activation=tf.nn.relu)
        # conv3_4 = tf.layers.conv2d(conv3_3, 512, 3, activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3_1, 2, 2)

        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu)
        # conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu)
        # conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, activation=tf.nn.relu)
        conv4_4 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4_4, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(pool4)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 4096)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 2048)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out
        # out = tf.nn.softmax(out)
    return out


is_training = tf.placeholder(tf.bool)
dropout = tf.placeholder(tf.float32)
# Create a graph for training
logits = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights, 注意测试的时候不丢弃网络
# logits_test = conv_net(X, N_CLASSES, dropout=0.0, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
# Initialize the iterator
with tf.Session() as sess:
    # sess.run(iterator.initializer)
    sess.run(init)
    sess.run(traindata_init)
    sess.run(valdata_init)
    saver = tf.train.Saver(max_to_keep=3)
    ckpt = tf.train.get_checkpoint_state('./model2_0')
    if ckpt is None:
        print("Model not found, please train your model first...")
    else:
        path = ckpt.model_checkpoint_path
        print('loading pre-trained model from %s.....' % path)
        saver.restore(sess, path)
    # Training cycle
    for step in range(1, num_steps + 1):
        sess.run(train_op, {is_training: True, dropout: 0.5})
        if step % train_display == 0 or step == 1:
            # Run optimization and calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], {is_training: False, dropout: 0.0})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

        if step % val_display == 0 and step is not 0:
            avg_acc = 0
            acc = check_accuracy(sess, correct_pred, is_training, valdata_init, 5000)
            loss = sess.run(loss_op, {is_training: False, dropout: 0.0})
            print("\033[1;36m=\033[0m"*60)
            print("\033[1;36mStep %d, Minibatch Loss= %.4f, Test Accuracy= %.4f\033[0m" % (step, loss, acc))
            print("\033[1;36m=\033[0m"*60)

        if step % 1000 == 0:
            path_name = "./model2_0/model" + str(step) + ".ckpt"
            print(path_name)
            saver.save(sess, path_name)
            print("model has been saved")

    print("Optimization Finished!")
