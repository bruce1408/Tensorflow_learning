import os
import tensorflow as tf
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

# Image Parameters
N_CLASSES = 2  # CHANGE HERE, total number of classes
IMG_HEIGHT = 128  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 128  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale
n_classes = N_CLASSES  # MNIST total classes (0-9 digits)
dropout = 0.25
num_steps = 1000
train_display = 100
val_display = 300
learning_rate = 0.0001
BATCHSIZE = 32


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
    # imagepaths, labels = list(), list()
    # if mode == 'file':
    #     # Read dataset file
    #     with open(dataset_path) as f:
    #         data = f.read().splitlines()
    #     for d in data:
    #         imagepaths.append(d.split(' ')[0])
    #         labels.append(int(d.split(' ')[1]))
    # elif mode == 'folder':
    #     # An ID will be affected to each sub-folders by alphabetical order
    #     label = 0
    #     # List the directory
    #
    #     classes = sorted(os.walk(dataset_path).__next__()[1])
    #     # List each sub-directory (the classes)
    #     for c in classes:
    #         c_dir = os.path.join(dataset_path, c)
    #         # print("the c_dir is: ", c_dir)
    #         walk = os.walk(c_dir).__next__()
    #         # print("the walk is: ", walk)
    #         # Add each image to the training set
    #         for sample in walk[2]:
    #             # Only keeps jpeg images
    #             if sample.endswith('.jpg') or sample.endswith('.jpeg'):
    #                 imagepaths.append(os.path.join(c_dir, sample))
    #                 labels.append(label)
    #         label += 1
    # else:
    #     raise Exception("Unknown mode.")
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
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


traindata = tf.data.TFRecordDataset("./train_dog_cat.tfrecord").\
    map(_parse_function).repeat().batch(BATCHSIZE).prefetch(BATCHSIZE)

valdata = tf.data.TFRecordDataset("./test_dog_cat.tfrecord").\
    map(_parse_function).repeat().batch(BATCHSIZE).prefetch(BATCHSIZE)
# Create an iterator over the dataset

iterator = tf.data.Iterator.from_structure(traindata.output_types, traindata.output_shapes)
X, Y = iterator.get_next()

traindata_init = iterator.make_initializer(traindata)
valdata_init = iterator.make_initializer(valdata)

print(X.shape)


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


# Create a graph for training
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
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
    ckpt = tf.train.get_checkpoint_state('./model1')
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
            loss, acc = sess.run([loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

        if step % val_display == 0:
            avg_acc = 0
            loss, acc = sess.run([loss_op, accuracy])
            print("\033[1;36m=\033[0m"*60)
            print("\033[1;36mStep %d, Minibatch Loss= %.4f, Test Accuracy= %.4f\033[0m" % (step, loss, acc))
            print("\033[1;36m=\033[0m"*60)

        if step % 500 == 0:
            path_name = "./model1/model" + str(step) + ".ckpt"
            print(path_name)
            saver.save(sess, path_name)
            print("model has been saved")

    print("Optimization Finished!")
