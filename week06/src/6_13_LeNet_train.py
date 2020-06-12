import tensorflow as tf
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
DATASET_PATH = '/raid/bruce/dog_cat/train/'  # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 2  # CHANGE HERE, total number of classes
IMG_HEIGHT = 32  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 32  # CHANGE HERE, the image width to be resized to
CHANNELS = 1  # The 3 color channels, change to 1 if grayscale
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Parameters
learning_rate = 0.0001
num_steps = 10000
batch_size = 128
display_step = 100
val_display = 300
save_check = 500
# Network Parameters
dropout = 0.15  # Dropout, probability to drop during training the units


# Reading the dataset
def get_files_path(file_dir):
    cats = []
    dogs = []
    label_cats = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir+file)
            label_cats.append(0)
        else:
            dogs.append(file_dir+file)
            label_dogs.append(1)

    print("there are %d cats and there are %d dogs" % (len(cats), len(dogs)))
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    # print(label_list)
    label_list = [int(i) for i in label_list]
    return image_list, label_list


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
    print(dirPath)
    np.random.seed(0)
    pathDir = dataset_path
    classes = os.walk(pathDir).__next__()[1]
    imagePaths = list()
    labels = list()
    label = 0
    for folder in classes:
        folderPath = os.path.join(pathDir, folder)
        for img in os.listdir(folderPath):
            if img.endswith(".jpeg") or img.endswith('.JPEG') or img.endswith(".jpg"):
                imgpath = os.path.join(folderPath, img)
                image = Image.open(imgpath)
                image = image.resize((32, 32))
                image = np.array(image)
                if image.shape == (32, 32, 3):
                    imagePaths.append(imgpath)
                    labels.append(label)
        label += 1

    print('the imagePaths is:', imagePaths)
    print("the labels is: ", labels)
    # Convert to Tensor,保存的是图片的路径 和 labels
    return imagePaths, labels


def _parse_function(imagepaths, labels):
    """
    数据预处理环节
    :param imagepaths:
    :param labels:
    :return:
    """
    image_string = tf.read_file(imagepaths)
    image_decode = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # image_decode = tf.image.rgb_to_grayscale(image_decode)
    image_decode = tf.image.convert_image_dtype(image_decode, tf.float32)
    image_resized = tf.image.resize_images(image_decode, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.AREA)
    return image_resized, labels


# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# image, labels = read_images(DATASET_PATH)
image, labels = get_files_path(DATASET_PATH)
image_train, image_test, label_train, label_test = train_test_split(image, labels, test_size=0.25, random_state=0)

# Create a dataset tensor from the images and the labels
traindata = tf.data.Dataset.from_tensor_slices((image_train, label_train)).\
    map(_parse_function).\
    shuffle(buffer_size=1000).\
    batch(batch_size).\
    repeat().\
    prefetch(batch_size)

valdata = tf.data.Dataset.from_tensor_slices((image_test, label_test)).\
    map(_parse_function).\
    shuffle(buffer_size=1000).\
    batch(batch_size).\
    repeat().\
    prefetch(batch_size)


iterator = tf.data.Iterator.from_structure(traindata.output_types, traindata.output_shapes)
X, Y = iterator.get_next()


traindata_init = iterator.make_initializer(traindata)
valdata_init = iterator.make_initializer(valdata)


def conv_net(x, n_classes, dropout, reuse, is_training):
    """
    Create model padding有两种类型，一种是valid，还有一种是same，valid表示不够卷积核大小就丢弃，same表示不够的话就补0
    max_pooling2d 默认的padding是valid，就是说不够的话丢弃，否则same补充0；
    :param x:
    :param n_classes:
    :param dropout:
    :param reuse:
    :param is_training:
    :return:
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        """
        LeNet 网络结构实现:
        输入为32*32单通单图片,
        卷积层: filter 大小是 5 * 5 步长为1, 输出通道是 6
        池化层: filter 大小是 2 * 2 步长是2
        卷积层: filter 大小是 5 * 5 步长是1, 输出通道是16
        池化层: filter 大小是 2 * 2 步长是2
        全连接层: 120 个神经元
        全连接层: 84 个神经元
        全连接层: 2个神经元
        """
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, activation=tf.nn.sigmoid)
        # average Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(pool1, 16, 5, activation=tf.nn.sigmoid)
        # average Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(pool2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 120)
        fc1 = tf.layers.dense(fc1, 84)
        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights, but has
# different behavior for 'dropout' (not applied).
logits_test = conv_net(X, N_CLASSES, dropout=0.0, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(init)
    sess.run(traindata_init)
    sess.run(valdata_init)
    # save the model
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('./model_lenet')
    if ckpt is None:
        print("Model not found, please train your model first...")
    else:
        path = ckpt.model_checkpoint_path
        print('loading pre-trained model from %s.....' % path)
        saver.restore(sess, path)

    # Training cycle
    for step in range(1, num_steps + 1):
        # Run optimization
        sess.run(train_op)
        # print('the logits train is:\n', sess.run(tf.argmax(logits_train)))
        # print('the logits test is: \n', sess.run(tf.argmax(logits_test, 1)))
        # print('the real y is: ', sess.run(Y))

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            # (note that this consume a new batch of data)
            loss, acc, correct_ = sess.run([loss_op, accuracy, correct_pred])
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

            path_name = "./model_lenet/model" + str(step) + ".ckpt"
            if step % save_check == 0:
                saver.save(sess, path_name)
                print("model has been saved")

        if step % val_display == 0:
            avg_acc = 0
            loss, acc = sess.run([loss_op, accuracy])
            # avg_acc += acc[0]
            print("="*58)
            print("Step " + str(step) + ', Minibatch Loss= ' + "{:.4f}".format(loss) + ", Test Accuracy= " +
                  "{:.3f}".format(acc))
            print("="*58)
print("Optimization Finished!")
