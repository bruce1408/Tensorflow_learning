import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Parameters
N_CLASSES = 2  # CHANGE HERE, total number of classes
IMG_HEIGHT = 224  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 224  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale
n_classes = N_CLASSES  # MNIST total classes (0-9 digits)
dropout = 0.25
num_steps = 1000
train_display = 100
val_display = 300
learning_rate = 0.0001
BATCHSIZE = 32
epochNum = 40


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
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


is_training = tf.placeholder(tf.bool)

traindata = tf.data.TFRecordDataset("/raid/bruced/dog_cat/train_dog_cat_224.tfrecord"). \
    map(_parse_function).repeat().batch(BATCHSIZE).prefetch(BATCHSIZE)

valdata = tf.data.TFRecordDataset("/raid/bruce/dog_cat/test_dog_cat_224.tfrecord"). \
    map(_parse_function).repeat().batch(BATCHSIZE).prefetch(BATCHSIZE)
# Create an iterator over the dataset

iterator = tf.data.Iterator.from_structure(traindata.output_types, traindata.output_shapes)
X, Y = iterator.get_next()

traindata_init = iterator.make_initializer(traindata)
valdata_init = iterator.make_initializer(valdata)


def inception_arg_scope(weight_decay=1e-4, is_training=True):

    # Set weight_decay for weights in Conv and FC layers. 给卷积和全连接设置默认参数, 卷积函数再单独设置激活函数
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu) as sc:
            return sc


with slim.arg_scope(inception_arg_scope(weight_decay=1e-4, is_training=is_training)):
    logits, _ = nets.vgg.vgg_16(X, num_classes=N_CLASSES, is_training=is_training)
# define the layers and fine-tune
var = tf.global_variables()  # 获取所有变量
var_to_restore = [val for val in var if 'fc8' not in val.name]  # 保留变量名中不含有fc8的变量
print(var_to_restore)
exclude = ['vgg_16/fc8']
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
restore_ckpt = tf.contrib.framework.assign_from_checkpoint_fn('../slimPretrainedModels/vgg_16.ckpt', variables_to_restore)

onehot_labels = tf.one_hot(Y, 2)
classif_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, label_smoothing=0.1)
l2_loss = tf.add_n(tf.losses.get_regularization_losses())

total_loss = tf.losses.get_total_loss()
global_step = tf.Variable(0, trainable=False)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step, clip_gradient_norm=2.0)

# without batch norm layers
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# evaluation metrics
prediction = tf.to_int32(tf.argmax(logits, 1))
correct_prection = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(correct_prection, tf.float32))
best_val_acc = tf.Variable(0.0, trainable=False)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # sess.run(iterator.initializer)
    sess.run(init)
    restore_ckpt(sess)

    saver = tf.train.Saver()
    best_acc = 0.0

    for epoch in range(epochNum):
        print("starting the learning epoch is %d " % epoch)
        sess.run(traindata_init)
        iteration = 0
        for _ in range(50):
            _, cur_step, acc = sess.run([train_step, global_step, accuracy], {is_training: True})
            if iteration % 10 == 0:
                print('E:%d/%d It:%d Step:%d acc is: %f' % (epoch, epochNum, iteration, cur_step - 1, acc))

            iteration += 1
        if epoch % 2 == 0 and epoch is not 0:
            sess.run(valdata_init)
            num_correct, num_samples = 0, 0
            correct_pred = sess.run(correct_prection, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
            curr_val_acc = float(num_correct)/num_samples
            print("the val acc is: ", curr_val_acc)
            if best_acc < curr_val_acc:
                curr_step = sess.run(global_step, {is_training: False})
                saver.save(sess, './models15/fine-tune_models', global_step=curr_step)
                best_accuracy = curr_val_acc
                print("saved models !")
                # sess.run(tf.assign(best_val_accuracy, best_accuracy))

    # sess.run(traindata_init)
    # sess.run(valdata_init)
    # saver = tf.train.Saver(max_to_keep=3)
    # ckpt = tf.train.get_checkpoint_state('./model1')
    # if ckpt is None:
    #     print("Model not found, please train your model first...")
    # else:
    #     path = ckpt.model_checkpoint_path
    #     print('loading pre-trained model from %s.....' % path)
    #     saver.restore(sess, path)
    # # Training cycle
    # for step in range(1, num_steps + 1):
    #     sess.run(train_op)
    #     if step % train_display == 0 or step == 1:
    #         # Run optimization and calculate batch loss and accuracy
    #         loss, acc = sess.run([loss_op, accuracy])
    #         print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +
    #               "{:.3f}".format(acc))
    #
    #     if step % val_display == 0:
    #         avg_acc = 0
    #         loss, acc = sess.run([loss_op, accuracy])
    #         print("\033[1;36m=\033[0m" * 60)
    #         print("\033[1;36mStep %d, Minibatch Loss= %.4f, Test Accuracy= %.4f\033[0m" % (step, loss, acc))
    #         print("\033[1;36m=\033[0m" * 60)
    #
    #     if step % 500 == 0:
    #         path_name = "./model1/model" + str(step) + ".ckpt"
    #         print(path_name)
    #         saver.save(sess, path_name)
    #         print("model has been saved")
    #
    print("Optimization Finished!")
