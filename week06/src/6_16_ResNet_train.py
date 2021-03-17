import tensorflow as tf
# from resnets_utils import *
import numpy as np
import os
from utils.logWriter import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
"""
ResNet 有 5个stage,第一个stage是卷积,其他都是block building块,每一个building 有3层
reference: https://blog.csdn.net/Cheungleilei/article/details/103610799
"""
# parameter 网络参数
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCHSIZE = 64
num_steps = 30000
train_display = 10
save_check = 2000
val_display = 500
learning_rate = 0.1
decay_rate = 0.96
decay_step = 200
check_acc = 150
log_path = './resnet_train'

global_step = tf.Variable(tf.constant(0), name='global_step', trainable=False)
training_id = tf.placeholder_with_default(False, shape=(), name='training')


def check_accuracy(sess, correct_prediction, training_id, dataset_init_op, batches_to_check):
    # Initialize the validation dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    for i in range(batches_to_check):
        try:
            correct_pred = sess.run(correct_prediction, {training_id: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


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


def identity_block(X_input, kernel_size, filters, stage, block, TRAINING):
    """
    Implementation of the identity block as defined in Figure 3
    shortcut 路径不包含卷积单元
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    https://www.cnblogs.com/wxshi/p/8317489.html
    https://blog.csdn.net/liangyihuai/article/details/79140481
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage" + str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1, kernel_size=(1, 1), strides=(1, 1), padding='same',
                             name=conv_name_base + '2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2a', training=TRAINING)
        x = tf.nn.relu(x)
        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')
        # batch_norm2 = tf.layers.batch_normalization(conv2, axis=3, name=bn_name_base+'2b', training=TRAINING)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, kernel_size=(1, 1), padding='same', name=conv_name_base + '2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(x, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def residual_block(X_input, kernel_size, filters, stage, block, TRAINING, stride=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments: 这里的shortcut 是有一条卷积的线路
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):
        # Retrieve Filters
        # filter1, filter2, filter3 = filters
        # Save the input value
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filters[0], kernel_size=(1, 1), strides=(stride, stride), padding='same',
                             name=conv_name_base + '2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2a', training=TRAINING)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filters[1], (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=TRAINING)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filters[2], (1, 1), padding='same', name=conv_name_base + '2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)
        # print('before x', x.shape)
        # print('before x_shortcut shape', X_shortcut.shape)

        # Short path
        X_shortcut = tf.layers.conv2d(X_shortcut, filters[2], (1, 1), strides=(stride, stride), padding='same',
                                      name=conv_name_base + '1')
        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        # print('the x_shortcut shape is: ', X_shortcut.shape)
        # print('the x shape is: ', x.shape)
        X_add_shortcut = tf.add(X_shortcut, x)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def ResNet50_reference(X, training, classes=2):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    Returns:
    """

    # input shape is (batch, 230, 230, channles)
    # x = tf.pad(X, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")

    # assert (x.shape == (x.shape[0], 230, 230, 3))  # 把原来的 224 * 224 变成 230 * 230

    # stage 1 ,或者上面的注释取消，加上padding也可以
    x = tf.layers.conv2d(X, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')
    print('the stage1.1 conv2d shape is:', x.get_shape())
    x = tf.layers.batch_normalization(x, axis=3, training=training, name='bn_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='same')
    print('the stage1.2 pool shape is:', x.get_shape())

    # stage 2: 1个convBlock + 2个identityBlock
    x = residual_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', TRAINING=training, stride=1)
    print('the stage2.1 residual shape is:', x.get_shape())
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', TRAINING=training)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', TRAINING=training)
    print('the stage2.2 identity shape is:', x.get_shape())

    # stage 3: 1个convBlock + 3个identityBlock
    x = residual_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a', TRAINING=training, stride=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', TRAINING=training)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', TRAINING=training)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', TRAINING=training)
    print('the stage3 identity shape is:', x.get_shape())

    # stage 4: 1个convBlock(shortcut 带卷积) + 5个identityBlock
    x = residual_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', TRAINING=training, stride=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', TRAINING=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', TRAINING=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', TRAINING=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', TRAINING=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', TRAINING=training)
    print('the stage4 identity shape is:', x.get_shape())

    # stage 5: 1个convBlock + 2个identityBlock
    x = residual_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a', TRAINING=training, stride=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', TRAINING=training)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', TRAINING=training)
    print('the stage5 identity shape is:', x.get_shape())

    # 均值池化层
    x = tf.layers.average_pooling2d(x, pool_size=(7, 7), strides=(1, 1))
    print('the stage pool shape is:', x.get_shape())

    # 全连接层
    flatten = tf.layers.flatten(x, name='flatten')
    print('the flatten shape is:', flatten.get_shape())
    # dense1 = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
    # print('the dense1 shape is:', dense1.get_shape())
    logits = tf.layers.dense(flatten, units=classes, activation=tf.nn.softmax)
    print('the last shape is:', logits.get_shape())

    return logits


def main():
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

    iterator = tf.data.Iterator.from_structure(traindata.output_types, traindata.output_shapes)
    X, Y = iterator.get_next()

    traindata_init = iterator.make_initializer(traindata)
    valdata_init = iterator.make_initializer(valdata)
    """
    number of training examples = 1080
    number of test examples = 120
    X_train shape: (64, 224, 224, 3)
    Y_train shape: (64,)
    X_test shape: (64, 224, 224, 3)
    Y_test shape: (64,)
    """
    global learning_rate
    m, H_size, W_size, C_size = X.shape
    classes = 2
    assert ((H_size, W_size, C_size) == (IMG_HEIGHT, IMG_WIDTH, 3))
    # 添加日志文件
    if not os.path.exists(log_path):
        print("====== The log folder was not found and is being generated !======")
        os.makedirs(log_path)
    else:
        print('======= The log path folder already exists ! ======')
    log = Logger('./resnet_train/resnet_train.log', level='info')
    logits = ResNet50_reference(X, training_id, classes)
    print("============ the layers shape above! ==============")
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits))

    learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_rate, staircase=True)

    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)

    # these lines are needed to update the batchnorma moving averages
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op)

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.cast(Y, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        assert (X.shape == (X.shape[0], IMG_HEIGHT, IMG_WIDTH, 3))

        sess.run(traindata_init)
        # saver bn moving_mean and moving_variance parameters
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=3)

        ckpt = tf.train.get_checkpoint_state('./model_resnet')
        if ckpt is None:
            print("Model not found, please train your model first...")
        else:
            path = ckpt.model_checkpoint_path
            print('loading pre-trained model from %s.....' % path)
            saver.restore(sess, path)

        for step in range(1, num_steps + 1):
            sess.run(train_op, {training_id: True})
            if step % train_display == 0 or step == 1:
                # Run optimization and calculate batch loss and accuracy
                lr, loss, acc = sess.run([learning_rate, loss_op, accuracy], {training_id: False, global_step: step})
                log.logger.info("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(
                    loss) + ", train acc = " + "{:.2f}".format(acc) + ", lr = " + "{:.4f}".format(lr))
                if step % val_display == 0 and step is not 0:
                    acc = check_accuracy(sess, correct_prediction, training_id, valdata_init, val_display)
                    loss = sess.run(loss_op, {training_id: False})
                    print("\033[1;36m=\033[0m" * 60)
                    log.logger.info("Step %d, Minibatch Loss= %.4f, Test Accuracy= %.4f\033[0m" % (step, loss, acc))
                    # print("\033[1;36mStep %d, Minibatch Loss= %.4f, Test Accuracy= %.4f\033[0m" % (step, loss, acc))
                    print("\033[1;36m=\033[0m" * 60)

            if step % save_check == 0:
                path_name = "./model_resnet/model" + str(step) + ".ckpt"
                saver.save(sess, path_name)
                print("model has been saved in %s" % path_name)

        acc = check_accuracy(sess, correct_prediction, training_id, valdata_init, check_acc)
        print('the val acc is: ', acc)
        print("Optimization Finished!")


if __name__ == '__main__':
    main()
