import tensorflow as tf
# from resnets_utils import *
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
"""
resnet 有 5个stage,第一个stage是卷积,其他都是block building块,每一个building 有3层
"""
# parameter 网络参数
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCHSIZE = 64
num_steps = 30000
train_display = 100
val_display = 1000
learning_rate = 0.01
# training_id = tf.placeholder(tf.bool)
training_id = tf.placeholder_with_default(True, shape=(), name='training')


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
    image = image/225.0
    image = image - 0.5
    image = image * 2.0
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


def identity_block(X_input, kernel_size, filters, stage, block, TRAINING):
    """
    Implementation of the identity block as defined in Figure 3
    shortcut路径不包含卷积单元
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
        x = tf.layers.conv2d(X_input, filter1, kernel_size=(1, 1), strides=(1, 1), padding='same', name=conv_name_base + '2a')
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

    Arguments:
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

        # SHORTCUT PATH
        X_shortcut = tf.layers.conv2d(X_shortcut, filters[2], (1, 1), strides=(stride, stride), padding='same',
                                      name=conv_name_base + '1')
        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
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
    x = tf.pad(X, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")

    assert (x.shape == (x.shape[0], 230, 230, 3))  # 把原来的 224 * 224 变成 230 * 230

    # stage 1
    x = tf.layers.conv2d(x, filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')  # 加上padding也可以
    x = tf.layers.batch_normalization(x, axis=3, training=training, name='bn_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='same')

    # stage 2: 1个convBlock + 2个identityBlock
    x = residual_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', TRAINING=training, stride=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', TRAINING=training)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', TRAINING=training)

    # stage 3: 1个convBlock + 3个identityBlock
    x = residual_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a', TRAINING=training, stride=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', TRAINING=training)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', TRAINING=training)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', TRAINING=training)

    # stage 4: 1个convBlock + 5个identityBlock
    x = residual_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', TRAINING=training, stride=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', TRAINING=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', TRAINING=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', TRAINING=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', TRAINING=training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', TRAINING=training)

    # stage 5: 1个convBlock + 2个identityBlock
    x = residual_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a', TRAINING=training, stride=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', TRAINING=training)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', TRAINING=training)

    # 均值池化层
    x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=(1, 1))

    # 全连接层
    flatten = tf.layers.flatten(x, name='flatten')
    dense1 = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1, units=classes, activation=tf.nn.softmax)
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
    X_train shape: (32, 224, 224, 3)
    Y_train shape: (32, 2)
    X_test shape: (32, 224, 224, 3)
    Y_test shape: (32, 2)
    """

    m, H_size, W_size, C_size = X.shape
    classes = 2
    assert ((H_size, W_size, C_size) == (IMG_HEIGHT, IMG_WIDTH, 3))

    Y = tf.one_hot(Y, 2)
    logits = ResNet50_reference(X, training_id, classes)
    loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits))

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    # these lines are needed to update the batchnorma moving averages
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op)

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        assert (X.shape == (X.shape[0], IMG_HEIGHT, IMG_WIDTH, 3))
        assert (Y.shape[1] == classes)

        sess.run(traindata_init)
        saver = tf.train.Saver(max_to_keep=3)
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
                loss, acc = sess.run([loss_op, accuracy], {training_id: False})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", train acc = " + "{:.2f}".format(acc))

                if step % val_display == 0 and step is not 0:
                    acc = check_accuracy(sess, correct_prediction, training_id, valdata_init, val_display)
                    loss = sess.run(loss_op, {training_id: False})
                    print("\033[1;36m=\033[0m" * 60)
                    print("\033[1;36mStep %d, Minibatch Loss= %.4f, Test Accuracy= %.4f\033[0m" % (step, loss, acc))
                    print("\033[1;36m=\033[0m" * 60)

            if step % 1000 == 0:
                path_name = "./model_resnet/model" + str(step) + ".ckpt"
                print(path_name)
                saver.save(sess, path_name)
                print("model has been saved")

        print("Optimization Finished!")

        # for i in range(10000):
        #     X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
        #     _, cost_sess = sess.run([train_op, loss], feed_dict={X: X_mini_batch, Y: Y_mini_batch})
        #
        #     if i % 50 == 0:
        #         print(i, cost_sess)
        #
        # sess.run(tf.assign(TRAINING, False))
        #
        # training_acur = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
        # testing_acur = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
        # print("traing acurracy: ", training_acur)
        # print("testing acurracy: ", testing_acur)


if __name__ == '__main__':
    main()
