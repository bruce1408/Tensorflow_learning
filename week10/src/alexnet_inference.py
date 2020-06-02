import tensorflow as tf

IMG_HEIGHT = 224
IMG_WIDTH = 224


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

