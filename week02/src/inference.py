import tensorflow as tf


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
