import os
import tensorflow as tf
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# Dataset Parameters - CHANGE HERE
DATASET_PATH = '101_ObjectCategories'  # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 102  # CHANGE HERE, total number of classes
IMG_HEIGHT = 64  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale
n_classes = N_CLASSES  # MNIST total classes (0-9 digits)
dropout = 0.75
num_steps = 2000
display_step = 100
learning_rate = 0.01


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path, batch_size):
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
    #     # print("the calsses is: ", classes)
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
    # print(imagepaths)
    # print(labels)

    path = os.getcwd()
    dirPath = os.path.join(path, dataset_path)
    dirPath = dataset_path
    print(dirPath)
    imagePaths = list()
    labels = list()
    label = 0
    for parent, _, filenames in os.walk(dirPath):
        # print("the parent is: ", parent)
        for img in filenames:
            if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith(".JPEG"):
                imagePaths.append(os.path.join(parent, img))
                labels.append(label)
        label += 1
    for i in range(len(labels)):
        labels[i] = labels[i]-1
    return imagePaths, labels


def _parse_function(imagepaths, labels):
    image_string = tf.read_file(imagepaths)
    image_decode = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    image_resized = tf.image.resize_images(image_decode, [IMG_HEIGHT, IMG_WIDTH])
    return image_resized, labels

sess = tf.Session()

# convert to tensor
imagespaths, labels = read_images(DATASET_PATH, 64)
print(labels)
imagespaths = tf.convert_to_tensor(imagespaths, dtype=tf.string)
labels = tf.convert_to_tensor(labels, dtype=tf.int32)

# dataset pipeline
dataset = tf.data.Dataset.from_tensor_slices((imagespaths, labels))
dataset = dataset.map(_parse_function)
dataset = dataset.batch(batch_size=128)
dataset = dataset.repeat()
dataset = dataset.prefetch(64)

# Create an iterator over the dataset
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer)

X, Y = iterator.get_next()

# Neural Net Input (images, labels)
print(X.shape)

    # # Convert to Tensor,保存的是图片的路径 和 labels
    # imagsePaths = tf.convert_to_tensor(imagsePaths, dtype=tf.string)
    # labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # # Build a TF Queue, shuffle data
    # image, label = tf.train.slice_input_producer([imagsePaths, labels], shuffle=True)
    #
    # # Read images from disk
    # image = tf.read_file(image)
    # image = tf.image.decode_jpeg(image, channels=CHANNELS)
    #
    # # Resize images to a common size
    # image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    #
    # # Normalize
    # image = image * 1.0 / 127.5 - 1.0
    #
    # # Create batches
    # X, Y = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 8, num_threads=4)
    #
    # return X, Y


def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.
# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.
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
# Run the initializer
sess.run(init)
# Saver object
saver = tf.train.Saver()


# Start training
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# Initialize the iterator
sess.run(iterator.initializer)


# Start the data queue
# tf.train.start_queue_runners()

# Training cycle
for step in range(1, num_steps + 1):
    sess.run(train_op)
    if step % display_step == 0 or step == 1:
        # Run optimization and calculate batch loss and accuracy
        _, loss, acc = sess.run([train_op, loss_op, accuracy])
        print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +
              "{:.3f}".format(acc))
    # else:
    #     # Only run the optimization op (backprop)
    #     sess.run(train_op)

print("Optimization Finished!")

# Save your model
saver.save(sess, './model1/my_tf_model.ckpt')
# coord.request_stop()
# coord.join(threads)

