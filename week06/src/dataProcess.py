import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets('../datasets/MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
image = tf.reshape(x, [-1, 28, 28, 1])

conv1 = tf.layers.conv2d(image, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same',
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                         name='conv1')

bn1 = tf.layers.batch_normalization(conv1, training=True, name='bn1')
pool1 = tf.layers.max_pooling2d(bn1, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool1')
conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                         name='conv2')

bn2 = tf.layers.batch_normalization(conv2, training=True, name='bn2')
pool2 = tf.layers.max_pooling2d(bn2, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool2')

flatten_layer = tf.contrib.layers.flatten(pool2, 'flatten_layer')
weights = tf.get_variable(shape=[flatten_layer.shape[-1], 10], dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(stddev=0.1), name='fc_weights')
biases = tf.get_variable(shape=[10], dtype=tf.float32,
                         initializer=tf.constant_initializer(0.0), name='fc_biases')

logit_output = tf.nn.bias_add(tf.matmul(flatten_layer, weights), biases, name='logit_output')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logit_output))

pred_label = tf.argmax(logit_output, 1)
label = tf.argmax(y_, 1)

accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_label, label), tf.float32))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                              initializer=tf.constant_initializer(0), trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=5000,
                                           decay_rate=0.1, staircase=True)
opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name='optimizer')
with tf.control_dependencies(update_ops):
    grads = opt.compute_gradients(cross_entropy)
    train_op = opt.apply_gradients(grads, global_step=global_step)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
sess = tf.InteractiveSession(config=tf_config)
sess.run(tf.global_variables_initializer())

# only save trainable and bn variables
var_list = tf.trainable_variables()
if global_step is not None:
    var_list.append(global_step)
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list,max_to_keep=5)
# save all variables
# saver = tf.train.Saver(max_to_keep=5)

if tf.train.latest_checkpoint('ckpts') is not None:
    saver.restore(sess, tf.train.latest_checkpoint('ckpts'))
train_loops = 10000
for i in range(train_loops):
    batch_xs, batch_ys = mnist.train.next_batch(32)
    _, step, loss, acc = sess.run([train_op, global_step, cross_entropy, accuracy],
                                  feed_dict={x: batch_xs, y_: batch_ys})
    if step % 100 == 0:  # print training info
        log_str = 'step:%d \t loss:%.6f \t acc:%.6f' % (step, loss, acc)
        tf.logging.info(log_str)
    if step % 1000 == 0:  # save current model
        save_path = os.path.join('ckpts', 'mnist-model.ckpt')
        saver.save(sess, save_path, global_step=step)

sess.close()
