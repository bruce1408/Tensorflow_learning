
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# 载入数据集
mnist = input_data.read_data_sets("/home/chenxi/Tensorflow_learning/MNIST_data", one_hot=True)
# 每个批次100张照片
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784], name='input_img')
y = tf.placeholder(tf.float32, [None, 10], name='input_label')
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W)+b, name='prediction')

#  loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""
定义一个小的网络结构，保存模型的时候使用Saver这个类
保存模型的时候
saver = tf.train.Saver()
saver.save(sess, ./path/MODELNAME.ckpt)

然后加载的话，使用saver.restore(sess, 'MODELNAME.ckpt')即可
"""
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ", Testing Accuracy " + str(acc))
    # 保存模型
    ckpt = tf.train.get_checkpoint_state('./net')
    if ckpt is None:
        print('Model not found, please train your model first')
        saver.save(sess, './net/myModel_10001')

    else:
        path = ckpt.model_checkpoint_path
        saver.save(sess, './net/myModel_10001')








