# coding:  utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# 载入数据集
mnist = input_data.read_data_sets("/Users/bruce/programme/Python/datasets/MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1)+b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3)+b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4)+b4)

# 二次代价函数
#  loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy " + str(test_acc) + ", Training Accuracy " + str(train_acc))


# example 2
import tensorflow.examples.tutorials.mnist.input_data as tf_input

mnist = tf_input.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2)+b2)

y_ = tf.placeholder(tf.float32,  [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

tf.global_variables_initializer().run()
resultMat = []
for i in range(5000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run({x: batch_x, y_: batch_y, keep_prob: 0.75})

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    print('the step is %d, accuracy is %.2f, the loss is %.2f' %
          (i, accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}),
           cross_entropy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
    # print(('the accur is %f')%accuracy.eval({x: mnist.test.images,  y_: mnist.test.labels,  keep_prob: 1.0}))
    resultMat.append(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
print(resultMat)



