# coding: utf-8
import os
import warnings
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# 载入数据集
mnist = input_data.read_data_sets("../datasets/fashion_mnist", one_hot=True)
# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次

n_batch = mnist.train.num_examples // batch_size  # 一次完全训练需要多少batch数
print(n_batch)  # n_batch 550, train=55000, test_example = 10000个

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
# W = tf.Variable(tf.random_normal([784, 10])) // 如果是随机正太分布的初始参数，效果没有0好
# b = tf.Variable(tf.random_normal([10]))
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)
print(prediction)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量，只要有variable必须加初始化
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率，tf.cast是类型转换，把correct_prediction转换为tf.float32,例如[true,false,true,false]->[1,0,1,0]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把这个加起来就是准确率
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):  # 一共是完整迭代整个数据21次
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, prediction_ = sess.run([train_step, prediction], feed_dict={x: batch_xs, y: batch_ys})
            # print("the prediction is:\n", prediction_.shape)

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ", Testing Accuracy " + str(acc))
# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(3):
        for batch in range(5500):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc, loss_ = sess.run([accuracy, loss], feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc) + " ,loss= " + str(loss_))

# import tensorflow as tf
# import numpy as np
# from sklearn.utils import shuffle
# from sklearn.preprocessing import OneHotEncoder
#
# # 导入 MINST 数据集
# # from tensorflow.examples.tutorials.mnist import input_data
# # mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
# mnist = tf.keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data("/Users/bruce/program/Python_file/datasets/mnist.npz")
# print('X_train.shape:', X_train.shape)
# print('X_test.shape:', X_test.shape)
# print('y_train.shape:', y_train.shape)
# print('y_test.shape:', y_test.shape)
#
# # 参数设置
# learning_rate = 0.001
# training_epochs = 50
# batch_size = 100
# display_step = 1
#
# # Network Parameters
# n_hidden_1 = 256  # 1st layer number of features
# n_hidden_2 = 256  # 2nd layer number of features
# n_input = 784  # MNIST data 输入 (img shape: 28*28)
# n_classes = 10  # MNIST 列别 (0-9 ，一共10类)
#
#
# def onehot(y, start, end, categories='auto'):
#     ohot = OneHotEncoder()
#     a = np.linspace(start, end - 1, end - start)
#     b = np.reshape(a, [-1, 1]).astype(np.int32)
#     ohot.fit(b)
#     c = ohot.transform(y).toarray()
#     return c
#
#
# def MNISTLable_TO_ONEHOT(X_Train, Y_Train, X_Test, Y_Test, shuff=True):
#     Y_Train = np.reshape(Y_Train, [-1, 1])
#     Y_Test = np.reshape(Y_Test, [-1, 1])
#     Y_Train = onehot(Y_Train.astype(np.int32), 0, n_classes)
#     Y_Test = onehot(Y_Test.astype(np.int32), 0, n_classes)
#     if shuff == True:
#         X_Train, Y_Train = shuffle(X_Train, Y_Train)
#         X_Test, Y_Test = shuffle(X_Test, Y_Test)
#         return X_Train, Y_Train, X_Test, Y_Test
#
#
# X_train, y_train, X_test, y_test = MNISTLable_TO_ONEHOT(X_train, y_train, X_test, y_test)
#
# # tf Graph input
# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])
#
#
# # Create model
# def multilayer_perceptron(x, weights, biases):
#     # Hidden layer with RELU activation
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     # Hidden layer with RELU activation
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#     # Output layer with linear activation
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     return out_layer
#
#
# # Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
#
# # 构建模型
# pred = multilayer_perceptron(x, weights, biases)
#
# # Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# # 启动session
# with tf.Session() as sess:
#     sess.run(init)
#
#     # 启动循环开始训练
#     for epoch in range(training_epochs):
#         avg_cost = 0.
#         total_batch = int(X_train.shape[0] / batch_size)
#         # 遍历全部数据集
#         for i in range(total_batch):
#             #    batch_x, batch_y = mnist.train.next_batch(batch_size)
#             # Run optimization op (backprop) and cost op (to get loss value)
#             batch_x = X_train[i * batch_size:(i + 1) * batch_size, :]
#             batch_x = np.reshape(batch_x, [-1, 28 * 28])
#             batch_y = y_train[i * batch_size:(i + 1) * batch_size, :]
#             correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#             Accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#             _, c, Acc = sess.run([optimizer, cost, Accuracy], feed_dict={x: batch_x,
#                                                                          y: batch_y})
#             # Compute average loss
#             avg_cost += c / total_batch
#         # 显示训练中的详细信息
#         if epoch % display_step == 0:
#             print("Epoch:", '%04d' % (epoch + 1), "cost=",
#                   "{:.9f}".format(avg_cost), "Accuracy:", Acc)
#     print(" Finished!")
#
#     # 测试 model
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     # 计算准确率
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     X_test = np.reshape(X_test, [-1, 28 * 28])
#     print("Test Accuracy:", accuracy.eval({x: X_test, y: y_test}))
#     print(sess.run(tf.argmax(y_test[:30], 1)), "Real Number")
#     print(sess.run(tf.argmax(pred[:30], 1), feed_dict={x: X_test, y: y_test}), "Prediction Number")
