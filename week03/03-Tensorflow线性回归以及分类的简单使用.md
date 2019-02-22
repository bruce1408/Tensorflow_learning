## TensorFlow非线性回归以及分类的简单问题，softmax介绍

### 一、TensorFlow实现非线性回归

（对应代码：`3-1非线性回归.py`）

``` python
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点，值在-0.5~0.5中，产生了200行一列的矩阵
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 产生随机噪声
noise = np.random.normal(0, 0.02, x_data.shape)
# 给y_data加入噪声 y = x^2 + noise
y_data = np.square(x_data) + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层,中间层权值为一行十列的矩阵
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# 产生偏置值
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 预测结果：y = x * w + b
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 使用tanh作为激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层，权值为十行一列的矩阵
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
```

运行结果如下：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/15616302.jpg)

### 二、TensorFlow解决手写数字识别（简单版本）

#### 1、MNIST数据集介绍

MNIST数据集的官网：[Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)

下载下来的数据集被分成两部分：60000 行的训练数据集（mnist.train）和 10000 行的测试数据集（mnist.test）。

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/21661076.jpg)

每一张图片包含`28*28`个像素，我们把这一个数组展开成一个向量，长度是`28*28=784`。因此在
 MNIST 训练数据集中 `mnist.train.images` 是一个形状为 [60000, 784] 的张量，第一个维度数字用
来索引图片，第二个维度数字用来索引每张图片中的像素点。图片里的某个像素的强度值介于 0-1 之间。

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/36607474.jpg)

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/96119059.jpg)

MNIST 数据集的标签是介于 0-9 的数字，我们要把标签转化为“one-hot vectors”。一个 one-
hot 向量除了某一位数字是 1 以外，其余维度数字都是 0，比如标签 0 将表示为`([1,0,0,0,0,0,0,0,0,0])`
，标签 3 将表示为`([0,0,0,1,0,0,0,0,0,0])` 。

因此， `mnist.train.labels` 是一个 [60000, 10] 的数字矩阵。

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/50265880.jpg)

#### 2、神经网络构建

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/25158586.jpg)

#### 3、Softmax函数

我们知道 MNIST 的结果是 0-9，我们的模型可能推测出一张图片是数字 9 的概率是80%，是数字 8
 的概率是 10%，然后其他数字的概率更小，总体概率加起来等于 1。这是一个使用 softmax 回归模
型的经典案例。softmax 模型可以用来给不同的对象分配概率。

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/96955391.jpg)

比如输出结果为[1,5,3]：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/57443260.jpg)

#### 4、编码实现

（对应代码：`3-2MNIST数据集分类简单版本.py`）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
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
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
```

在我笔记本上运行结果如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0,Testing Accuracy 0.8313
Iter 1,Testing Accuracy 0.8703
Iter 2,Testing Accuracy 0.8813
Iter 3,Testing Accuracy 0.8876
Iter 4,Testing Accuracy 0.8938
Iter 5,Testing Accuracy 0.8976
Iter 6,Testing Accuracy 0.9002
Iter 7,Testing Accuracy 0.9013
Iter 8,Testing Accuracy 0.9043
Iter 9,Testing Accuracy 0.9056
Iter 10,Testing Accuracy 0.9064
Iter 11,Testing Accuracy 0.9066
Iter 12,Testing Accuracy 0.9082
Iter 13,Testing Accuracy 0.9095
Iter 14,Testing Accuracy 0.9096
Iter 15,Testing Accuracy 0.9109
Iter 16,Testing Accuracy 0.9128
Iter 17,Testing Accuracy 0.9128
Iter 18,Testing Accuracy 0.9132
Iter 19,Testing Accuracy 0.9137
Iter 20,Testing Accuracy 0.9137
```

关于载入数据集代码`mnist = input_data.read_data_sets("MNIST_data", one_hot=True)`补充下：

1. 第一个参数直接填写文件夹名称，则表示使用的为当前程序路径，可以改为其他目录，比如`D:\\mnist_data\\`

2. 下载后的数据集如下：

   ![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/52299788.jpg)

   如果下载不下来，可以网上搜索单独下载保存到本地。







