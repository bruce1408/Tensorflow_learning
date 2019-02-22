## 卷积神经网络CNN，用CNN解决MNIST分类问题

### 一、卷积神经网络

#### 认识卷积神经网络

相关资料：

- [李理：详解卷积神经网络](https://blog.csdn.net/qunnie_yi/article/details/80127218)
- 机器之心：[从入门到精通：卷积神经网络初学者指南](https://www.jiqizhixin.com/articles/2016-08-01-3)

- [图文并茂地讲解卷积神经网络](https://mp.weixin.qq.com/s/ixwEVn_WMkH28w5aYITnBw)
- charlotte77博客：[【深度学习系列】卷积神经网络CNN原理详解(一)——基本原理](https://www.cnblogs.com/charlotte77/p/7759802.html)

- 知乎：[能否对卷积神经网络工作原理做一个直观的解释？](https://www.zhihu.com/question/39022858/answer/194996805)
- ......

上面一些文章讲解的很清楚。

在这，顺带就多絮叨几句。到底深度学习是什么？有什么特点？下面举例来理解下这玩意：

> 假设有一张图，要做分类，传统方法需要手动提取一些特征，比如纹理啊，颜色啊，或者一些更高级的特征。然后再把这些特征放到像随机森林等分类器，给到一个输出标签，告诉它是哪个类别。而深度学习是输入一张图，经过神经网络，直接输出一个标签。特征提取和分类一步到位，避免了手工提取特征或者人工规则，从原始数据中自动化地去提取特征，是一种端到端（end-to-end）的学习。相较于传统的方法，深度学习能够学习到更高效的特征与模式。
>
> ![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/28008326.jpg)

应用到计算机视觉方向来说，简单来说就是深度学习可以自己学习到图像特征（其背后数学层面来看，也就是学到一个含非常多参数的函数），而不要我们自己去提取特征，即，不要我们去定义具有怎样特征才是猫，比如是否头部近圆形，颜面部短，耳呈三角形这样的特征才是猫，我们不用关心，深度学习能自动学习到特征（当然其实我们也不知道它到底学到了什么特征，所以被很多人称为「黑匣子」，可以看这篇文章 [1.1.1 什么是神经网络](https://blog.csdn.net/jiangjunshow/article/details/77368314) 体会下为什么这么说）。

传统经典网络存在的问题：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/94425588.jpg)

- 权值太多，计算量太大
- 权值太多，需要大量样本进行训练

经验之谈：样本数量最好是参数数量的 5—30 倍。数据量小而模型参数过的多容易出现过拟合现象。

#### 局部感受野

> 1962 年哈佛医学院神经生理学家 Hubel 和 Wiesel 通过对猫视觉皮层细胞的研究，提出了感受野（receptive field）的概念，1984 年日本学者 Fukushima 基于感受野概念提出的神经认知机（neocognitron）可以看作是卷积神经网络的第一个实现网络，也是感受野概念在人工神经网络领域的首次应用。

怎么理解局部感受野？举例来说。

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/54428842.jpg)

如上是一个全连接神经网络，全连接指的是：对 n-1 层和 n 层而言，n-1 层的任意一个节点，都和第 n 层所有节点有连接。明显地，网络很大的时候，参数很多，训练速度会很慢。

但在卷积网络里，我们把输入看成二维神经元，它的每一个神经元对应于图片在这个像素点的强度（灰度值），如下图所示：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/82756302.jpg)

把输入像素连接到隐藏层的神经元（怎么做的呢？——先把“图像所有像素值拉直”，再连接到隐藏层的神经元，见下图体会）。但是我们这里**不再把输入的每一个像素都连接到隐藏层的每一个神经元**，与之不同，我们把很小的相临近的区域内的输入连接在一起。具体的来讲，隐藏层的每一个神经元都会与输入层一个很小的区域（比如一个 3×3 的区域，也就是 9 个像素点）相连接。

![](http://p35l3ejfq.bkt.clouddn.com/18-10-8/45145457.jpg)

​					（*上图来源台湾大学李宏毅老师《深度学习》PPT内容*）

输入图像的这个区域叫做那个隐藏层神经元的局部感知域。这是输入像素的一个小窗口。每个连接都有一个可以学习的权重，此外还有一个 bias（偏置）。对于最右上的那个神经元（即，Filter——称过滤器、或滤波器、或卷积核）你可以想象成用来分析这个局部感知域的。

然后在整个输入图像上滑动这个局部感知域，这里就会涉及到步伐的问题了。我们可以一次移动一个像素（这个移动的值叫 stride），也可以一次移动不止一个像素。

说明：如果需要让图像在经过这样一次卷积处理后尺寸可以不变小，可以使用 padding，简单讲，就是把图片像素的边边角角拼一段像素上去，有两种方式，一种是填 0，另一种是将边边角角的像素直接复制一个填进去。那padding 要拼多少像素可以根据 filter 大小来定，filter 越大，需要拼的就越多。padding 是不是一定比不做效果好，这个视情况而定，多炼丹才知道。

另外关于 padding 有两种类型：

- SAME PADDING
- VALID PADDING

关于两者区别，下面摘录知乎一个回答：

> 唐突做一下解释：在卷积核移动逐渐扫描整体图时候，因为步长的设置问题，可能导致剩下未扫描的空间不足以提供给卷积核的，大小扫描 比如有图大小为`5*5`，卷积核为`2*2`，步长为 2，卷积核扫描了两次后，剩下一个元素，不够卷积核扫描了，这个时候就在后面补零，补完后满足卷积核的扫描，这种方式就是 same。如果说把刚才不足以扫描的元素位置抛弃掉，就是 valid 方式。
>
> 知乎：[TensorFlow中padding的SAME和VALID两种方式有何异同？](https://www.zhihu.com/question/60285234)

SAME PADDING：可能会给平面外部补 0，卷积窗口采样后得到一个跟原来平面大小相同的平面。

VALID PADDING：不会超出平面外部，卷积窗口采样后得到一个比原来平面小的平面。

1）假如有一个`28*28`的平面，用`2*2`并且步长为2的窗口对其进行 pooling 操作：

- 使用 SAME PADDING 的方式，得到`14*14`的平面
- 使用 VALID PADDING 的方式，得到`14*14`的平面

2）假如有一个`2*3`的平面，用`2*2`并且步长为 2 的窗口对其进行 pooling 操作

- 使用 SAME PADDING 的方式，得到`1*2`的平面
- 使用 VALID PADDING 的方式，得到`1*1`的平面

#### 权值共享

权值共享这个词最开始其实是由 LeNet5 模型提出来，在 1998 年，LeCun 发布了 LeNet 网络架构，就是下面这个： 

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/1224071.jpg)

虽然现在大多数的说法是 2012 年的 AlexNet 网络模型是深度学习的开端，但是 CNN 的开端最早其实可以追溯到 LeNet5 模型，它的几个特性在 2010 年初的卷积神经网络研究中被广泛的使用——其中一个就是**权值共享**。

到底怎么理解权值共享呢？——举例来说，所谓的权值共享就是说，给一张输入图片，用一个 filter 去扫这张图，filter 里面的数就叫权重，这张图每个位置是被同样的 filter 扫的，所以权重是一样的，也就是共享，说白了，就是整张图片在使用同一个 filter 的参数。

比如一个`3*3*1`的 filter（卷积核，另说明下：这里的 `*1` 表示为单通道图像），这个 filter 内 9 个的参数被整张图共享，而不会因为 filter 在图像上滑动后位置的不同而改变 filter 内的权系数，说的再直白一些，就是用一个 filter 不改变其内权系数的情况下卷积处理整张图片（当然 CNN 中每一层不会只有一个 filter 的，这样说只是为了方便解释而已）。下图为台大李宏毅老师《深度学习》PPT 某页内容，可以对照着理解下：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/37228310.jpg)

​				（*上图来源台湾大学李宏毅老师《深度学习》PPT内容*）

参考：

- 知乎：[如何理解卷积神经网络中的权值共享？](https://www.zhihu.com/question/47158818)
- [如何理解卷积神经网络中的权值共享](https://blog.csdn.net/chaipp0607/article/details/73650759)

推荐 B 站视频：[李宏毅-Convolutional Neural Network（CNN）-卷积神经网络](https://www.bilibili.com/video/av23593949/)

#### 卷积

单通道图像卷积过程（如下使用了一个卷积核卷积）：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/30544234.jpg)

动态图过程：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/3764121.jpg)

三通道（R、G、B ，可以理解为深度为 3）图像卷积过程（如下使用了两个卷积核卷积）：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/43496624.jpg)

多个卷积核卷积用来提取不同特征：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/87362924.jpg)

#### 池化(Pooling)

pooling 层可以非常有效地缩小图片的尺寸，显著减少参数数量，但 pooling 的目的并不仅在于此。pooling 目的是为了保持某种不变性（旋转、平移、伸缩等），常用的有 mean-pooling，max-pooling 和 Stochastic-pooling 三种。

1）mean-pooling（平均池化）：即对邻域内特征点只求平均，对背景保留更好

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/73494598.jpg)

2）max-pooling（最大池化）：对邻域内特征点取最大，对纹理提取更好

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/20901289.jpg)

3）Stochastic-pooling：介于两者之间，通过对像素点按照数值大小赋予概率，再按照概率进行亚采样，在平均意义上，与 mean-pooling 近似，在局部意义上，则服从 max-pooling 的准则

### 二、编码实现

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/8454243.jpg)

![](http://p35l3ejfq.bkt.clouddn.com/18-10-9/24953964.jpg)

定义 weight、bias；

卷积、激活、池化、下一层；

然后接 2 个全连接层，softmax，交叉熵、loss

（代码对应：`6-1卷积神经网络应用于MNIST数据集分类.py`，有修改——增加很多命名空间 scope）

``` python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
    return tf.Variable(initial)

# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x, W):
    # x input tensor of shape '[batch,in_height,in_width,in_channles]'
    # W filter / kernel tensor of shape [filter_height,filter_width,in_channels,out_channels]
    # `strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 2d的意思是二维的卷积操作

# 池化层
def max_pool_2x2(x):
    # ksize [1,x,y,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])  # 28*28
y = tf.placeholder(tf.float32, [None, 10])

# 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5, 5, 1, 32])  # 5*5的采样窗口，32个卷积核从1个平面抽取特征
b_conv1 = bias_variable([32])  # 每一个卷积核一个偏置值

# 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

# 初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5, 5, 32, 64])  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
b_conv2 = bias_variable([64])  # 每一个卷积核一个偏置值

# 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

# 28*28的图片第一次卷积后还是28*28（数组变小了，但是图像大小不变），第一次池化后变为14*14
# 第二次卷积后为14*14（卷积不会改变平面的大小），第二次池化后变为了7*7
# 进过上面操作后得到64张7*7的平面

# 初始化第一个全连接层的权值
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc1 = bias_variable([1024])  # 1024个节点

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
```

PS：我的笔记本跑不动啊o(╥﹏╥)o  显卡不支持深度学习框架。

显卡是否支持深度学习得看是否支持 CUDA（Compute Unified Device Architecture），如何查看显卡型号是否支持 CUDA：[TensorFlow-GPU：查看电脑显卡型号是否支持CUDN,以及相关软件下载与介绍](https://www.cnblogs.com/chamie/p/8707420.html)

遂还是拿实验室电脑，显卡 1080ti GPU 上跑吧，训练和测试过程如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0, Testing Accuracy= 0.8637
Iter 1, Testing Accuracy= 0.9654
Iter 2, Testing Accuracy= 0.9733
Iter 3, Testing Accuracy= 0.9783
Iter 4, Testing Accuracy= 0.9829
Iter 5, Testing Accuracy= 0.9832
Iter 6, Testing Accuracy= 0.9847
Iter 7, Testing Accuracy= 0.9873
Iter 8, Testing Accuracy= 0.9867
Iter 9, Testing Accuracy= 0.988
Iter 10, Testing Accuracy= 0.9901
Iter 11, Testing Accuracy= 0.9908
Iter 12, Testing Accuracy= 0.989
Iter 13, Testing Accuracy= 0.991
Iter 14, Testing Accuracy= 0.9903
Iter 15, Testing Accuracy= 0.9911
Iter 16, Testing Accuracy= 0.9909
Iter 17, Testing Accuracy= 0.9916
Iter 18, Testing Accuracy= 0.9913
Iter 19, Testing Accuracy= 0.9901
Iter 20, Testing Accuracy= 0.991
```

使用传统的神经网络我们可能只能达到 98% 点多的准确率，可以看到，使用卷积神经网络之后，我们可以达到 99% 的准确率，虽说差了百分之一，但是接近 100%，应该说算是比较大的提升。

完成卷积神经网络，记录下准确率和 loss 率的变化，完整代码如下：（代码对应：`7-1第六周作业.py`）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图
```



``` python
# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
    return tf.Variable(initial, name=name)
```



``` python
# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
```



``` python
# 卷积层
def conv2d(x, W):
    # x input tensor of shape `[batch, in_height, in_width, in_channels]`
    # W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # `strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```



``` python
# 池化层
def max_pool_2x2(x):
    # ksize [1,x,y,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```



``` python
# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')  # 5*5的采样窗口，32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

# 28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
# 第二次卷积后为14*14，第二次池化后变为了7*7
# 进过上面操作后得到64张7*7的平面

with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')  # 上一场有7*7*64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点

    # 把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # keep_prob用来表示神经元的输出概率
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(1001):	
        # 训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        # 记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))
```

运行结果：（用的实验室电脑，显卡 GTX 1080ti 跑的）

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0, Testing Accuracy= 0.1051, Training Accuracy= 0.1119
Iter 100, Testing Accuracy= 0.595, Training Accuracy= 0.5961
Iter 200, Testing Accuracy= 0.7324, Training Accuracy= 0.7365
Iter 300, Testing Accuracy= 0.7594, Training Accuracy= 0.7579
Iter 400, Testing Accuracy= 0.8423, Training Accuracy= 0.8376
Iter 500, Testing Accuracy= 0.9393, Training Accuracy= 0.9327
Iter 600, Testing Accuracy= 0.9509, Training Accuracy= 0.9468
Iter 700, Testing Accuracy= 0.9562, Training Accuracy= 0.953
Iter 800, Testing Accuracy= 0.9589, Training Accuracy= 0.9582
Iter 900, Testing Accuracy= 0.9624, Training Accuracy= 0.9584
Iter 1000, Testing Accuracy= 0.9633, Training Accuracy= 0.9617
```

程序运行完成之后会在当前程序路径下生成 logs 文件夹，logs 文件夹下会有：

![](http://p35l3ejfq.bkt.clouddn.com/20181012204117.png)

可视化网络训练过程：`tensorboard --logdir=logs目录的路径`

准确率：

![](http://p35l3ejfq.bkt.clouddn.com/20181012204758.png)

在 logs 文件夹下有两个子文件夹，对应着图中两条线，橙色对应测试集测出来的数据，蓝色对应训练集训练出来的数据，可以看到，两条线非常接近，代表模型没有欠拟合和过拟合现象。如果是过拟合情况，那么蓝色的线就会比较高，橙色的线就会比较低。

交叉熵：

![](http://p35l3ejfq.bkt.clouddn.com/20181012205455.png)

网络结构：

![](http://p35l3ejfq.bkt.clouddn.com/20181012205527.png)

fc2 内部：

![](http://p35l3ejfq.bkt.clouddn.com/20181012210012.png)