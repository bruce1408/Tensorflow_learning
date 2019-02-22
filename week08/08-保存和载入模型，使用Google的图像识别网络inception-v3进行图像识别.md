## 保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别

### 一、保存和载入模型

#### 1、保存模型

可以使用：
/Users/bruce/programme/Python/datasets/MNIST_data/Users/bruce/programme/Python/datasets/MNIST_data
``` python
saver = tf.train.Saver()
saver.save() 
```

来保存模型。

完整代码如下：（对应代码：`8-1saver_save.py`）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次100张照片
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
    # 保存模型
    saver.save(sess, 'net/my_net.ckpt')
```

上面定义了一个 saver：

``` python
saver = tf.train.Saver()
```

训练结束了使用：

``` python
saver.save(sess, 'net/my_net.ckpt')
```

将训练好的模型保存在 net/my_net.ckpt 文件中。

训练过程如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter 0,Testing Accuracy 0.8239
Iter 1,Testing Accuracy 0.8893
Iter 2,Testing Accuracy 0.9001
Iter 3,Testing Accuracy 0.9051
Iter 4,Testing Accuracy 0.9081
Iter 5,Testing Accuracy 0.9094
Iter 6,Testing Accuracy 0.9112
Iter 7,Testing Accuracy 0.9132
Iter 8,Testing Accuracy 0.9142
Iter 9,Testing Accuracy 0.9158
Iter 10,Testing Accuracy 0.9171
```

最后 net 目录下有如下文件：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-10/50408346.jpg)

#### 2、载入模型

可以使用该方式来调用一个训练好的模型：

``` python
saver = tf.train.Saver()
saver.restore()
```

案例完整代码如下：（对应代码：`8-2saver_restore.py`）

``` python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次100张照片
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    saver.restore(sess,'net/my_net.ckpt')
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
```

测试结果如下：

``` xml
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
0.098
0.9166
```

如上使用了：

``` python
saver = tf.train.Saver()
saver.restore(sess,'net/my_net.ckpt')
```

来调用上节训练好的手写数字识别模型。代码做了个测试，一开始直接将测试集送往没有训练好的网络，得到的测试结果是 0.098，然后调用训练好的网络，测试结果为 0.9166。

### 二、使用Google的图像识别网络inception-v3进行图像识别

先了解下 inception 网络模型，参考博客：

- [TensorFlow学习笔记：使用Inception v3进行图像分类](https://www.jianshu.com/p/cc830a6ed54b)
- [Google Inception Net介绍及Inception V3结构分析](https://blog.csdn.net/weixin_39881922/article/details/80346070)
- [深入浅出——网络模型中Inception的作用与结构全解析](https://blog.csdn.net/u010402786/article/details/52433324)
- [tensorflow+inceptionv3图像分类网络结构的解析与代码实现【附下载】](https://blog.csdn.net/k87974/article/details/80221215)
- ......

#### 1、下载inception-v3网络模型

（对应代码：`8-3下载google图像识别网络inception-v3并查看结构.py`）

``` py
# coding: utf-8

import tensorflow as tf
import os
import tarfile
import requests

# inception模型下载地址
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# 模型存放地址
inception_pretrain_model_dir = "inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

# 获取文件名，以及文件路径
filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

# 下载模型
if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish: ", filename)
# 解压文件
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

# 模型结构存放文件
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classify_image_graph_def.pb为google训练好的模型
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    # 创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()
```

在 Jupyter Notebook 中运行代码后显示：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-10/50884014.jpg)

然后在相应目录会出现如下两个文件夹：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-10/14887577.jpg)

其中，inception_log 文件夹保存模型的结构：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-10/84328307.jpg)

inception_model 文件夹下是保存的训练结果：（其他文件其实都是`inception-2015-12-05.tgz`文件解压后的）

![](http://p35l3ejfq.bkt.clouddn.com/18-10-10/79742604.jpg)

其中，`classify_image_graph_def.pb`是已经训练过的 inception-v3 的模型。

#### 2、使用inception-v3网络模型进行图像识别

我们先打开 inception_model 文件夹下的 `imagenet_2012_challenge_label_map_proto.pbtxt` 和 `imagenet_synset_to_human_label_map.txt` 看看。

两个文件内容如下：

![](http://p35l3ejfq.bkt.clouddn.com/18-10-10/98506442.jpg)

简单说明：左侧文件中 target_class 后面的数字代表目标的分类，数值为 1——1000（inception 模型是用来做 1000 个分类的），target_class_string 后面的字符串值对应到右侧文件的第一列，右侧文件的第二列表示对第一列的描述，相当是对分类的描述，从而来表示属于哪一类。

在运行代码之前，先在在当前程序路径下新建 images 文件夹，网上找几张图片保存在 images 下。

完整代码如下：（对应代码：`8-4使用inception-v3做各种图像的识别.py`）

``` python
# coding: utf-8

import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

class NodeLookup(object):
    def __init__(self):  
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'   
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n********对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        #一行一行读取数据
        for line in proto_as_ascii_lines :
            #去掉换行符
            line=line.strip('\n')
            #按照'\t'分割
            parsed_items = line.split('\t')
            #获取分类编号
            uid = parsed_items[0]
            #获取分类名称
            human_string = parsed_items[1]
            #保存编号字符串n********与分类名称映射关系
            uid_to_human[uid] = human_string  # n00004475->organism, being

        # 加载分类字符串n********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                #获取分类编号1-1000
                target_class = int(line.split(': ')[1])  # target_class: 449
            if line.startswith('  target_class_string:'):
                #获取编号字符串n********
                target_class_string = line.split(': ')[1]  # target_class_string: "n01440764"
                #保存分类编号1-1000与编号字符串n********映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]  # 449->n01440764

        #建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            #获取分类名称
            name = uid_to_human[val]
            #建立分类编号1-1000到分类名称的映射关系
            node_id_to_name[key] = name  # 449->organism, being
        return node_id_to_name

    #传入分类编号1-1000返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]
```



``` python
#创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
```



``` python
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录
    for root,dirs,files in os.walk('images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
            predictions = np.squeeze(predictions)#把结果转为1维数据

            #打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)
            #显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            #排序
            top_k = predictions.argsort()[-5:][::-1]
            print('top_k:', top_k)
            node_lookup = NodeLookup()
            for node_id in top_k:     
                #获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
```

代码中，程序的头读取了两个文件：

``` xml
    label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'   
    uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
```

代码中，类  `NodeLookup` 的目的就是建立两个文件之间的关系，将`imagenet_2012_challenge_label_map_proto.pbtxt`中的 target_class 对应于`imagenet_synset_to_human_label_map.txt`中的类。

最后的排序代码解释下：

``` python
			#排序
            top_k = predictions.argsort()[-5:][::-1]
            print('top_k:', top_k)
            node_lookup = NodeLookup()
            for node_id in top_k:     
                #获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
```

因为概率从小到大排序，所以如上第一行代码表示从倒数第 5 的位置开始取至倒数第 1 的位置，从而得到概率顺序从小到大的前 5 的概率值，再对这 5 个值做个倒序，进而得到从大到小的 5 个概率值。

最后的运行结果如下：

``` xml
images/lion.jpg
```

![](http://p35l3ejfq.bkt.clouddn.com/18-10-11/34173500.jpg)

``` xml
top_k: [190  11 206  85  30]
lion, king of beasts, Panthera leo (score = 0.96306)
cougar, puma, catamount, mountain lion, painter, panther, Felis concolor (score = 0.00161)
cheetah, chetah, Acinonyx jubatus (score = 0.00079)
leopard, Panthera pardus (score = 0.00057)
jaguar, panther, Panthera onca, Felis onca (score = 0.00033)
```

``` xml
images/panda.jpg
```

![](http://p35l3ejfq.bkt.clouddn.com/18-10-11/5249455.jpg)

``` xml
top_k: [169   7 222 374 878]
giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.96960)
lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00078)
soccer ball (score = 0.00067)
lawn mower, mower (score = 0.00065)
earthstar (score = 0.00040)
```

``` xml
images/rabbit.jpg
```

![](http://p35l3ejfq.bkt.clouddn.com/18-10-11/48396384.jpg)

``` xml
top_k: [164 840 129 950 188]
Angora, Angora rabbit (score = 0.36784)
hamper (score = 0.17425)
hare (score = 0.13834)
shopping basket (score = 0.10668)
wood rabbit, cottontail, cottontail rabbit (score = 0.04976)
```