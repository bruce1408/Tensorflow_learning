import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# input_data.read_data_sets函数生成的类会自动将MNIST数据集划分为train, validation和test三个数据集
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)

batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.99
max_steps = 30000


# 输入网络的尺寸为32×32×1
def hidden_layer(input_tensor, regularizer, avg_class, resuse):
    # 创建第一个卷积层，得到特征图大小为32@28x28
    # 这行代码指定了第一个卷积层作用域为C1-conv，在这个作用域下有两个变量conv1_weights和conv1_biases
    with tf.variable_scope("C1-conv", reuse=resuse):
        # tf.get_variable共享变量
        # [5, 5, 1, 32]卷积核大小为5×5×1，有32个
        # stddev正太分布的标准差
        conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        # tf.constant_initializer初始化为常数，这个非常有用，通常偏置项就是用它初始化的
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        # strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
        # padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 创建第一个池化层，池化后的结果为32@14x14
    # tf.name_scope的主要目的是为了更加方便地管理参数命名。
    # 与 tf.Variable() 结合使用。简化了命名
    with tf.name_scope("S2-max_pool", ):
        # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
        # 因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        # strides：窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 创建第二个卷积层，得到特征图大小为64@14x14。注意，第一个池化层之后得到了32个
    # 特征图，所以这里设输入的深度为32，我们在这一层选择的卷积核数量为64，所以输出
    # 的深度是64，也就是说有64个特征图
    with tf.variable_scope("C3-conv", reuse=resuse):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 创建第二个池化层，池化后结果为64@7x7
    with tf.name_scope("S4-max_pool", ):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # get_shape()函数可以得到这一层维度信息，由于每一层网络的输入输出都是一个batch的矩阵，
        # 所以通过get_shape()函数得到的维度信息会包含这个batch中数据的个数信息
        # shape[1]是长度方向，shape[2]是宽度方向，shape[3]是深度方向
        # shape[0]是一个batch中数据的个数，reshape()函数原型reshape(tensor,shape,name)
        shape = pool2.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]  # nodes=3136
        reshaped = tf.reshape(pool2, [shape[0], nodes])

    # 创建第一个全连层
    with tf.variable_scope("layer5-full1", reuse=resuse):
        Full_connection1_weights = tf.get_variable("weight", [nodes, 512],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        # if regularizer != None:
        tf.add_to_collection("losses", regularizer(Full_connection1_weights))
        Full_connection1_biases = tf.get_variable("bias", [512],
                                                  initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            Full_1 = tf.nn.relu(tf.matmul(reshaped, Full_connection1_weights) + \
                                Full_connection1_biases)
        else:
            Full_1 = tf.nn.relu(tf.matmul(reshaped, avg_class.average(Full_connection1_weights))
                                + avg_class.average(Full_connection1_biases))

    # 创建第二个全连层
    with tf.variable_scope("layer6-full2", reuse=resuse):
        Full_connection2_weights = tf.get_variable("weight", [512, 10],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        # if regularizer != None:
        tf.add_to_collection("losses", regularizer(Full_connection2_weights))
        Full_connection2_biases = tf.get_variable("bias", [10],
                                                  initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            result = tf.matmul(Full_1, Full_connection2_weights) + Full_connection2_biases
        else:
            result = tf.matmul(Full_1, avg_class.average(Full_connection2_weights)) + \
                     avg_class.average(Full_connection2_biases)
    return result


# tf.placeholder(dtype, shape=None, name=None)
x = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="x-input")
y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")

# L2正则化是一种减少过拟合的方法
regularizer = tf.contrib.layers.l2_regularizer(0.0001)

# 调用定义的CNN的函数
y = hidden_layer(x, regularizer, avg_class=None, resuse=False)
# 定义存储训练轮数的变量
training_step = tf.Variable(0, trainable=False)
# tf.train.ExponentialMovingAverage是指数加权平均的求法
# 可以加快训练早期变量的更新速度。
variable_averages = tf.train.ExponentialMovingAverage(0.99, training_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())

average_y = hidden_layer(x, regularizer, variable_averages, resuse=True)

# 使用交叉熵作为损失函数。这里使用
# sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。因为手写体是一个长度为
# 10的一维数组，而该函数需要提供的是一个正确答案的数字，所以需要使用tf.argmax函数来
# 得到正确答案对应的类别编号
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
# 计算在当前batch中所有样例的交叉熵平均值
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 总损失等于交叉熵损失和正则化损失的和
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
# 设置指数衰减的学习率
learning_rate = tf.train.exponential_decay(learning_rate,  # 基础的学习率，随着迭代的进行，更新变量时使用的学习率
                                           training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay, staircase=True)

# 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
train_step = tf.train.GradientDescentOptimizer(learning_rate). \
    minimize(loss, global_step=training_step)

with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')
crorent_predicition = tf.equal(tf.arg_max(average_y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(crorent_predicition, tf.float32))

# 初始化会话并开始训练过程
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(max_steps):
        if i % 1000 == 0:
            x_val, y_val = mnist.validation.next_batch(batch_size)
            reshaped_x2 = np.reshape(x_val, (batch_size, 28, 28, 1))
            validate_feed = {x: reshaped_x2, y_: y_val}

            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d trainging step(s) ,validation accuracy"
                  "using average model is %g%%" % (i, validate_accuracy * 100))

        x_train, y_train = mnist.train.next_batch(batch_size)

        reshaped_xs = np.reshape(x_train, (batch_size, 28, 28, 1))
        sess.run(train_op, feed_dict={x: reshaped_xs, y_: y_train})
