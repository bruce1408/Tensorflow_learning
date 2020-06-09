# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.contrib.slim import nets
from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""
查看恢复的模型参数
f.trainable_variables()查看的是所有可训练的变量；
tf.global_variables()获得的与tf.trainable_variables()类似，只是多了一些非trainable的变量，
比如定义时指定为trainable=False的变量；
sess.graph.get_operations()则可以获得几乎所有的operations相关的tensor
"""


def getLabel():
    """
    获取iamgenet图像类别标签对应的数字
    :return:
    """
    labelDict = dict()
    with open('./label1000.txt', 'r') as f:
        for eachline in f:
            eachline = eachline.strip()
            labelDict[eachline.split(' ')[0].strip(":")] = eachline.split(' ')[1:]
    return labelDict


def tensorName():
    var = tf.global_variables()
    # name = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node] # 包含所有的节点
    var_to_restore = [val for val in var if 'fc8' not in val.name]  # 保留变量名中不含有fc8的变量
    print(var_to_restore)
    for val in var:
        print('the global variable: ', val.name)
    tra = tf.trainable_variables()
    for t in tra:
        print('the trainable variable is:', t.name)
    # 列出图里所有的tensor名，以便决定获取哪层作为特征
    train_layers = ['fc8', 'fc7', 'fc6']
    # var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers]
    for i in var_list:
        print('the name is: ', i.name)


# 读取图片方式
# img = Image.open("./images/lion.jpg")
# img = img.resize((224, 224))
# img = np.array(img)
# img = np.expand_dims(img, 0)
# img = img.astype(np.float32)


# result, endpoint = vgg.vgg_16(img)
# arg_scope = vgg.vgg_arg_scope()
# label = tf.argmax(result, 1)

# with tf.Session() as sess:
#     restorer = tf.train.Saver()
#     print("model restore")
#     restorer.restore(sess, './vgg_16.ckpt')
#     numlabel = sess.run(label)
#     print(labelDict[str(numlabel[0])])
#     print(endpoint)


# 定义网络变量，net为一个tensor变量，endpoints为一个词典，记录了网络的各层组成
X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='X')
net, endpoints = nets.vgg.vgg_16(inputs=X, is_training=False)  # FIXME：修改网络结构
feat = tf.get_default_graph().get_tensor_by_name('vgg_16/fc8/squeezed:0')
label = tf.argmax(feat, 1)
# 读取真实图片

img = Image.open("./images/panda.jpg")  # FIXME：修改图片路径
img = img.resize((224, 224))
img = np.array(img)
img = np.expand_dims(img, 0)
img = img.astype(np.float32)
labels = getLabel()

# 开始提取特征！
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint_path = './vgg_16.ckpt'  # FIXME：修改checkpoint路径
    restorer = tf.train.Saver()
    restorer.restore(sess, checkpoint_path)

    # 将真实数据送入会话，输出特征
    feat_ = sess.run(label, feed_dict={X: img})
    print('feature:', feat_)  # 输出一个(1,1000)的numpy特征
    print(labels[str(feat_[0])])

# """
# Created on Wed Jun  6 11:56:58 2018
#
# @author: zy
# """
#
# '''
# 利用已经训练好的vgg16网络对flowers数据集进行微调
# 把最后一层分类由2000->5 然后重新训练，我们也可以冻结其它所有层，只训练最后一层
# '''
#
# from tensorflow.contrib.slim.nets import vgg
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np
# import input_data
# import os
#
# slim = tf.contrib.slim
#
# DATA_DIR = '/raid/bruce/datasets/flower_photos'
# # 输出类别
# NUM_CLASSES = 5
#
# # 获取图片大小
# IMAGE_SIZE = vgg.vgg_16.default_image_size
#
#
# def flowers_fine_tuning():
#     """
#     演示一个VGG16的例子
#     微调 这里只调整VGG16最后一层全连接层，把1000类改为5类
#     对网络进行训练
#     """
#
#     '''
#     1.设置参数，并加载数据
#     '''
#     # 用于保存微调后的检查点文件和日志文件路径
#     train_log_dir = './log/vgg16/fine_tune'
#     train_log_file = 'flowers_fine_tune.ckpt'
#
#     # 官方下载的检查点文件路径
#     checkpoint_file = './log/vgg16/vgg_16.ckpt'
#
#     # 设置batch_size
#     batch_size = 256
#
#     learning_rate = 1e-4
#
#     # 训练集数据长度
#     n_train = 3320
#     # 测试集数据长度
#     # n_test = 350
#     # 迭代轮数
#     training_epochs = 3
#
#     display_epoch = 1
#
#     if not tf.gfile.Exists(train_log_dir):
#         tf.gfile.MakeDirs(train_log_dir)
#
#     # 加载数据
#     train_images, train_labels = input_data.get_batch_images_and_label(DATA_DIR, batch_size, NUM_CLASSES, True,
#                                                                        IMAGE_SIZE, IMAGE_SIZE)
#     test_images, test_labels = input_data.get_batch_images_and_label(DATA_DIR, batch_size, NUM_CLASSES, False,
#                                                                      IMAGE_SIZE, IMAGE_SIZE)
#
#     # 获取模型参数的命名空间
#     arg_scope = vgg.vgg_arg_scope()

# {'<function convolution2d at 0x7f0dec6faf28>': {'activation_fn': <function relu at 0x7f0e1962c048>,
# 'weights_regularizer': <function l2_regularizer.<locals>.l2 at 0x7f0e936876a8>, 'biases_initializer':
# <tensorflow.python.ops.init_ops.Zeros object at 0x7f0e745c0400>, 'padding': 'SAME'},
# '<function fully_connected at 0x7f0dec6fcea0>': {'activation_fn': <function relu at 0x7f0e1962c048>,
# 'weights_regularizer': <function l2_regularizer.<locals>.l2 at 0x7f0e936876a8>, 'biases_initializer':
# <tensorflow.python.ops.init_ops.Zeros object at 0x7f0e745c0400>}}

#
#     # 创建网络
#     with slim.arg_scope(arg_scope):
#
#         '''
#         2.定义占位符和网络结构
#         '''
#         # 输入图片
#         input_images = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
#         # 图片标签
#         input_labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])
#         # 训练还是测试？测试的时候弃权参数会设置为1.0
#         is_training = tf.placeholder(dtype=tf.bool)
#
#         # 创建vgg16网络  如果想冻结所有层，可以指定slim.conv2d中的 trainable=False
#         logits, end_points = vgg.vgg_16(input_images, is_training=is_training, num_classes=NUM_CLASSES)
#         # print(end_points)  每个元素都是以vgg_16/xx命名
#
#         '''
#         #从当前图中搜索指定scope的变量，然后从检查点文件中恢复这些变量(即vgg_16网络中定义的部分变量)
#         #如果指定了恢复检查点文件中不存在的变量，则会报错 如果不知道检查点文件有哪些变量，我们可以打印检查点文件查看变量名
#         params = []
#         conv1 = slim.get_variables(scope="vgg_16/conv1")
#         params.extend(conv1)
#         conv2 = slim.get_variables(scope="vgg_16/conv2")
#         params.extend(conv2)
#         conv3 = slim.get_variables(scope="vgg_16/conv3")
#         params.extend(conv3)
#         conv4 = slim.get_variables(scope="vgg_16/conv4")
#         params.extend(conv4)
#         conv5 = slim.get_variables(scope="vgg_16/conv5")
#         params.extend(conv5)
#         fc6 = slim.get_variables(scope="vgg_16/fc6")
#         params.extend(fc6)
#         fc7 = slim.get_variables(scope="vgg_16/fc7")
#         params.extend(fc7)
#         '''
#
#         # Restore only the convolutional layers: 从检查点载入当前图除了fc8层之外所有变量的参数
#         params = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
#         # 用于恢复模型  如果使用这个保存或者恢复的话，只会保存或者恢复指定的变量
#         restorer = tf.train.Saver(params)
#
#         # 预测标签
#         pred = tf.argmax(logits, axis=1)
#
#         """
#         3 定义代价函数和优化器
#         """
#         # 代价函数
#         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits))
#
#         # 设置优化器
#         optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#
#         # 预测结果评估
#         correct = tf.equal(pred, tf.argmax(input_labels, 1))  # 返回一个数组 表示统计预测正确或者错误
#         accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 求准确率
#
#         num_batch = int(np.ceil(n_train / batch_size))
#
#         # 用于保存检查点文件
#         save = tf.train.Saver(max_to_keep=1)
#
#         # 恢复模型
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#
#             # 检查最近的检查点文件
#             ckpt = tf.train.latest_checkpoint(train_log_dir)
#             if ckpt != None:
#                 save.restore(sess, ckpt)
#                 print('从上次训练保存后的模型继续训练！')
#             else:
#                 restorer.restore(sess, checkpoint_file)
#                 print('从官方模型加载训练！')
#
#             # 创建一个协调器，管理线程
#             coord = tf.train.Coordinator()
#
#             # 启动QueueRunner, 此时文件名才开始进队。
#             threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#             '''
#             4 查看预处理之后的图片
#             '''
#             imgs, labs = sess.run([train_images, train_labels])
#             print('原始训练图片信息：', imgs.shape, labs.shape)
#             show_img = np.array(imgs[0], dtype=np.uint8)
#             plt.imshow(show_img)
#             plt.title('Original train image')
#             plt.show()
#
#             imgs, labs = sess.run([test_images, test_labels])
#             print('原始测试图片信息：', imgs.shape, labs.shape)
#             show_img = np.array(imgs[0], dtype=np.uint8)
#             plt.imshow(show_img)
#             plt.title('Original test image')
#             plt.show()
#
#             print('开始训练！')
#             for epoch in range(training_epochs):
#                 total_cost = 0.0
#                 for i in range(num_batch):
#                     imgs, labs = sess.run([train_images, train_labels])
#                     _, loss = sess.run([optimizer, cost],
#                                        feed_dict={input_images: imgs, input_labels: labs, is_training: True})
#                     total_cost += loss
#
#                 # 打印信息
#                 if epoch % display_epoch == 0:
#                     print('Epoch {}/{}  average cost {:.9f}'.format(epoch + 1, training_epochs, total_cost / num_batch))
#
#                 # 进行预测处理
#                 imgs, labs = sess.run([test_images, test_labels])
#                 cost_values, accuracy_value = sess.run([cost, accuracy],
#                                                        feed_dict={input_images: imgs, input_labels: labs,
#                                                                   is_training: False})
#                 print('Epoch {}/{}  Test cost {:.9f}'.format(epoch + 1, training_epochs, cost_values))
#                 print('准确率:', accuracy_value)
#
#                 # 保存模型
#                 save.save(sess, os.path.join(train_log_dir, train_log_file), global_step=epoch)
#                 print('Epoch {}/{}  模型保存成功'.format(epoch + 1, training_epochs))
#
#             print('训练完成')
#
#             # 终止线程
#             coord.request_stop()
#             coord.join(threads)
#
#
# def flowers_test():
#     """
#     使用微调好的网络进行测试
#     """
#     '''
#     1.设置参数，并加载数据
#     '''
#     # 微调后的检查点文件和日志文件路径
#     save_dir = './log/vgg16/fine_tune'
#
#     # 设置batch_size
#     batch_size = 128
#
#     # 加载数据
#     train_images, train_labels = input_data.get_batch_images_and_label(DATA_DIR, batch_size, NUM_CLASSES, True,
#                                                                        IMAGE_SIZE, IMAGE_SIZE)
#     test_images, test_labels = input_data.get_batch_images_and_label(DATA_DIR, batch_size, NUM_CLASSES, False,
#                                                                      IMAGE_SIZE, IMAGE_SIZE)
#
#     # 获取模型参数的命名空间
#     arg_scope = vgg.vgg_arg_scope()
#
#     # 创建网络
#     with slim.arg_scope(arg_scope):
#         '''
#         2.定义占位符和网络结构
#         '''
#         # 输入图片
#         input_images = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
#         # 训练还是测试？测试的时候弃权参数会设置为1.0
#         is_training = tf.placeholder(dtype=tf.bool)
#
#         # 创建vgg16网络
#         logits, end_points = vgg.vgg_16(input_images, is_training=is_training, num_classes=NUM_CLASSES)
#
#         # 预测标签
#         pred = tf.argmax(logits, axis=1)
#
#         restorer = tf.train.Saver()
#
#         # 恢复模型
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             ckpt = tf.train.latest_checkpoint(save_dir)
#             if ckpt != None:
#                 # 恢复模型
#                 restorer.restore(sess, ckpt)
#                 print("Model restored.")
#
#             # 创建一个协调器，管理线程
#             coord = tf.train.Coordinator()
#
#             # 启动QueueRunner, 此时文件名才开始进队。
#             threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#             '''
#             查看预处理之后的图片
#             '''
#             imgs, labs = sess.run([test_images, test_labels])
#             print('原始测试图片信息：', imgs.shape, labs.shape)
#             show_img = np.array(imgs[0], dtype=np.uint8)
#             plt.imshow(show_img)
#             plt.title('Original test image')
#             plt.show()
#
#             pred_value = sess.run(pred, feed_dict={input_images: imgs, is_training: False})
#             print('预测结果为：', pred_value)
#             print('实际结果为：', np.argmax(labs, 1))
#             correct = np.equal(pred_value, np.argmax(labs, 1))
#             print('准确率为:', np.mean(correct))
#
#             # 终止线程
#             coord.request_stop()
#             coord.join(threads)
#

# if __name__ == '__main__':
#     tf.reset_default_graph()
#     flowers_fine_tuning()
#     flowers_test()
