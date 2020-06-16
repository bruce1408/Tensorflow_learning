"""
    在测试的时候可以直接使用
    ***sess = tf.InteractiveSession()和这个语句输出结果.***
    1.0.0
    tensorflow name_scope, variable_scope 如何理解。因为如果使用Variable 的话每次都会新建变量，
    但是大多数时候我们是希望一些变量重用的，所以就用到了get_variable()。它会去搜索变量名，然后没有就新建，有就直接用。
    在name_scope下，如果get_variable命名相同，而且你没有共享，那么报错,但是Variable没有这个问题。
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
#
# with tf.name_scope('name_scope_x'):
#     var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
#     # var2 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32) # 这一句会报错，没有共享 注释掉了
#     var3 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
#     var4 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(var1.name, sess.run(var1))
#     # print(var2.name, sess.run(var2))
#     print(var3.name, sess.run(var3))
#     print(var4.name, sess.run(var4))


# 为了使用共享，就要用到variable_scope
# import tensorflow as tf
#
# with tf.variable_scope('name_scope_1') as scope:
#     var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
#     scope.reuse_variables()  # 设置共享变量
#     var1_reuse = tf.get_variable(name='var1')
#     var2 = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
#     var2_reuse = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(var1.name, sess.run(var1))
#     print(var1_reuse.name, sess.run(var1_reuse))
#     print(var2.name, sess.run(var2))
#     print(var2_reuse.name, sess.run(var2_reuse))


"""
1.0.1
loss 计算函数tf.nn.sparse_softmax_cross_entropy_with_logits函数是softmax 和 cross-entry相结合。
logits通过softmax算出来之后，取label对应的那个值，而不是logits 最大的那个值！
logits是mxn的矩阵，labels是标签，是一个[1xm]构成的行向量。m是样本数目:记忆方式是这里只看行数。
!! 如果你的样本是 样本num x 特征num。 你的logits是4x3的话，那么你的label肯定是1x4，4是样本数。如果label不是这样的形式，那么要加入
tf.argmax 函数来创建一个label矩阵才行,也就是说这里的tf.nn.sparse_softmax_函数的label是不能用one-hot编码的.
和稀疏编码不同的是# tf.softmax_cross_entropy_with_logits函数的label是one-hot编码的.
"""
import tensorflow as tf
import numpy as np

label2 = tf.convert_to_tensor([[0, 0, 1, 0]], dtype=tf.int64)
logit2 = tf.convert_to_tensor([[-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
# y3 = tf.argmax(y2, 1)
c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit2, labels=tf.argmax(label2, 1))  # 是以e为底数的对数
c2_ = tf.nn.softmax_cross_entropy_with_logits(logits=logit2, labels=label2)

label3 = tf.convert_to_tensor([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=tf.int64)
logit3 = tf.convert_to_tensor([[-2.6, -1.7, -3.2, 0.1], [-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
# y3_result = tf.argmax(y_3, 1)
y3_soft = tf.nn.softmax(logit3)
y3_label = tf.argmax(label3, 1)
c3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit3, labels=tf.argmax(label3, 1))  # label 创建

y4 = tf.convert_to_tensor([[0, 1, 0, 0]], dtype=tf.int64)
y_4 = tf.convert_to_tensor([[-2.6, -1.7, -3.2, 0.1]], dtype=tf.float32)
c4 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_4, labels=tf.argmax(y4, 1))
soft_result = tf.nn.softmax(logit2)

sess = tf.InteractiveSession()
label2 = tf.convert_to_tensor([[0, 0, 1, 0]], dtype=tf.int64)
logit2 = tf.convert_to_tensor([[-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
print("label 2 is:", sess.run(label2).shape)
print("logit2 2 is:", sess.run(logit2).shape)
# ==> (1,4)
# ==> (1,4)

y3 = tf.argmax(label2, 1)  # 最大的值的位置索引,返回的是[2]
print("y3 is:", sess.run(y3))
c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit2, labels=tf.argmax(label2, 1))

label3 = tf.convert_to_tensor([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=tf.int64)
logit3 = tf.convert_to_tensor([[-2.6, -1.7, -3.2, 0.1], [-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
y3_soft = tf.nn.softmax(logit3)
print("y3_soft", sess.run(y3_soft))
y3_label = tf.argmax(label3, 1)
c3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit3, labels=tf.argmax(label3, 1))  # label 创建
print("y3_label is:", sess.run(y3_label))

y4 = tf.convert_to_tensor([[0, 1, 0, 0]], dtype=tf.int64)
y_4 = tf.convert_to_tensor([[-2.6, -1.7, -3.2, 0.1]], dtype=tf.float32)
c4 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_4, labels=tf.argmax(y4, 1))
print("c4 is: ", c4.eval())
testa = np.arange(12).reshape([4, 3])
testinput = tf.convert_to_tensor(testa, dtype=tf.float32)
testb = np.array([0, 1, 0, 1])
testinputb = tf.convert_to_tensor(testb, dtype=tf.float32)
output = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=testinput, labels=testb)  # label 不用创建
print('output is: ', output.eval())
sess.close()
# with tf.Session() as sess:
#     # print(sess.run(y3_result))
#     print("c2:", sess.run([c2, c2_]))
#     print('c3: ', sess.run(c3))
#     print('y3_soft: \n', sess.run(y3_soft))
#     print('c4: ', sess.run(c4))
#     print(sess.run(soft_result))
#     print(sess.run(y3_label))
#     print('the output is:', sess.run(output))
# print('output is:', output.eval())

# ------------------------------------------#
# tf.softmax_cross_entropy_with_logits函数，这个函数和sparse不一样的是，他的真实labels需要one-hot编码才可以
# ------------------------------------------#
# our NN's output
# logits = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
# y = tf.nn.softmax(logits)  # y shape is 3x3
# print('y is:\n', y.eval())
# y_ = tf.constant([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # one-hot 编码的label
# cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_ * tf.log(y)))
# cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_))
# # 不管是交叉熵还是函数的结果都是一样的.
# print("cross_entropy2 is", cross_entropy2.eval())  # 如果是损失函数那么就应该是reduce_mean
# print("cross_entropy is:", cross_entropy.eval())

"""
1.0.2
第一个参数是预测值，[样本数x特征数] ,第二个参数是[1x样本数]，就是表示第i个样本是否在第几列。k是前k个数
tf.nn.in_top_k(prediction, target, K)
prediction就是表示你预测的结果，大小就是预测样本的数量乘以输出的维度，类型是tf.float32等。
K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。一般都是取1。

因为A张量里面的第一个元素的最大值的标签是0，第二个元素的最大值的标签是1.。但是实际的确是1和1,
所以输出就是False 和True。如果把K改成2，那么第一个元素的前面2个最大的元素的位置是0，1，第二个的就是1，2。
实际结果是1和1。包含在里面，所以输出结果就是True 和True.如果K的值大于张量A的列，那就表示输出结果都是true
"""
# input = tf.convert_to_tensor([[0.8, 0.6, 0.3], [0.1, 0.6, 0.4]], tf.float32)
# k = 1
# output = tf.nn.in_top_k(input, [1, 1], k)  # 每一行的最大值都在第3列（0为第一列）
# with tf.Session() as sess:
#     print('the input is: ', sess.run(input))
#     print('the output is: ', sess.run(output))


"""
1.0.3  tf.contrib.layers.embed_sequence 函数的使用方法
利用embed_sequence函数生成数据。
"""
# import tensorflow as tf
# import numpy as np
# features = [[1, 2, 3], [4, 5, 6]]
# input_data = tf.placeholder(tf.int32, [None, 3])
#
# # vocab_size 一定要大于这个id里面的最大的数
# outputs = tf.contrib.layers.embed_sequence(input_data, vocab_size=10, embed_dim=4)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     print(sess.run(outputs, feed_dict={input_data: features}))

"""
1.0.4 tf.cond 类似于if else 的功能
"""

# import tensorflow as tf
# import numpy as np
# x = tf.constant(3)
# y = tf.constant(4)
# a = np.arange(24).reshape(2, 3, 4)
# b = tf.convert_to_tensor(a, dtype=tf.float32)
# bb = tf.transpose(b, perm=[1, 0, 2])
# def get_index(i):
#     i = 0
#     return bb[i]
#
# batch_size = 3
# input_size = 4
# result = tf.cond(x > y, lambda: tf.zeros([batch_size, input_size], dtype=tf.float32), get_index)
# with tf.Session() as session:
#     print(result.eval())


"""
1.0.5 tf.TensorArray
"""
# import tensorflow as tf
# import numpy as np
# sess = tf.Session()
# x = np.arange(20)
# input_ta = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True)
# input_ta = input_ta.unstack(x)     #TensorArray可以传入array或者tensor
#
#
# for time in range(5):
#     input_ta = input_ta.write(time+len(x), time)
# a = input_ta.stack()
# print(sess.run(a))

"""
1.0.6 tf.reduce_all 如果存在维度的话，每个都要进行维度上的逻辑与&
"""
# import tensorflow as tf
# a = tf.constant([[True, True, False, False], [True, False, False, True]])
# z=tf.reduce_all(a)
# z2=tf.reduce_all(a, 0)
# z3=tf.reduce_all(a, 1)
# with tf.Session() as sess:
#     print(sess.run(z))

"""
1.0.7 tf.assign 函数赋值使用
"""
# x = tf.Variable(0)
# y = tf.assign(x, 1)  # 给x赋值为1, 同时给y赋值为1;
# z = x.assign(2)  # 给x赋值为2,同时给z赋值也是x
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print('the origin x is: ', sess.run(x))
#     print('the y is:', sess.run(y))
#     print('the x is: ', sess.run(x))
#     print('the y is:', sess.run(z))
#     print('the z is:', sess.run(x))

"""
1.0.8 tf.random_crop函数表示图片裁剪成给定的尺寸,裁剪位置是随机的,不是按照从中心裁剪而是任意裁剪;
tf.image.random_flip_left_right,表示把图片从左到右翻转,每次翻转的时候都会随机对图像进行
放大,缩小,中心点位置随机
"""
# import tensorflow as tf
# import matplotlib.image as img
# import matplotlib.pyplot as plt
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# sess = tf.InteractiveSession()
# image = img.imread('./imageProcess/picRecog/PicNormalized/dog.jpeg')
#
# reshaped_image = tf.cast(image, tf.float32)
# size = tf.cast(tf.shape(reshaped_image).eval(), tf.int32)
# height = sess.run(size[0] // 2)
# width = sess.run(size[1] // 2)
# distorted_image = tf.random_crop(reshaped_image, [224, 224, 3])
# flap_left_right = tf.image.random_flip_left_right(reshaped_image)
# print(tf.shape(reshaped_image).eval())
# print(tf.shape(distorted_image).eval())
# print(tf.shape(flap_left_right).eval())
#
# fig = plt.figure()
# fig1 = plt.figure()
# fig2 = plt.figure()
# ax = fig.add_subplot(111)
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax.imshow(tf.cast(reshaped_image, tf.uint8).eval())
# ax1.imshow(tf.cast(distorted_image, tf.uint8).eval())
# ax2.imshow(tf.cast(flap_left_right, tf.uint8).eval())
# plt.show()

"""
1.0.9 tf.control_dependencies 使用,保证其辖域中的操作必须是该函数传递的参数中
的操作完成后再进行.
"""
# import tensorflow as tf
# a_1 = tf.Variable(1)
# b_1 = tf.Variable(2)
# update_op = tf.assign(a_1, 10)
# add = tf.add(a_1, b_1)
#
# a_2 = tf.Variable(1)
# b_2 = tf.Variable(2)
# update_op = tf.assign(a_2, 10)
# with tf.control_dependencies([update_op]):
#     add_with_dependencies = tf.add(a_2, b_2)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     ans_1, ans_2 = sess.run([add, add_with_dependencies])
#     print("Add: ", ans_1)
#     print("Add_with_dependency: ", ans_2)

"""
1.10.0 tf.data.Dataset shuffle, batch, repeat 顺序问题
参考文章如下:
https://www.cnblogs.com/marsggbo/p/9603789.html
"""
# import tensorflow as tf
# dataset = tf.data.Dataset.range(10).shuffle(10).batch(6).repeat()
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(5):
#         value = sess.run(next_element)
#         print(value)

"""
1.10.1 使用tf.get_variable(name='a1', shape=[1]), 使用variable_scope()函数来给变量命名，主要是重用一些变量的情况下使用
"""
# import tensorflow as tf
#
# with tf.variable_scope('V1'):
#     a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
#
# with tf.variable_scope('V1', reuse=True):
#     a3 = tf.get_variable('a1')
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(a1.name)
#     print(sess.run(a1))
#     print(a3.name)
#     print(sess.run(a3))

"""
1.10.2 tf.pad 函数使用, 给tensor添加0, 
"""
# t = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape(2，3)
# paddings = tf.constant([[1, 1], [2, 2]])  # shape(2，2)
# c = tf.pad(t, paddings, "CONSTANT")  # shape(4, 7) = 2 + 1 + 1, 3 + 2 + 2
# with tf.Session() as sess:
#     a = sess.run(c)
#     print(a.shape)

"""
1.10.3  tf.assign()函数,赋值的时候需要复制被完成才可以成功赋值, 也就是需要对齐进行sess.run()这个运算符才可以
成功赋值;参考Tensorflow：tf.assign()函数的使用方法及易错点https://blog.csdn.net/Invokar/article/details/89041501
"""
# import tensorflow as tf
# sess = tf.InteractiveSession()

# 例子1, 赋值没有完成
# ref_a = tf.Variable(tf.constant(1))
# ref_b = tf.Variable(tf.constant(2))
# update = tf.assign(ref_a, 10)
# ref_sum = tf.add(ref_a, ref_b)
# sess.run(tf.global_variables_initializer())
# print(sess.run(ref_sum))  # 结果是3

# 例子2, 赋值完成
# ref_a = tf.Variable(tf.constant(1))
# ref_b = tf.Variable(tf.constant(2))
# update = tf.assign(ref_a, 10)
# ref_sum = tf.add(ref_a, ref_b)
# sess.run(tf.global_variables_initializer())
# sess.run(update)  # 唯一修改的地方
# print(sess.run(ref_sum))  # 结果是12

# 例子3, 另外可以通过参数直接赋值更新
# ref_a = tf.Variable(tf.constant(1))
# ref_b = tf.Variable(tf.constant(2))
# ref_a = tf.assign(ref_a, 10)
# ref_sum = tf.add(ref_a, ref_b)
#
# sess.run(tf.global_variables_initializer())
# print(sess.run(ref_sum))


# 例子4, 使用tf.control_dependencies()
# ref_a = tf.Variable(tf.constant(1))
# ref_b = tf.Variable(tf.constant(2))
# update = tf.assign(ref_a, 10)

# 这句话的意思就是必须先执行control_dependencies函数来更新其参数 update 操作完成之后再进行, 所以实际运行的时候,先update然后相加
# with tf.control_dependencies([update]):
#     ref_sum = tf.add(ref_a, ref_b)
#
# sess.run(tf.global_variables_initializer())
# print(sess.run(ref_sum))
# sess.close()

"""
1.10.4 关于tf.GraphKeys.UPDATE_OPS关于tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用。
关于在batch_norm中，即为更新mean和variance的操作,参考链接: https://blog.csdn.net/huitailangyz/article/details/85015611
"""

"""
1.10.5 使用tensorflow tf.layers.max_pooling2d这个函数padding='same'和'valid'来尝试
"""
import tensorflow as tf

x = tf.Variable(tf.random_normal([10, 28, 27, 3]))  # [batch_szie,height,weight,channel]

max_pool = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='same')
print(max_pool)
max_pool = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='valid')
print(max_pool)
