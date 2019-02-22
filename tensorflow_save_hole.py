"""
    1.0.0
    tensorflow name_scope, variable_scope 如何理解。因为如果使用Variable 的话每次都会新建变量，
    但是大多数时候我们是希望一些变量重用的，所以就用到了get_variable()。它会去搜索变量名，然后没有就新建，有就直接用。
    在name_scope下，如果get_variable命名相同，而且你没有共享，那么报错,但是Variable没有这个问题。
"""
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
tf.argmax 函数来创建一个label矩阵才行

"""
# import tensorflow as tf
# import numpy as np
# label2 = tf.convert_to_tensor([[0, 0, 1, 0]], dtype=tf.int64)
# logit2 = tf.convert_to_tensor([[-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
# # y3 = tf.argmax(y2, 1)
# c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit2, labels=tf.argmax(label2, 1))
#
# label3 = tf.convert_to_tensor([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=tf.int64)
# logit3 = tf.convert_to_tensor([[-2.6, -1.7, -3.2, 0.1], [-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
# # y3_result = tf.argmax(y_3, 1)
# y3_soft = tf.nn.softmax(logit3)
# y3_label = tf.argmax(label3, 1)
# c3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit3, labels=tf.argmax(label3, 1))  # label 创建
# y4 = tf.convert_to_tensor([[0, 1, 0, 0]], dtype=tf.int64)
# y_4 = tf.convert_to_tensor([[-2.6, -1.7, -3.2, 0.1]], dtype=tf.float32)
# c4 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_4, labels=tf.argmax(y4, 1))
# soft_result = tf.nn.softmax(logit2)
#
# testa = np.arange(12).reshape([4, 3])
# testinput = tf.convert_to_tensor(testa, dtype=tf.float32)
# testb = np.array([0, 1, 0, 1])
# testinputb = tf.convert_to_tensor(testb, dtype=tf.float32)
# output = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=testinput, labels=testb)  # label 不用创建
#
# with tf.Session() as sess:
#     # print(sess.run(y3_result))
#     print('c3: ', sess.run(c3))
#     print('y3_soft: \n', sess.run(y3_soft))
#     print('c4: ', sess.run(c4))
#     print(sess.run(soft_result))
#     print(sess.run(y3_label))
#     print('the output is:', sess.run(output))
"""
1.0.2
第一个参数是预测值，[样本数x特征数] ,第二个参数是[1x样本数]，就是表示第i个样本是否在第几列。k是前k个数

"""
# input = tf.constant(np.random.rand(3, 4), tf.float32)
# k = 1
# output = tf.nn.in_top_k(input, [3, 3, 3], k)  # 每一行的最大值都在第3列（0为第一列）
# with tf.Session() as sess:
#     print(sess.run(input))
#     print(sess.run(output))
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
import tensorflow as tf
a = tf.constant([[True, True, False, False], [True, False, False, True]])
z=tf.reduce_all(a)
z2=tf.reduce_all(a, 0)
z3=tf.reduce_all(a, 1)
with tf.Session() as sess:
    print(sess.run(z))

