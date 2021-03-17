import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='1'
"""
原始训练部分模型案例
"""
# w1 = tf.Variable(tf.constant(2.0, shape=[1]), name='w1')
# w2 = tf.Variable(tf.constant(3.0, shape=[1]), name='w2')
#
# a = tf.placeholder('float', name='a')
# b = tf.placeholder('float', name='b')
# r1 = tf.multiply(w1, a)
# r2 = tf.multiply(w2, b)
# y = tf.add(r1, r2, name='final_op_add')
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()  # 默认是保存所有的变量
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(a)
#     print(b)
#     print(y)
#     print(sess.run(y, feed_dict={a: 10.0, b: 10.0}))
#     saver.save(sess, './8_5basic/model.ckpt')
"""
原来保存的参数w1 和 w2 分别是2,3，这里的4和5分别是为了构建图结构而定义的,
所以在最后使用的时候并不使用4,5，而是使用保存的2和3来做。
这种方法很不方便，因为在训练的时候已经定义了模型的图结构，所以在训练的时候不想重新
定义图结构，希望能够去读取一个文件然后可以直接使用。
"""
# w1 = tf.Variable(tf.constant(4.0, shape=[1]), name='w1')
# w2 = tf.Variable(tf.constant(5.0, shape=[1]), name='w2')
# result = 10 * w1 + 10 * w2
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, '8_5basic/model.ckpt')
#     print('the result is', sess.run(result))
#     print('the w1 is:', sess.run(w1))

"""
不再重新定义图结构，而是使用保存模型的图结构即可，使用tf.train.import_meta_graph(checkpoint.meta 地址即可)
仅仅加载图是没用的，还需要使用前面的模型参数，比如本例子中的w1:0这个参数
"""
# saver = tf.train.import_meta_graph('./8_5basic/model.ckpt.meta')
# with tf.Session() as sess:
#     saver.restore(sess, '8_5basic/model.ckpt')
#     print(sess.run(tf.get_default_graph().get_tensor_by_name('w1:0')))  # 不管是w1还是w2都是可以打印出来的
#     print(sess.run("w2:0"))

"""
不再重新构建图，同时加载运算
"""
saver = tf.train.import_meta_graph('./8_5basic/model.ckpt.meta')
with tf.Session() as sess:
    saver.restore(sess, '8_5basic/model.ckpt')
    graph = tf.get_default_graph()
    a = graph.get_tensor_by_name("a:0")
    b = graph.get_tensor_by_name("b:0")
    feed_dict = {a: 12.0, b: 12.0}
    # for x, y in enumerate(range(9)):
    #     feed_dict['a'] = x
    #     feed_dict['b'] = y
    add_op = graph.get_tensor_by_name("final_op_add:0")
    print(sess.run(add_op, feed_dict))

"""
多个输入进行预测
"""
# saver = tf.train.import_meta_graph('./8_5basic/model.ckpt.meta')
# sess = tf.Session()
# saver.restore(sess, '8_5basic/model.ckpt')
# graph = tf.get_default_graph()
# a = graph.get_tensor_by_name("a:0")
# b = graph.get_tensor_by_name("b:0")
# feed_dict = {a: 12.0, b: 12.0}
# print(feed_dict)
# for x, y in enumerate(range(9)):
#     feed_dict = {a: x, b: y}
#     add_op = graph.get_tensor_by_name("final_op_add:0")
#     print(sess.run(add_op, feed_dict))
