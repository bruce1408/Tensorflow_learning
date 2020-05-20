import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# w1 = tf.placeholder("float", name="w1")
# w2 = tf.placeholder("float", name="w2")
# b1 = tf.Variable(2.0, name="bias")
#
# # 定义一个op，用于后面恢复
# w3 = tf.add(w1, w2)
# w4 = tf.multiply(w3, b1, name="op_to_restore")
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# # 创建一个Saver对象，用于保存所有变量
# saver = tf.train.Saver()
#
# # 通过传入数据，执行op
# print(sess.run(w4, feed_dict={w1: 4, w2: 8}))
# # 打印 24.0 ==>(w1+w2)*b1
#
# # 现在保存模型
# saver.save(sess, './checkpoint_dir/MyModel', global_step=1000)



"""
加载模型部分
"""
sess = tf.Session()
# 先加载图和变量
saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict = {w1: 13.0, w2: 17.0}

# 接下来，访问你想要执行的op
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# 在当前图中能够加入op
add_on_op = tf.multiply(op_to_restore, 2)

print(sess.run(add_on_op, feed_dict))
# 打印120.0==>(13+17)*2*2
