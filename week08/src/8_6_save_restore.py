import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sess = tf.Session()
# 先加载图和变量
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

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


"""
加载模型部分
"""
# saver = tf.train.import_meta_graph('vgg.meta')
# # 访问图
# graph = tf.get_default_graph()
#
# # 访问用于fine-tuning的output
# fc7 = graph.get_tensor_by_name('fc7:0')
#
# # 如果你想修改最后一层梯度，需要如下
# fc7 = tf.stop_gradient(fc7)  # It's an identity function
# fc7_shape = fc7.get_shape().as_list()
#
# new_outputs = 2
# weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
# biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
# output = tf.matmul(fc7, weights) + biases
# pred = tf.nn.softmax(output)

# Now, you run this with fine-tuning data in sess.run()
