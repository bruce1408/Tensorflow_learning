import os
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# 1、创建数据，使用dataset api
train_data = np.random.randn(100, 5)
test_data = np.random.randn(20, 5)
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

# 创建一个reinitializable iterator
re_iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types, train_dataset.output_shapes)
next_element = re_iterator.get_next()
train_init_op = re_iterator.make_initializer(train_dataset)
test_init_op = re_iterator.make_initializer(test_dataset)

with tf.Session() as sess:
    # 训练
    n_epochs = 2
    for i in range(n_epochs):
        sess.run(train_init_op)
        for j in range(100):
            print(sess.run(next_element).shape)
    # 测试
    sess.run(test_init_op)
    for i in range(20):
        print(sess.run(next_element).shape)
print("hello")

# 2、创建数据，使用dataset api
# train_data = np.random.randn(100, 5)
# val_data = np.random.randn(20, 5)

# n_epochs = 20
# train_dataset = tf.data.Dataset.from_tensor_slices(train_data).repeat(n_epochs)
# val_dataset = tf.data.Dataset.from_tensor_slices(val_data)

# # 创建一个feedable iterator
# handle = tf.placeholder(tf.string, [])
# feed_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
#                                                     train_dataset.output_shapes)
# next_element = feed_iterator.get_next()

# # 创建不同的iterator
# train_iterator = train_dataset.make_one_shot_iterator()
# val_iterator = val_dataset.make_initializable_iterator()

# with tf.Session() as sess:
#     # 生成对应的handle
#     train_handle = sess.run(train_iterator.string_handle())
#     val_handle = sess.run(val_iterator.string_handle())

#     # 训练
#     for n in range(n_epochs):
#         for i in range(100):
#             print(i, sess.run(next_element, feed_dict={handle: train_handle}))
#         # 验证
#         if n % 10 == 0:
#             sess.run(val_iterator.initializer)
#             for i in range(20):
#                 print(sess.run(next_element, feed_dict={handle: val_handle}))
