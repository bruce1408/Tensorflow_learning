"""
HelloWorld example using TensorFlow library.
"""

from __future__ import print_function
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 方法 1，使用传统的图
hello = tf.constant('Hello, TensorFlow!')
# Start tf session
sess = tf.Session()
# Run the op
print(sess.run(hello))

# 方法 2，使用interactivesession 来直接大于，使用eval就可以输出
x = tf.constant(10)
sess = tf.InteractiveSession()
print(x.eval())