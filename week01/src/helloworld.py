'''
HelloWorld example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

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