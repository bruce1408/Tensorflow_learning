"""
Basic Operations example using TensorFlow library.
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf


# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
# with tf.Session() as sess:
#     print("a=2, b=3")
#     print("Addition with constants: %i" % sess.run(a+b))
#     print("Multiplication with constants: %i" % sess.run(a*b))

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
# a = tf.placeholder(tf.int16)
# b = tf.placeholder(tf.int16)
# add = tf.add(a, b)
# mul = tf.multiply(a, b)

# Launch the default graph.
# with tf.Session() as sess:
#     print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
#     print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# More in details:
# Matrix Multiplication from TensorFlow official tutorial

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.], [2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
# product = tf.matmul(matrix1, matrix2)
#
# with tf.Session() as sess:
#     result = sess.run(product)
#     print(result)
#     # ==> [[ 12.]]

# tf.squeeze 只能对维数是1的维度进行压缩
x = np.arange(0, 25).reshape(1, 5, 5, 1)
x = tf.convert_to_tensor(x)
z = tf.squeeze(x, [0, 3])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    value = sess.run(z)
    print(x.shape)
    print(z.shape)
    print("z[0][1]: ", value[0][1])
