"""
Author: your name
Date: 2021-03-17 22:16:13
LastEditTime: 2021-03-17 22:17:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Tensorflow_learning/week01/src/1_1_basic_eager_api.py
"""
"""
Basic introduction to TensorFlow's Eager API.

What is Eager API?
" Eager execution is an imperative, define-by-run interface where operations are
executed immediately as they are called from Python. This makes it easier to
get started with TensorFlow, and can make research and development more
intuitive. A vast majority of the TensorFlow API remains the same whether eager
execution is enabled or not. As a result, the exact same code that constructs
TensorFlow graphs (e.g. using the layers API) can be executed imperatively
by using eager execution. Conversely, most models written with Eager enabled
can be converted to a graph that can be further optimized and/or extracted
for deployment in production without changing code. " - Rajat Monga

"""
# from __future__ import absolute_import, division, print_function


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.python.framework.dtypes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set Eager API
print("Setting Eager mode...")
tfe.enable_eager_execution()

# Define constant tensors
print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)

# Full compatibility with Numpy
print("Mixing operations with Tensors and Numpy Arrays")

# Define constant tensors
a = tf.constant([[2., 1.], [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
b = np.array([[3., 0.], [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")

c = a + b
print("a + b = %s" % c)

d = tf.matmul(a, b)
print("a * b = %s" % d)

print("Iterate through Tensor 'a':")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])

x = ['1', '2', '3', '4', '5', '<e>', '0', '0', '0']
y = tf.convert_to_tensor(x)
print(y)
