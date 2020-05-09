import tensorflow as tf


# 矩阵乘法
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)


# 变量

state = tf.Variable(0, name='counter')
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()  # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


# 举证乘法


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
ouput = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))


# 函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# save model
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np

"""Save to file part"""
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

# init = tf.initialize_all_variables()

# saver = tf.train.Saver()

# with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess, ".my_net/save_net.ckpt")
#    print("Save to path: ", save_path)


"""restore variables part"""
# redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./my_net/save_net.ckpt")
    print("weights:\n", sess.run(W))
    print("biases:\n", sess.run(b))

