import os
import numpy as np
import tensorflow as tf

# import tensorflow.contrib.eager as tfe

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Setting Eager mode...")
# tfe.enable_eager_execution()

# 神经网络的输出
output = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
# logits = tf.constant(output)
logits = tf.convert_to_tensor(output, dtype=tf.float32)
# 对输出做softmax操作
y = tf.nn.softmax(logits)  # 预测值


# 手动计算softmax
def softmaxMaunl(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)


# 真实数据标签，one hot形式
y_ = tf.constant([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
# 将标签稠密化, 找到最大的值的index
dense_y = tf.argmax(y_, 1)  # dense_y = [2 2 2]
# 采用普通方式计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 使用softmax_cross_entropy_with_logits方法计算交叉熵, y的真实标签是one-hot类型
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
# 使用sparse_softmax_cross_entropy_with_logits方法计算交叉熵，y的真实标签不是one-hot类型
cross_entropy3 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=dense_y))

with tf.Session() as sess:
    softmax = sess.run(y)
    print("step1:softmax result=")
    print(softmax)
    print("手动计算softmax:\n", softmaxMaunl(output))
    print("y_ = ")
    print(sess.run(y_))
    print("tf.log(y) = ")
    print(sess.run(tf.log(y)))
    print("dense_y =")
    print(sess.run(dense_y))
    print("step2:cross_entropy result=")
    c_e = sess.run(cross_entropy)
    print(c_e)
    print("Function(softmax_cross_entropy_with_logits) result=")
    c_e2 = sess.run(cross_entropy2)
    print(c_e2)
    print("Function(sparse_softmax_cross_entropy_with_logits) result=")
    c_e3 = sess.run(cross_entropy3)
    print(c_e3)
