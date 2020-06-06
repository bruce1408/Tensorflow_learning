import os
import tensorflow as tf
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.saved_model import tag_constants
from tensorflow.examples.tutorials.mnist import input_data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
如果
"""
mnist = input_data.read_data_sets("/raid/bruce/MNIST_data", one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784], name="Input")  # 为输入op添加命名"Input"
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b, name='softmax')
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))
tf.identity(y, name="Output")  # 为输出op命名为"Output"

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("val data acc is:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

# 将模型保存到文件
# 简单方法：
# tf.saved_model.simple_save(sess, "./advanceSaverAPI_model_simple", inputs={"Input": x}, outputs={"Output": y})
# 复杂方法, 如果已经保存了这个地址，那么下次运行就会报错
builder = tf.saved_model.builder.SavedModelBuilder("./advanceSaverAPI_model_complex")
signature = predict_signature_def(inputs={'Input': x}, outputs={'Output': y})
builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING], signature_def_map={'predict': signature})
builder.save()
