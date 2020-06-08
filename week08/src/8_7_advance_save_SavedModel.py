import os
import tensorflow as tf
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.saved_model import tag_constants
from tensorflow.examples.tutorials.mnist import input_data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""

1.  saved_model模块主要用于TensorFlow Serving。TF Serving是一个将训练好的模型部署至生产环境的系统，
    主要的优点在于可以保持Server端与API不变的情况下，部署新的算法或进行试验，同时还有很高的性能。
2.  保持Server端与API不变有什么好处呢？有很多好处，我只从我体会的一个方面举例子说明一下，比如我们需要部署一个文本分类模型，
    那么输入和输出是可以确定的，输入文本，输出各类别的概率或类别标签。为了得到较好的效果，我们可能想尝试很多不同的模型，
    CNN，RNN，RCNN等，这些模型训练好保存下来以后，在inference阶段需要重新载入这些模型，我们希望的是inference的代码有一份就好，
    也就是使用新模型的时候不需要针对新模型来修改inference的代码。

1.  仅用Saver来保存/载入变量。这个方法显然不行，仅保存变量就必须在inference的时候重新定义Graph(定义模型)，
    这样不同的模型代码肯定要修改。即使同一种模型，参数变化了，也需要在代码中有所体现，至少需要一个配置文件来同步，这样就很繁琐了。
2.  使用tf.train.import_meta_graph导入graph信息并创建Saver， 再使用Saver restore变量。相比第一种，不需要重新定义模型，
    但是为了从graph中找到输入输出的tensor，还是得用graph.get_tensor_by_name()来获取，
    也就是还需要知道在定义模型阶段所赋予这些tensor的名字。如果创建各模型的代码都是同一个人完成的，还相对好控制，
    强制这些输入输出的命名都一致即可。如果是不同的开发者，要在创建模型阶段就强制tensor的命名一致就比较困难了。
    这样就不得不再维护一个配置文件，将需要获取的tensor名称写入，然后从配置文件中读取该参数。
    
    使用SavedModel保存模型可以解决上面的问题。代码参考 8_7， 8_8， 8_9
    补充，官方给的关于mnist的样例代码地址如下：
    https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py#L102-L114
    https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md

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
