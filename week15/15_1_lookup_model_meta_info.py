import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
os.environ['CUDA_VISIBLE_DEVICES']='0'
from tensorflow.contrib.slim import nets

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='inputs')
net, end_points = nets.vgg.vgg_16(inputs, num_classes=1000)
with tf.Session() as sess:
    saver = tf.train.Saver()

    saver.restore(sess, './vgg_16.ckpt')  # 权重保存为.ckpt则需要加上后缀
    """
       查看恢复的模型参数
       tf.trainable_variables()查看的是所有可训练的变量；
       tf.global_variables()获得的与tf.trainable_variables()类似，只是多了一些非trainable的变量，比如定义时指定为trainable=False的变量；
       sess.graph.get_operations()则可以获得几乎所有的operations相关的tensor
    """
    tvs = [v for v in tf.trainable_variables()]

    print('获得所有可训练变量的权重:')
    for v in tvs:
        print(v.name)

    gv = [v for v in tf.global_variables()]
    print('获得所有变量:')
    for v in gv:
        print(v.name)

    # sess.graph.get_operations()可以换为tf.get_default_graph().get_operations()
    ops = [o for o in sess.graph.get_operations()]
    print('获得所有operations相关的tensor:')
    for o in ops:
        print(o.name)
    print('slim: ')
    for var in slim.get_model_variables():
        print(var.op.name)
        print(var.op.name.startwith('fc7'))
