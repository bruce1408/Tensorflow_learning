import os
import numpy as np
import tensorflow as tf
from tensorflow import saved_model as sm
from tensorflow.saved_model import tag_constants
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.saved_model.signature_def_utils import predict_signature_def
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(10)
# 需要建立一个会话对象，将模型恢复到其中
with tf.Session() as sess:
    """
    第一种加载模型的方式
    """
    # 解析得到 SignatureDef protobuf
    # path = './advanceSaverAPI_model_complex'
    # MetaGraphDef = sm.loader.load(sess, tags=[sm.tag_constants.SERVING], export_dir=path)
    # SignatureDef_d = MetaGraphDef.signature_def
    # SignatureDef = SignatureDef_d['predict']
    #
    # # # 解析得到 3 个变量对应的 TensorInfo protobuf
    # X = SignatureDef.inputs['Input'].name
    # y = SignatureDef.outputs['Output'].name

    """
    第二种方法和第一种方法其实是一样的，就是细节上有点不一样
    """
    path = './advanceSaverAPI_model_complex'
    MetaGraphDef = sm.loader.load(sess, tags=[sm.tag_constants.SERVING], export_dir=path)
    SignatureDef_d = MetaGraphDef.signature_def
    SignatureDef = SignatureDef_d['predict']

    # # 解析得到 3 个变量对应的 TensorInfo protobuf
    X_TensorInfo = SignatureDef.inputs['Input']
    y_TensorInfo = SignatureDef.outputs['Output']
    #
    # # 解析得到具体 Tensor
    # get_tensor_from_tensor_info() 函数中可以不传入 graph 参数，TensorFlow 自动使用默认图
    X = sm.utils.get_tensor_from_tensor_info(X_TensorInfo, sess.graph)
    y = sm.utils.get_tensor_from_tensor_info(y_TensorInfo, sess.graph)
    """
    第三种加载模型的方式,最简单
    """
    # path = './advanceSaverAPI_model_complex'
    # sm.loader.load(sess, tags=[sm.tag_constants.SERVING], export_dir=path)
    # graph = tf.get_default_graph()
    # X = sess.graph.get_tensor_by_name("Input:0")
    # y = sess.graph.get_tensor_by_name("Output:0")

    # 输出结果
    x_ = np.random.random((32, 784))
    predict = sess.run(y, feed_dict={X: x_})
    prediction = tf.argmax(predict, 1)
    print(sess.run(prediction))
