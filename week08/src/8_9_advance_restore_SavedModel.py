import tensorflow as tf
from tensorflow import saved_model as sm

# 首先定义一个极其简单的计算图
X = tf.placeholder(tf.float32, shape=(3,))
scale = tf.Variable([10, 11, 12], dtype=tf.float32)
y = tf.multiply(X, scale)

# 在会话中运行
with tf.Session() as sess:

    sess.run(tf.initializers.global_variables())
    value = sess.run(y, feed_dict={X: [1., 2., 3.]})
    print(value)

    # 准备存储模型
    path = '8_9_model/'
    builder = sm.builder.SavedModelBuilder(path)

    # 构建需要在新会话中恢复的变量的 TensorInfo protobuf
    X_TensorInfo = sm.utils.build_tensor_info(X)
    scale_TensorInfo = sm.utils.build_tensor_info(scale)
    y_TensorInfo = sm.utils.build_tensor_info(y)

    # 构建 SignatureDef protobuf
    SignatureDef = sm.signature_def_utils.build_signature_def(
        inputs={'input_1': X_TensorInfo, 'input_2': scale_TensorInfo},
        outputs={'output': y_TensorInfo}, method_name='what')

    # 将 graph 和变量等信息写入 MetaGraphDef protobuf
    # 这里的 tags 里面的参数和 signature_def_map 字典里面的键都可以是自定义字符串，TensorFlow 为了方便使用，不在新地方将自定义的字符串忘记，可以使用预定义的这些值
    builder.add_meta_graph_and_variables(sess, tags=[sm.tag_constants.TRAINING],
                                         signature_def_map={sm.signature_constants.CLASSIFY_INPUTS: SignatureDef})
# 将 MetaGraphDef 写入磁盘
builder.save()


import tensorflow as tf
from tensorflow import saved_model as sm


# 需要建立一个会话对象，将模型恢复到其中
with tf.Session() as sess:
    path = '8_9_model/'
    MetaGraphDef = sm.loader.load(sess, tags=[sm.tag_constants.TRAINING], export_dir=path)

    # 解析得到 SignatureDef protobuf
    SignatureDef_d = MetaGraphDef.signature_def
    SignatureDef = SignatureDef_d[sm.signature_constants.CLASSIFY_INPUTS]

    # 解析得到 3 个变量对应的 TensorInfo protobuf
    X_TensorInfo = SignatureDef.inputs['input_1']
    scale_TensorInfo = SignatureDef.inputs['input_2']
    y_TensorInfo = SignatureDef.outputs['output']

    # 解析得到具体 Tensor
    # .get_tensor_from_tensor_info() 函数中可以不传入 graph 参数，TensorFlow 自动使用默认图
    X = sm.utils.get_tensor_from_tensor_info(X_TensorInfo, sess.graph)
    scale = sm.utils.get_tensor_from_tensor_info(scale_TensorInfo, sess.graph)
    y = sm.utils.get_tensor_from_tensor_info(y_TensorInfo, sess.graph)

    print(sess.run(scale))
    print(sess.run(y, feed_dict={X: [3., 2., 1.]}))
