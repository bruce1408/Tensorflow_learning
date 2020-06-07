import os
import tensorflow as tf
from tensorflow import saved_model as sm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
1.  保存模型, 和之前不通的是，我们只需要知道你网络输入输出的别名，然后利用signatureDef来创建即可，不用知道具体
    输入和输出的tensor的名字是什么
"""
# # 首先定义一个极其简单的计算图
X = tf.placeholder(tf.float32, shape=(3,), name='x1')
scale = tf.Variable([10, 11, 12], dtype=tf.float32, name='scale1')
y = tf.multiply(X, scale, name='y1')

# # 在会话中运行
with tf.Session() as sess:

    sess.run(tf.initializers.global_variables())
    value = sess.run(y, feed_dict={X: [1., 2., 3.]})
    print(value)

    # 准备存储模型
    path = '8_9_models/'
    builder = sm.builder.SavedModelBuilder(path)

    # 构建需要在新会话中恢复的变量的 TensorInfo protobuf，graph 和变量等信息写入 MetaGraphDef protobuf
    X_TensorInfo = sm.utils.build_tensor_info(X)  # 定义输入签名
    scale_TensorInfo = sm.utils.build_tensor_info(scale)
    y_TensorInfo = sm.utils.build_tensor_info(y)

    # 构建 SignatureDef protobuf 签名信息，把输入，输出，函数等生成具体的签名对象。
    SignatureDef = sm.signature_def_utils.build_signature_def(
        inputs={'input_1': X_TensorInfo, 'input_2': scale_TensorInfo},
        outputs={'output': y_TensorInfo},
        method_name='what')

    # 这里的 tags 里面的参数和 signature_def_map 字典里面的键都可以是自定义字符串，TensorFlow 为了方便使用，不在新地方将自定义的字符串忘记，可以使用预定义的这些值
    builder.add_meta_graph_and_variables(sess,
                                         tags=[sm.tag_constants.TRAINING],
                                         signature_def_map={sm.signature_constants.CLASSIFY_INPUTS: SignatureDef})
# 将 MetaGraphDef 写入磁盘
builder.save()


"""
2. 加载模型
"""
import os
import tensorflow as tf
from tensorflow import saved_model as sm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 需要建立一个会话对象，将模型恢复到其中
with tf.Session() as sess:
    path = '8_9_models/'
    MetaGraphDef = sm.loader.load(sess, tags=[sm.tag_constants.TRAINING], export_dir=path)

    # 解析得到 SignatureDef protobuf，从signature_def 中加载模型中的签名
    SignatureDef_d = MetaGraphDef.signature_def
    SignatureDef = SignatureDef_d[sm.signature_constants.CLASSIFY_INPUTS]

    # 解析得到 3 个变量对应的 TensorInfo protobuf，在签名里面加上需要的输入输出
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
