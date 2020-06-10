import tensorflow as tf
import inference
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
image_size = 128  # 输入层图片大小
from PIL import Image

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model1/"
MODEL_NAME = "model1000.ckpt.data-00000-of-00001"

# 加载需要预测的图片
# img = Image.open("../../pred/1.jpg")
# img = img.resize(image_size, image_size, 3)

imgPath = "/raid/bruce/tmp/tmp/tensorflow_learning_remote/pred/5.jpeg"

image_data = tf.gfile.FastGFile(imgPath, 'rb').read()

# 将图片格式转换成我们所需要的矩阵格式，第二个参数为1，代表1维
decode_image = tf.image.decode_jpeg(image_data, 3)

# 再把数据格式转换成能运算的float32
resized = tf.image.resize_images(decode_image, [image_size, image_size])
decode_image = tf.image.convert_image_dtype(resized, tf.float32)

# 转换成指定的输入格式形状
image = tf.reshape(decode_image, [-1, image_size, image_size, 3])
print(image.shape)

# 定义预测结果为logit值最大的分类，这里是前向传播算法，也就是卷积层、池化层、全连接层那部分
test_logit = inference.conv_net(image, 2, 0.2, reuse=False, is_training=False)

# 利用softmax来获取概率
probabilities = tf.nn.softmax(test_logit)

# 获取最大概率的标签位置
correct_prediction = tf.argmax(test_logit, 1)

# 定义Savar类
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    # 加载检查点状态，这里会获取最新训练好的模型
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        # 加载模型和训练好的参数
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("加载模型成功：" + ckpt.model_checkpoint_path)

        # 通过文件名得到模型保存时迭代的轮数.格式：model.ckpt-6000.data-00000-of-00001
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        # 获取预测结果
        probabilities, label = sess.run([probabilities, correct_prediction])

        # 获取此标签的概率
        probability = probabilities[0][label]

        print("After %s training step(s),validation label = %d, has %g probability" % (global_step, label, probability))
    else:
        print("模型加载失败！" + ckpt.model_checkpoint_path)
