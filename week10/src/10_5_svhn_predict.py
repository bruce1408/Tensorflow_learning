import tensorflow as tf
import svhn_inference
import os, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
MODEL_SAVE_PATH = "model_svhn5/"
MODEL_NAME = "model10000.ckpt.data-00000-of-00001"
# imgPath = "/raid/bruce/tmp/tmp/tensorflow_learning_remote/pred/"
# imgPath = "../../week03/src/images/dogs"
imgPath = "../datasets/svhn/mchar_test_a"
from natsort import natsorted


image = tf.placeholder(tf.float32, [None, 224, 224, 3])
test_logit0, test_logit1, test_logit2, test_logit3 = svhn_inference.conv_net(image,
                                                                                          11,
                                                                                          0.2,
                                                                                          reuse=False,
                                                                                          is_training=False)

probabilities0 = tf.nn.softmax(test_logit0)
probabilities1 = tf.nn.softmax(test_logit1)
probabilities2 = tf.nn.softmax(test_logit2)
probabilities3 = tf.nn.softmax(test_logit3)
# probabilities4 = tf.nn.softmax(test_logit4)

# 获取最大概率的标签位置
correct_prediction0 = tf.argmax(test_logit0, 1)
correct_prediction1 = tf.argmax(test_logit1, 1)
correct_prediction2 = tf.argmax(test_logit2, 1)
correct_prediction3 = tf.argmax(test_logit3, 1)
# correct_prediction4 = tf.argmax(test_logit4, 1)

saver = tf.train.Saver()


def plot_images(images, labels, num):
    title = ""
    for i in labels:
        title += '_'+i
    for i in np.arange(0, 30):
        plt.subplot(5, 6, i + 1)
        plt.axis('off')
        # print(labels[i])
        plt.title(title, fontsize=8)
        plt.subplots_adjust(wspace=0.5, hspace=3)
        plt.imshow(images[i])
    plt.show()


imageList = list()
cnt = 0
predict_val = list()
data = dict()
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

        for im in natsorted(os.listdir(imgPath)):
            num = len(os.listdir(imgPath))
            labelList = list()

            cnt += 1
            imgpath = os.path.join(imgPath, im)
            img = Image.open(imgpath)

            img = img.resize((224, 224))

            image_ = np.array(img)
            # print('image shape ', image_.shape)

            # image_ = np.array(img.convert('L'))
            image_ = image_.reshape([1, 224, 224, 3])

            # image = np.array(image.convert('L'))  # 转成灰度图即可 尺寸变成 224*224*1

            # 获取预测结果
            label0, label1, label2, label3 = sess.run(
                [correct_prediction0, correct_prediction1,
                 correct_prediction2, correct_prediction3], feed_dict={image: image_})

            # 获取此标签的概率
            labelList.append(label0[0])
            labelList.append(label1[0])
            labelList.append(label2[0])
            labelList.append(label3[0])
            # labelList.append(label4[0])

            imageList.append(img)
            sorted(data.keys())
            tempLabel = ""
            for i in range(4):
                tempLabel += str(labelList[i])
            data[im] = tempLabel

            print("After %s training step(s), validation label0 = %d, label1 = %d, label2 = %d, label3 = %d, "
                  "the img path is %s" % (global_step, label0[0], label1[0], label2[0], label3[0], imgpath))
        # print('the result is:', data)
            sys.stdout.write('\r>> Creating image %d/%d' % (cnt + 1, num))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        result = pd.DataFrame.from_dict(data, orient='index', columns=['label'])
        result = result.reset_index().rename(columns={'index': 'id'})
        result.to_csv('../datasets/svhn/test_a_result.csv', index=False)
        print("predict is done!")

        # plot_images(imageList, labelList, cnt)
    else:
        print("模型加载失败！" + ckpt.model_checkpoint_path)
