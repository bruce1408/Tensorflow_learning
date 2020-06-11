import tensorflow as tf
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.contrib.slim import nets


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
MODEL_SAVE_PATH = "models15/"
MODEL_NAME = "fine-tune_models-750.data-00000-of-00001"
imgPath = "/raid/bruce/dog_cat/test1"
from natsort import natsorted

labels = {'0': 'cat', '1': 'dog'}

image = tf.placeholder(tf.float32, [None, 224, 224, 3])
test_logit, _ = nets.vgg.vgg_16(image, num_classes=2, is_training=False)
probabilities = tf.nn.softmax(test_logit)
# 获取最大概率的标签位置
correct_prediction = tf.argmax(test_logit, 1)
saver = tf.train.Saver()
imageList = list()
labelList = list()


def plot_images(images, labels, num, reallabel):
    for i in np.arange(0, 30):
        plt.subplot(5, 6, i + 1)
        plt.axis('off')
        # print(labels[i])
        plt.title(reallabel[str(labels[i])], fontsize=8)
        plt.subplots_adjust(wspace=0.5, hspace=3)
        plt.imshow(images[i])
    plt.savefig('./result.jpg')
    plt.show()


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
        num = len(os.listdir(imgPath))
        for i in natsorted(os.listdir(imgPath)):
            cnt += 1
            imgpath = os.path.join(imgPath, i)
            img = Image.open(imgpath)

            img = img.resize((224, 224))
            image_ = np.array(img)
            image_ = image_.reshape([1, 224, 224, 3])

            # 获取预测结果
            probabilities_, label = sess.run([probabilities, correct_prediction], feed_dict={image: image_})

            # 获取此标签的概率
            probability = probabilities_[0][label[0]]
            labelList.append(label[0])
            imageList.append(img)
            data[i.split('.')[0]] = label[0].clip(min=0.05, max=0.995)
            # print(data)

            sys.stdout.write('\r>> Creating image %d/%d' % (cnt + 1, num))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

            # print("After %s training step(s),validation label = %d, has %g probability, the img path is %s" %
            #       (global_step, label, probability, imgpath))
        sorted(data.keys())
        print('the result is:', data)
        result = pd.DataFrame.from_dict(data, orient='index', columns=['label'])
        result = result.reset_index().rename(columns={'index': 'id'})
        result.to_csv('/raid/bruce/dog_cat/result.csv', index=False)
        print("predict is done!")

        plot_images(imageList, labelList, cnt, labels)
    else:
        print("模型加载失败！" + ckpt.model_checkpoint_path)
