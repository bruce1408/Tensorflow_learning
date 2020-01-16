from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('/home/bruce/PycharmProjects/picRecog/PicNormalized/dog.jpeg')
image = io.imread("/home/bruce/PycharmProjects/picRecog/PicNormalized/dog.jpeg")
print('lena\n', lena)
print('image\n', image)
print(image.shape)
plt.imshow(image)
plt.show()

image1 = tf.image.per_image_standardization(image)
with tf.Session() as sess:
    result = sess.run(image1)
    print(result.shape)
    plt.imshow(result)
    plt.show()