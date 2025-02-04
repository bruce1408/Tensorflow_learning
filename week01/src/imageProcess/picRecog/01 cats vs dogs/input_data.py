#By @Kevin Xu
#kevin28520@gmail.com

# 11.08 2017 更新
# 最近入驻了网易云课堂讲师，我的第一门课《使用亚马逊云计算训练深度学习模型》。
# 有兴趣的同学可以学习交流。
# * 我的网易云课堂主页： http://study.163.com/provider/400000000275062/index.htm

# 深度学习QQ群, 1群满): 153032765
# 2群：462661267
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.


# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


import tensorflow as tf
import numpy as np
import os
from PIL import Image

# you need to change this to your data directory
train_dir = '/home/bruce/Downloads/dogs-vs-cats-redux-kernels-edition/train/'
HIGHTSIZE = 128
WIDTHSIZE = 128


def get_files(file_dir):
    """
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        print(name)
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\n There are %d dogs' %(len(cats), len(dogs)))
    
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list


imagepaths, labels = get_files(train_dir)
print(imagepaths)
print(labels)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def imageToTfrecord(filepath, labelpath, tfrecordName):
    writer = tf.python_io.TFRecordWriter(tfrecordName)
    for path, label in zip(filepath, labelpath):
        image = Image.open(path)
        # print(image.size)
        image = image.resize((HIGHTSIZE, WIDTHSIZE), Image.ANTIALIAS)
        # print(image.size)
        image = np.array(image)
        print(image.shape)
        img_raw = image.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            # value=[index]决定了图片数据的类型label
            "label": _int64_feature(label),
            "image": _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


# 生成tfrecord
# imageToTfrecord(img_train, label_train, './train.tfrecord')
def convert_to_tfrecord(images, labels):
    errorCount = 0
    filename = './train_dogs_cat.tfrecord'
    n_samples = len(labels)
    if np.shape(images)[0] != n_samples:
        raise ValueError('Image size %d does not match label size %d.' % (images.shape, n_samples.shape))
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start....')
    for i in np.arange(n_samples):
        try:
            image = Image.open(images[i])
            image = image.resize((HIGHTSIZE, WIDTHSIZE))
            image = np.array(image)
            if image.shape == (HIGHTSIZE, WIDTHSIZE, 3):
                errorCount += 1
                # print(images[i])
                # print(labels[i])
                image_raw = image.tostring()
                label = int(labels[i])
                example = tf.train.Example(features=tf.train.Features(
                    feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                             "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
                ))
                writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error : %s' % e)
            print('Skip it! \n')
    writer.close()
    print("Transform done!")
    print(errorCount)


convert_to_tfrecord(imagepaths, labels)


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes




#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 2
#CAPACITY = 256
#IMG_W = 208
#IMG_H = 208
#
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#
#image_list, label_list = get_files(train_dir)
#image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([image_batch, label_batch])
#            
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)



