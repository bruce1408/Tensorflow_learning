# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import trange, tqdm
import collections
import math
import os
import sys
import argparse
import random
from time import sleep
from tempfile import gettempdir
import logging
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import trange, tqdm

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - [line:%(lineno)d] - %(message)s')
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default=os.path.join(current_path, 'log'),
                    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'
localPath = '/Users/bruce/PycharmProjects/tensorflow-demo/data'


# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(localPath, filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                       local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename


filename = maybe_download('text8.zip', 31344016)  # text8.zip data path


# print(filename)


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))
print('data set size', len(set(vocabulary)))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in tqdm(count, desc='build datasets 1'):
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in tqdm(words, desc='build datasets 2'):
        index = dictionary.get(word, 0)  # 如果这个键值存在就是返回键值，否则返回0
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])  # 都是index 整数
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    # with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                                             num_sampled=num_sampled, num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    # writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    logging.info('going iterator...')
    for step in trange(num_steps, desc='iterator batch'):
        # for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
        _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict, run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        # writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        # if step == (num_steps - 1):
        #     writer.add_run_metadata(run_metadata, 'step%d' % step)

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]  # 从小到大的位置索引
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
        # sleep(0.01)

    final_embeddings = normalized_embeddings.eval()
    print(final_embeddings)

    # Write corresponding labels for the embeddings.
    with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
        for i in xrange(vocabulary_size):
            f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints.
    saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
    # projector.visualize_embeddings(writer, config)


# writer.close()


# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.savefig(filename)


try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join('./', 'tsne.png'))

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)


# # 网络代码详解
# import collections
# import sys
# import re
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import jieba
# from sklearn.manifold import TSNE
#
# '''
# 读取源文件,并转为list输出
# @param filename:文件名
# @return: list of words
# '''
#
#
# # # 读中文的
# # def wseg_sentence(input):
# #     with open(input, 'r',encoding='utf-8') as fp:
# #         ret=[]
# #         for line in fp.readlines():
# #             word = line.strip()
# #             if word !=None and len(word)>0 :
# #                 u_ret = jieba.cut(word, cut_all = False) #精准切分模式
# #                 ret.append(u_ret)
# #                 # print(ret)
# #         print(ret)
# #         return ret
# # 读英文的。
# def read_file(filename):
#     f = open(filename, 'r')
#     file_read = f.read()
#     words_ = re.sub("[^a-zA-Z]+", " ", file_read).lower()  # 正则匹配,只留下单词，且大写改小写
#     words = list(words_.split())  # length of words:1121985
#
#     # #将写出的格式，保存在某个文本里
#     # m=0
#     # for  i in words:
#     #     m=m+1
#     #     if m<10:
#     #         file_object.write(i)
#     return words
#
#
# # words= wseg_sentence('../dataset/hong.txt')
# words = read_file('../dataset/heli.txt')
# file_object = open('../dataset/1111', 'w')
#
# vocabulary_size = 2000  # 预定义频繁单词库的长度
# count = [['UNK', -1]]  # 初始化单词频数统计集合
#
# '''
# 1.给@param “words”中出现过的单词做频数统计，取top 1999频数的单词放入dictionary中，以便快速查询。
# 2.给哈利波特这本“单词库”@param“words”编码，出现在top 1999之外的单词，统一令其为“UNK”（未知），编号为0，并统计这些单词的数量。
# @return: 哈利波特这本书的编码data，每个单词的频数统计count，词汇表dictionary及其反转形式reverse_dictionary
# '''
#
#
# def build_dataset(words):
#     # most_common方法： 去top2000的频数的单词，创建一个dict,放进去。以词频排序
#     counter = collections.Counter(words).most_common(vocabulary_size - 1)
#     # length of all counter:22159 取top1999频数的单词作为vocabulary，其他的作为unknown
#     # print('123  :', type(count))  #list类型
#     # print(len(counter)) #1999个
#     # print(counter)
#     # [('the', 51897), ('and', 27525), ('to', 26996), ('he', 22203), ('of', 21851), ('a', 21043), ('harry', 18165), \
#     # ('was', 15637), ('you', 14627), ('it', 14489),。 ('member', 49), ('fake', 49)。]
#     count.extend(counter)
#     # 搭建dictionary
#     dictionary = {}
#     for word, _ in count:
#         dictionary[word] = len(dictionary)
#     # print('234   :',dictionary.get('the'))   输出为1
#     data = []
#     # 全部单词转为编号
#     # 先判断这个单词是否出现在dictionary，如果是，就转成编号，如果不是，则转为编号0（代表UNK）
#     unk_count = 0
#     for word in words:
#         if word in dictionary:
#             index = dictionary[word]
#         else:
#             index = 0
#             unk_count += 1
#         data.append(index)
#
#     count[0][1] = unk_count
#     reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#     # print('aaaaa   ',data) #根据文章，将全文每个数，对应的排名标记， 比如 the 出现最多，则对应的数为1，不在前1999的标记为0
#     # [959, 18, 7, 6, 1728, 0, 306, 13, 450, 1175, 32, 180, 265, 3, 9, 205,]
#     # print('00000   ',count)  #前1999，个对应的出现的次数，第一个数是 总共多少不在前1999的字数   比如： the:51897
#     # print('bbbbb   ',dictionary)          #前1999的数的dict  ['the':1]这种
#     # print('1111   :',reverse_dictionary)
#     # #按照dict值来排序，变得有规律起来。{0: 'UNK', 1: 'the', 2: 'and', 3: 'to', 4: 'he', 5: 'of', 6: 'a', 7: 'harry', 8: 'was', 9:}
#     return data, count, dictionary, reverse_dictionary
#
#
# data, count, dictionary, reverse_dictionary = build_dataset(words)
# del words  # 删除原始单词列表，节约内存
#
# data_index = 0
#
# '''
# 采用Skip-Gram模式,  生成我们需要的 input label 的那种格式。，可以参看第二个小例子。
# 生成word2vec训练样本
# @param batch_size:每个批次训练多少样本
# @param num_skips: 为每个单词生成多少样本（本次实验是2个），batch_size必须是num_skips的整数倍,这样可以确保由一个目标词汇生成的样本在同一个批次中。
# @param skip_window:单词最远可以联系的距离（本次实验设为1，即目标单词只能和相邻的两个单词生成样本），2*skip_window>=num_skips
# '''
#
#
# def generate_batch(batch_size, num_skips, skip_window):
#     global data_index
#     assert batch_size % num_skips == 0
#     assert num_skips <= 2 * skip_window
#
#     batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#     # print(batch)  128个一维数组
#     labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#     # print(labels)  128个二维数组
#     span = 2 * skip_window + 1  # 入队长度
#     # print(span)
#     buffer = collections.deque(maxlen=span)
#
#     for _ in range(span):  # 双向队列填入初始值
#
#         buffer.append(data[data_index])
#         data_index = (data_index + 1) % len(data)
#         # print('cishu :000', batch_size//num_skips)
#     for i in range(batch_size // num_skips):  # 第一次循环，i表示第几次入双向队列deque
#         for j in range(span):  # 内部循环，处理deque
#             if j > skip_window:
#                 batch[i * num_skips + j - 1] = buffer[skip_window]
#                 labels[i * num_skips + j - 1, 0] = buffer[j]
#             elif j == skip_window:
#                 continue
#             else:
#                 batch[i * num_skips + j] = buffer[skip_window]
#                 labels[i * num_skips + j, 0] = buffer[j]
#         buffer.append(data[data_index])  # 入队一个单词，出队一个单词
#         data_index = (data_index + 1) % len(data)
#     # print('batch    :',batch)
#     # print('label    :',labels)
#     return batch, labels
#
#
# # 开始训练
# batch_size = 128
# embedding_size = 128
# skip_window = 1
# num_skips = 2
# num_sampled = 64  # 训练时用来做负样本的噪声单词的数量
# # 验证数据
# valid_size = 16  # 抽取的验证单词数
# valid_window = 100  # 验证单词只从频数最高的100个单词中抽取
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 不重复在0——10l里取16个
#
# graph = tf.Graph()
# with graph.as_default():
#     train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
#     train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#     valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
#     # 单词维度为 2000单词大小，128向量维度
#     embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1))  # 初始化embedding vector
#     # 使用tf.nn.embedding_lookup查找输入train_inputs 对应的向量embed
#     embed = tf.nn.embedding_lookup(embeddings, train_inputs)
#
#     # 用NCE loss作为优化训练的目标
#     # tf.truncated_normal初始化nce loss中的权重参数 nce_weights,并将nce_biases初始化为0
#     nce_weights = tf.Variable(
#         tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.math.sqrt(embedding_size)))
#     nce_bias = tf.Variable(tf.zeros([vocabulary_size]))
#     # 计算学习出的词向量embedding在训练数据上的loss,并使用tf.reduce_mean进行汇总
#     loss = tf.reduce_mean(
#         tf.nn.nce_loss(nce_weights, nce_bias, embed, train_labels, num_sampled, num_classes=vocabulary_size))
#     # 学习率为1.0，L2范式标准化后的enormalized_embedding。
#     # 通过cos方式来测试  两个之间的相似性，与向量的长度没有关系。
#     optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
#     norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
#     normalized_embeddings = embeddings / norm  # 除以其L2范数后得到标准化后的normalized_embeddings
#
#     valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
#     # 如果输入的是64，那么对应的embedding是normalized_embeddings第64行的vector
#     similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)  # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
#     print('相似性：', similarity)
#     init = tf.global_variables_initializer()
#
# num_steps = 100001
# with tf.Session(graph=graph) as session:
#     init.run()
#     print("Initialized")
#     avg_loss = 0
#     for step in range(num_steps):
#         batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)  # 产生批次训练样本
#         feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}  # 赋值
#         _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
#         avg_loss += loss_val
#         # 每2000次，计算一下平均loss并显示出来。
#         if step % 2000 == 0:
#             if step > 0:
#                 avg_loss /= 2000
#             print("Avg loss at step ", step, ": ", avg_loss)
#             avg_loss = 0
#             # 每10000次，验证单词与全部单词的相似度，并将与每个验证单词最相似的8个找出来。
#         if step % 10000 == 0:
#             sim = similarity.eval()
#             for i in range(valid_size):
#                 valid_word = reverse_dictionary[valid_examples[i]]  # 得到验证单词
#                 top_k = 8
#                 nearest = (-sim[i, :]).argsort()[1:top_k + 1]  # 每一个valid_example相似度最高的top-k个单词
#                 log_str = "Nearest to %s:" % valid_word
#                 for k in range(top_k):
#                     close_word = reverse_dictionary[nearest[k]]
#                     log_str = "%s %s," % (log_str, close_word)
#                 print(log_str)
#     final_embedding = normalized_embeddings.eval()
# '''
# 可视化Word2Vec散点图并保存
# '''
#
#
# def plot_with_labels(low_dim_embs, labels, filename):
#     # low_dim_embs 降维到2维的单词的空间向量
#     assert low_dim_embs.shape[0] >= len(labels), "more labels than embedding"
#     plt.figure(figsize=(18, 18))
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         plt.scatter(x, y)
#         # 展示单词本身
#         plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
#     plt.savefig(filename)
#
#
# '''
# tsne实现降维，将原始的128维的嵌入向量降到2维
# '''
# print('123213123')
# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# plot_number = 100
# low_dim_embs = tsne.fit_transform(final_embedding[:plot_number, :])
# labels = [reverse_dictionary[i] for i in range(plot_number)]
# plot_with_labels(low_dim_embs, labels, './plot.png')
# print('1231231')
