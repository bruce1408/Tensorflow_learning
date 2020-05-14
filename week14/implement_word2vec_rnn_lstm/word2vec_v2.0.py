# coding=utf-8
import math
import os
import File_Interface as FI
from operator import itemgetter as _itemgetter
import numpy as np
import jieba
from sklearn import preprocessing
from collections import Counter
import numpy as np
from time import sleep
from tqdm import trange, tqdm

np.random.seed(0)
np.set_printoptions(suppress=True, threshold=np.NaN, linewidth=100)


class Word2Vec():
    def __init__(self, vec_len=15000, learn_rate=0.025, win_len=5, model='cbow'):
        """
        :param vec_len: the size of word vector
        :param learn_rate: the learning rate
        :param win_len: the num of n-gram's n
        :param model: chose the method of cbow or the method of skip-gram
        return the dict of the word,include the vector of this word and other parameters
        """
        self.cutted_text_list = None
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.win_len = win_len
        self.model = model
        self.word_dict = None  # each element is a dict, including: word,possibility,vector,huffmancode
        self.huffman = None  # the object of HuffmanTree

    def Load_Word_Freq(self, word_freq_path):
        # load the info of word frequence
        # will generate a word dict
        if self.word_dict is not None:
            raise RuntimeError('the word dict is not empty')
        word_freq = FI.load_pickle(word_freq_path)
        self.__Gnerate_Word_Dict(word_freq)

    def __Gnerate_Word_Dict(self, word_freq):
        # generate a word dict
        # which containing the word, freq, possibility, a random initial vector and Huffman value
        if not isinstance(word_freq, dict) and not isinstance(word_freq, list):
            raise ValueError('the word freq info should be a dict or list')
        word_dict = {}  # 这个词典包含多个键值
        if isinstance(word_freq, dict):
            # if word_freq is in type of dictionary
            sum_count = sum(word_freq.values())
            for word in word_freq:
                temp_dict = dict(
                    word=word,
                    freq=word_freq[word],
                    possibility=word_freq[word] / sum_count,
                    vector=np.random.random([1, self.vec_len]),
                    Huffman=None
                )
                word_dict[word] = temp_dict
        else:
            # if word_freq is in type of list
            freq_list = [x[1] for x in word_freq]
            # print(freq_list)
            sum_count = sum(freq_list)

            for item in word_freq:
                # 默认的字典 叶子节点随机赋值
                temp_dict = dict(
                    word=item[0],
                    freq=item[1],
                    possibility=item[1] / sum_count,
                    vector=np.random.random([1, self.vec_len]),
                    Huffman=None
                )
                word_dict[item[0]] = temp_dict
        self.word_dict = word_dict  # 到这一步的话 word_dict 已经赋值了

    def Import_Model(self, model_path):
        model = FI.load_pickle(model_path)  # a dict, {'word_dict','huffman','vec_len'}
        self.word_dict = model.word_dict
        self.huffman = model.huffman
        self.vec_len = model.vec_len
        self.learn_rate = model.learn_rate
        self.win_len = model.win_len
        self.model = model.model

    def Export_Model(self, model_path):
        data = dict(
            word_dict=self.word_dict,
            huffman=self.huffman,
            vec_len=self.vec_len,
            learn_rate=self.learn_rate,
            win_len=self.win_len,
            model=self.model
        )
        FI.save_pickle(data, model_path)

    def Train_Model(self, text_list):
        # generate the word_dict and huffman tree
        if self.huffman is None:
            # if the dict is not loaded, it will generate a new dict
            if self.word_dict is None:
                wc = WordCounter(text_list)
                # print('the train_model is: ', wc.count_res.larger_than(2))
                self.__Gnerate_Word_Dict(wc.count_res.larger_than(1))
                self.cutted_text_list = wc.text_list  # 这个是单词分词之后的列表了
                # print(self.cutted_text_list)
            # generate a huffman tree according to the possibility of words
            self.huffman = HuffmanTree(self.word_dict, vec_len=self.vec_len)  # 产生霍夫曼编码
        print('word_dict and huffman tree already generated, ready to train vector')

        # start to train word vector
        before = (self.win_len - 1) >> 1  # 选择词的个数
        after = self.win_len - 1 - before

        if self.model == 'cbow':
            method = self.__Deal_Gram_CBOW  # 这个方法的设置也是很诡异的
        else:
            method = self.__Deal_Gram_SkipGram

        if self.cutted_text_list:
            # print(self.cutted_text_list)
            # if the text has been cutted
            total = self.cutted_text_list.__len__()  # 其实是一个二维的，每个维度中是单词的个数 3
            count = 0
            for line in tqdm(self.cutted_text_list, desc='train model:'):
                sleep(0.01)
                line_len = line.__len__()  # 每一句是多少单词
                for i in range(line_len):
                    method(line[i], line[max(0, i - before):i] + line[i + 1:min(line_len, i + after + 1)])  # 窗口字符大小
                    # print('the word is:', line[i])
                    # print('the before word near is:', line[max(0, i-before):i])
                    # print('the after word is:', line[i + 1:min(line_len, i + after + 1)])
                count += 1

                # print('{c} of {d}'.format(c=count, d=total))


        else:
            # if the text has note been cutted
            for line in text_list:
                line = list(jieba.cut(line, cut_all=False))
                line_len = line.__len__()
                for i in range(line_len):
                    method(line[i], line[max(0, i - before):i] + line[i + 1:min(line_len, i + after + 1)])
        print('word vector has been generated')

    def __Deal_Gram_CBOW(self, word, gram_word_list):

        if not self.word_dict.__contains__(word):  # 是否包含该字符word
            return

        word_huffman = self.word_dict[word]['Huffman']  # 得到这个词的霍夫曼编码
        gram_vector_sum = np.zeros([1, self.vec_len])  # 初始化非叶子节点的向量为0
        # print('the len is:', gram_word_list.__len__())  # 长度为4
        for i in range(gram_word_list.__len__())[::-1]:  # 逆序输出
            item = gram_word_list[i]
            if self.word_dict.__contains__(item):  # 所有叶子节点的向量相加，非叶子节点直接弹出不计
                gram_vector_sum += self.word_dict[item]['vector']
            else:
                gram_word_list.pop(i)

        if gram_word_list.__len__() == 0:
            return

        e = self.__GoAlong_Huffman(word_huffman, gram_vector_sum, self.huffman.root)

        for item in gram_word_list:
            self.word_dict[item]['vector'] += e
            self.word_dict[item]['vector'] = preprocessing.normalize(self.word_dict[item]['vector'])

    def __Deal_Gram_SkipGram(self, word, gram_word_list):

        if not self.word_dict.__contains__(word):
            return

        word_vector = self.word_dict[word]['vector']
        for i in range(gram_word_list.__len__())[::-1]:
            if not self.word_dict.__contains__(gram_word_list[i]):
                gram_word_list.pop(i)

        if gram_word_list.__len__() == 0:
            return

        for u in gram_word_list:
            u_huffman = self.word_dict[u]['Huffman']
            e = self.__GoAlong_Huffman(u_huffman, word_vector, self.huffman.root)
            self.word_dict[word]['vector'] += e
            self.word_dict[word]['vector'] = preprocessing.normalize(self.word_dict[word]['vector'])

    def __GoAlong_Huffman(self, word_huffman, input_vector, root):
        """
        :param word_huffman: word huffman code
        :param input_vector:
        :param root:
        :return:
        """

        node = root
        e = np.zeros([1, self.vec_len])
        for level in range(word_huffman.__len__()):
            huffman_charat = word_huffman[level]
            q = self.__Sigmoid(input_vector.dot(node.value.T))
            grad = self.learn_rate * (1 - int(huffman_charat) - q)
            e += grad * node.value
            node.value += grad * input_vector
            node.value = preprocessing.normalize(node.value)
            if huffman_charat == '0':
                node = node.right
            else:
                node = node.left
        return e

    def __Sigmoid(self, value):
        return 1 / (1 + math.exp(-value))


class HuffmanTreeNode():
    def __init__(self, value, possibility):
        # common part of leaf node and tree node
        self.possibility = possibility
        self.left = None
        self.right = None
        # value of leaf node  will be the word, and be
        # mid vector in tree node
        self.value = value  # the value of word
        self.Huffman = ""  # store the huffman code

    def __str__(self):
        return 'HuffmanTreeNode object, value: {v}, possibility: {p}, Huffman: {h}' \
            .format(v=self.value, p=self.possibility, h=self.Huffman)


class HuffmanTree():
    """
        this class is to build a huffman tree
        return a binary tree of huffman
    """

    def __init__(self, word_dict, vec_len=15000):

        self.vec_len = vec_len  # the length of word vector
        self.root = None

        word_dict_list = list(word_dict.values())
        node_list = [HuffmanTreeNode(x['word'], x['possibility']) for x in word_dict_list]
        self.build_tree(node_list)
        # self.build_CBT(node_list)
        self.generate_huffman_code(self.root, word_dict)

    def build_tree(self, node_list):
        # node_list.sort(key=lambda x:x.possibility,reverse=True)
        # for i in range(node_list.__len__()-1)[::-1]:
        #     top_node = self.merge(node_list[i],node_list[i+1])
        #     node_list.insert(i,top_node)
        # self.root = node_list[0]
        lines = tqdm(total=2, desc='huffman tree')
        while node_list.__len__() > 1:
            i1 = 0  # i1表示概率最小的节点
            i2 = 1  # i2 概率第二小的节点
            if node_list[i2].possibility < node_list[i1].possibility:
                [i1, i2] = [i2, i1]
            for i in range(2, node_list.__len__()):  # 找到最小的两个节点
                if node_list[i].possibility < node_list[i2].possibility:
                    i2 = i
                    if node_list[i2].possibility < node_list[i1].possibility:
                        [i1, i2] = [i2, i1]
            top_node = self.merge(node_list[i1], node_list[i2])
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, top_node)
            lines.update(1)

        self.root = node_list[0]
        lines.close()

    def build_CBT(self, node_list):  # build a complete binary tree
        node_list.sort(key=lambda x: x.possibility, reverse=True)
        node_num = node_list.__len__()
        before_start = 0
        while node_num > 1:
            for i in range(node_num >> 1):
                top_node = self.merge(node_list[before_start + i * 2], node_list[before_start + i * 2 + 1])
                node_list.append(top_node)
            if node_num % 2 == 1:
                top_node = self.merge(node_list[before_start + i * 2 + 2], node_list[-1])
                node_list[-1] = top_node
            before_start = before_start + node_num
            node_num = node_num >> 1
        self.root = node_list[-1]

    def generate_huffman_code(self, node, word_dict):
        # # use recursion in this edition
        # if node.left==None and node.right==None :
        #     word = node.value
        #     code = node.Huffman
        #     print(word,code)
        #     word_dict[word]['Huffman'] = code
        #     return -1
        #
        # code = node.Huffman
        # if code==None:
        #     code = ""
        # node.left.Huffman = code + "1"
        # node.right.Huffman = code + "0"
        # self.generate_huffman_code(node.left, word_dict)
        # self.generate_huffman_code(node.right, word_dict)

        # use stack butnot recursion in this edition
        stack = [self.root]
        while stack.__len__() > 0:
            node = stack.pop()
            # go along left tree
            while node.left or node.right:
                code = node.Huffman
                node.left.Huffman = code + "1"
                node.right.Huffman = code + "0"
                stack.append(node.right)
                node = node.left
            word = node.value
            code = node.Huffman
            # print(word,'\t',code.__len__(),'\t',node.possibility)
            word_dict[word]['Huffman'] = code

    def merge(self, node1, node2):
        top_pos = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode(np.zeros([1, self.vec_len]), top_pos)
        if node1.possibility >= node2.possibility:
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node


class WordCounter():
    """
        the class is used to statistic the word number of this text!
    """

    # can calculate the freq of words in a text list

    # for example
    # >>> data = ['Merge multiple sorted inputs into a single sorted output',
    #           'The API below differs from textbook heap algorithms in two aspects']
    # >>> wc = WordCounter(data)
    # >>> print(wc.count_res)

    # >>> MulCounter({' ': 18, 'sorted': 2, 'single': 1, 'below': 1, 'inputs': 1, 'The': 1, 'into': 1, 'textbook': 1,
    #                'API': 1, 'algorithms': 1, 'in': 1, 'output': 1, 'heap': 1, 'differs': 1, 'two': 1, 'from': 1,
    #                'aspects': 1, 'multiple': 1, 'a': 1, 'Merge': 1})

    def __init__(self, text_list):
        self.text_list = text_list
        self.stop_word = self.Get_Stop_Words()
        self.count_res = None

        self.Word_Count(self.text_list)

    def Get_Stop_Words(self):
        ret = []
        ret = FI.load_pickle('./static/stop_words.pkl')
        return ret

    def Word_Count(self, text_list, cut_all=True):

        filtered_word_list = []
        count = 0
        for line in tqdm(text_list, desc='word count: '):
            res = jieba.cut(line, cut_all=cut_all)
            res = list(res)
            text_list[count] = res
            count += 1
            filtered_word_list += res

        self.count_res = MulCounter(filtered_word_list)
        for word in self.stop_word:
            try:
                self.count_res.pop(word)
            except:
                pass


class MulCounter(Counter):
    """
        the class is inherit the class Counter,the main function is to statics the num of text word!
    """

    # a class extends from collections.Counter
    # add some methods, larger_than and less_than
    def __init__(self, element_list):
        super().__init__(element_list)

    def larger_than(self, minvalue, ret='list'):
        temp = sorted(self.items(), key=_itemgetter(1), reverse=True)
        low = 0
        high = temp.__len__()
        while high - low > 1:
            mid = (low + high) >> 1
            if temp[mid][1] >= minvalue:
                low = mid
            else:
                high = mid
        if temp[low][1] < minvalue:
            if ret == 'dict':
                return {}
            else:
                return []
        if ret == 'dict':
            ret_data = {}
            for ele, count in temp[:high]:
                ret_data[ele] = count
            return ret_data
        else:
            return temp[:high]

    def less_than(self, maxvalue, ret='list'):
        temp = sorted(self.items(), key=_itemgetter(1))
        low = 0
        high = temp.__len__()
        while (high - low) > 1:
            mid = (low + high) >> 1
            if temp[mid][1] <= maxvalue:
                low = mid
            else:
                high = mid
        if temp[low][1] > maxvalue:
            if ret == 'dict':
                return {}
            else:
                return []
        if ret == 'dict':
            ret_data = {}
            for ele, count in temp[:high]:
                ret_data[ele] = count
            return ret_data
        else:
            return temp[:high]


if __name__ == '__main__':
    # text = FI.load_pickle('./static/demo.pkl')
    # text =[ x['dealed_text']['left_content'][0] for x in text]

    data = ['Merge multiple sorted inputs into a single sorted output',
            'The API below differs from textbook heap algorithms in two spects',
            'two aspects is more like single sorted output and the algorithm inputs quickly']
    wv = Word2Vec(vec_len=100)
    wv.Train_Model(data)
    # print(wv.word_dict)
    # print(whole_data)

    # ------------------- big data -----------------------
    # whole_data = []
    # with open('./static/input_data.txt', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip('\n')
    #         whole_data.append(line)
    #     # print(whole_data)
    # # print(whole_data)
    # wv = Word2Vec(vec_len=130)
    # print('begin train...')
    # wv.Train_Model(whole_data)
    #
    # # print(wv.word_dict)
    # total_words_key = [i for i in wv.word_dict.keys()]
    # vector_total = wv.word_dict.values()
    # path = './test_vector.txt'
    # if not os.path.exists(path):
    #     os.system(r'touch %s' % path)
    # with open(path, 'w') as f:
    #     for key in total_words_key:
    #         line = key + '\t' + str(wv.word_dict[key]['vector'][0]) + '\n'
    #         print(line)
    #         f.write(line)
    # -----------------------------------------------------
    # with open('./words_vector.txt', 'w') as f:
    #     for key in total_words_key:
    #         line = key + '\t' + str(wv.word_dict[key]['vector'][0]) +'\n'
    #         print(line)
    #         f.write(line)
    # FI.save_pickle(wv.word_dict, './static/wv.pkl')

    # data = FI.load_pickle('./static/wv.pkl')
    # data = FI.load_pickle('./static/stop_words.pkl')
    # print(data)
    # x = {}
    # for key in data:
    #     temp = data[key]['vector']
    #     temp = preprocessing.normalize(temp)
    #     x[key] = temp
    # FI.save_pickle(x,'./static/normal_wv.pkl')
    #
    # x = FI.load_pickle('./static/normal_wv.pkl')
    # #
    # def cal_simi(data,key1,key2):
    #     return data[key1].dot(data[key2].T)[0][0]
    # keys=list(x.keys())
    # for key in keys:
    #     print(key, '\t', cal_simi(x, 'three', key))
