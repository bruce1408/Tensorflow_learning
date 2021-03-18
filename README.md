# Tensorflow_learning

[![](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/bruce1408/Pytorch_learning)
![](https://img.shields.io/badge/platform-TensorFlow-brightgreen.svg)
![](https://img.shields.io/badge/python-3.7-blue.svg)


This repository provides tutorial code for deep learning researchers to learn [TensorFlow](https://www.tensorflow.org/)

TensorFlow is a machine learning system that operates at large scale and in heterogeneous environments.
TensorFlow uses dataflow graphs to represent computation, shared state, and the operations that mutate that state.
It maps the nodes of a dataflow graph across many machines in a cluster, and within a machine across multiple
computational devices, including multicore CPUs, general-purpose GPUs, and custom-designed ASICs known as Tensor
Processing Units (TPUs). This architecture gives flexibility to the application developer: whereas in previous “parameter server” designs the management of shared state is built into the system, TensorFlow enables
developers to experiment with novel optimizations and training algorithms. TensorFlow supports a variety of applications, with a focus on training and inference on deep neural networks. Several Google services use TensorFlow in production, we have released it as an open-source project, and it has become widely used for machine learning research. In this paper, we describe the TensorFlow dataflow model and demonstrate the compelling performance that TensorFlow achieves for several real-world applications.

This repository contains:

- **week01 Tensorflow Anaconda intro，Tensorflow install in CPU.**
- **week02 Tensorflow Basic knowledge，include graphs, session, tensor, Variable.**
- **week03 Tensorflow Basic Algorithm Linear Regreesion.**
- **week04 Loss Function like softmax, cross-entropy and  Tricks Dropout intro.**
- **week05 Use Tensorboard to inspect and understand Tensorflow runs and graphs.**
- **week06 Implementing a One-Layer/Multilayer Neural Network.**
- **week07 Demonstrating how to use RNN and LSTM**
- **week08 Save and Restore Model**
- **week09 Design your own Network and train them on IMG classify.**
- **week10 Usage of trained Tensorflow model to detect verification code.**
- **week11 Tensorflow in NLP I**
- **week12 Tensorflow in NLP II**
- **week13 TensorFlow in GAN.**
- **week14 Taking TensorFlow to production**
- **week15 Fine-tune the Network with trained model**

## Table of Contents

- [Install](#install)
- [Dataset](#Dataset)
- [Related impacts](#Related-impacts)
- [Contributors](#Contributors)
- [Reference](#Reference)

## Install

This project uses [TensorfFow](https://tensorflow.org/). Go check them out if you don't have them locally installed and thirt-party dependencies.

```sh
CUDA 10.0+
Tensorflow 1.14.0
$ pip install -r requirements.txt
```

## Dataset

All data for this project can be found as follow

- data <https://pan.baidu.com/s/1sxojcCXFKqFRVnOWD3vouw>  pasd: ljpf
- inception-2015-12-05.tgz: <https://pan.baidu.com/s/1o_BCsopsbgKMPqlNzMwTYw>  pasd: zt3t
- classify_image_graph_def.pd: <https://pan.baidu.com/s/1yMoF8ol4HemE4SnqCIDa0A>  pasd: 7a6k
- captcha/images: <https://pan.baidu.com/s/1p_ZYQyv7quiYdLydFw8SmA>  pasd:m1y4

## Related Impacts

- [Aymeric Damien](https://github.com/aymericdamien)
- [Hvass-Labs](https://github.com/Hvass-Labs)

## Reference

### Online Video

- YouTube：[tensorflow教程（十课）](https://www.youtube.com/watch?v=eAtGqz8ytOI&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk&index=2&t=0s)
- B 站：[《深度学习框架TensorFlow学习与应用》](https://www.bilibili.com/video/av20542427/)

-《深度学习框架Tensorflow学习与应用》（含视频+代码+课件，视频总时长：13小时31分钟）

> 链接: <https://pan.baidu.com/s/16OINOrFiRXbqmqOFjCFzLQ> 密码: 1a8j

-《深度学习框架Tensorflow学习与应用[只有videos-720p]》（该份资料只有视频文件）

> 链接: <https://pan.baidu.com/s/1oQLgWFEBsVrcKJN4swEdzg> 密码: i3e2

- 油管视频：[TF Girls 修炼指南](https://www.youtube.com/watch?v=TrWqRMJZU8A&list=PLwY2GJhAPWRcZxxVFpNhhfivuW0kX15yG&index=2) 、或 B 站观看： [TF Girls 修炼指南](https://space.bilibili.com/16696495/#/channel/detail?cid=1588)


### Books & Pdf

- [TensorFlow 官方文档中文版](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)
- [《Building Machine Learning Projects with TensorFlow》](https://www.amazon.com/Building-Machine-Learning-Projects-TensorFlow/dp/1786466589) Rodolfo Bonnin:

- [《Learning TensorFlow》](https://www.amazon.com/Learning-TensorFlow-Guide-Building-Systems/dp/1491978511) Tom Hope

- [《TensorFlow Maching Learning Cookbook》](https://www.amazon.com/TensorFlow-Machine-Learning-Cookbook-intelligent/dp/1789131685) Nick McClure:
- [《Tensorflow：实战Google深度学习框架》](https://book.douban.com/subject/26976457/) 郑泽宇/顾思宇：
- [《TensorFlow实战》](https://book.douban.com/subject/26974266/) 黄文坚/唐源

### Slides

- [01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装](/week01/01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装.md)
- [02-Tensorflow的基础使用，包括对图(graphs),会话(session),张量(tensor),变量(Variable)的一些解释和操作](/week02/02-Tensorflow的基础使用，包括对图\(graphs\),会话\(session\),张量\(tensor\),变量\(Variable\)的一些解释和操作.md)
- [03-Tensorflow线性回归以及分类的简单使用](/week03/03-Tensorflow线性回归以及分类的简单使用.md)
- [04-softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍](/week04/04-softmax，交叉熵\(cross-entropy\)，dropout以及Tensorflow中各种优化器的介绍.md)
- [05-使用Tensorboard进行结构可视化，以及网络运算过程可视化](/week05/05-使用Tensorboard进行结构可视化，以及网络运算过程可视化.md)
- [06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题-](/week06/06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题.md)
- [07-递归神经网络LSTM的讲解，以及LSTM网络的使用](/week07/07-递归神经网络LSTM的讲解，以及LSTM网络的使用.md)
- [08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别](/week08/08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别.md)
- [09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别](/week09/09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别.md)
- [10-使用Tensorflow进行验证码识别](/week10/10-使用Tensorflow进行验证码识别.md)
- [11-Tensorflow在NLP中的使用(一)](/week11/11-Tensorflow在NLP中的使用\(一\).md)
- [12-Tensorflow在NLP中的使用(二)](/week12/12-Tensorflow在NLP中的使用\(二\).md)
- [13-Tensorflow例子汇总学习](/week13/contributing.md)
- [14-Tensorflow代码参考学习](/week14/README.md)
- [15-Tensorflow模型微调和迁移学习](/week15)

## Contributors

This project exists thanks to all the people who contribute.
Everyone is welcome to submit code.
