import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time
# 输入训练数据 keras接收numpy数组类型的数据
x = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 3, 2],
              [3, 2, 1]])
y = np.array([0, 0, 1, 1]).T
# 最简单的序贯模型，序贯模型是多个网络层的线性堆叠
simple_model = Sequential()
# dense层为全连接层
# 第一层隐含层为全连接层 5个神经元 输入数据的维度为3
simple_model.add(Dense(5, input_dim=3, activation='relu'))
# 第二个隐含层 4个神经元
simple_model.add(Dense(4, activation='relu'))
# 输出层为1个神经元
simple_model.add(Dense(1, activation='sigmoid'))
# 编译模型,训练模型之前需要编译模型
# 编译模型的三个参数：优化器、损失函数、指标列表
simple_model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

# 训练网络 2000次
# Keras以Numpy数组作为输入数据和标签的数据类型。训练模型一般使用fit函数
simple_model.fit(x, y, epochs=2000000)
# 应用模型 进行预测
y_ = simple_model.predict_classes(x[0:1])
print("[0,1,0]的分类结果：" + str(y[0]))
