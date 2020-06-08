## 保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别

### 一、使用Saver类中的save和restore来保存和载入模型


#### 1 保存和加载模型

- tensorflow 保存模型的话就是使用 
saver = tf.train.Saver(), 没有指定任何东西,保存的是所有的变量.保存模型是saver.save(sess, 'xxx')
- 模型名称可以随意写,一般是写成./model/model01.ckpt
- 那么就会存在这么几个文件,
```
checkpoint  保存最新的checkpoint文件的记录
model01.ckpt.index  保存的是变量的key和value的对应索引
model01.ckpt.meta  保存完整的网络图结构
model01.ckpt.data-00000-of-00001  变量取值,也是最后要加载的部分
```

- 还有人会写成这样./model/model01
这样后缀不是ckpt,而是
```
model01.index, 
model01.meta, 
model01.data-00000-of-00001
```

加载模型的时候,可以这么写,
```
saver = tf.train.import_meta_graph('./net/myModel_10001.meta')
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./net')
    if ckpt is None:
        print('Model not found, please train your model first')
    else:
        path = ckpt.model_checkpoint_path
        print(path)
        saver.restore(sess, path)
```
第一步就是先加载图结构,加载到saver里面,然后开始从保存模型的文件中读取模型,
这里使用tf.train.get_checkpoint_state('./model')
或者是直接使用restore来直接读取.

- 如果是1000次迭代之后再保存模型,可以使用global_step=1000来设置
- 保存的时候图结构不用每次保存,所以可以:
```
saver.save(sess, 'my-model', global_step=step,write_meta_graph=False)  
```
- 保存最新的4个模型,那么希望在训练每两个小时就保存一次,可以使用max_to_keep
```
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)  
```

``` 
saver = tf.train.Saver()
saver.save() 
```
### 二、使用tensorflow SavedModel来进行保存和加载模型

### 三、使用Google的图像识别网络inception-v3进行图像识别

#### 1、下载inception-v3网络模型

(对应代码：`8-3下载google图像识别网络inception-v3并查看结构.py`)<br>
inception_model 文件夹下是保存的训练结果：（其他文件其实都是`inception-2015-12-05.tgz`文件解压后)<br>
其中，`classify_image_graph_def.pb`是已经训练过的 inception-v3 的模型。

#### 2、使用inception-v3网络模型进行图像识别
- SavedModel 是一种跨语言的序列化格式（protobuf），可以保存和加载模型变量、图和图的元数据，适用于将训练得到的模型保存用于生产环境中的预测过程。由于跨语言的特性,可以使用一种语言保存模型，如训练时使用Python代码保存模型；使用另一种语言恢复模型，如使用C++代码恢复模型，进行前向推理，提高效率。
·SavedModel可以为保存的模型添加签名signature，用于保存指定输入输出的graph, 另外可以为模型中的输入输出tensor指定别名，这样子使用模型的时候就不必关心训练阶段模型的输入输出tensor具体的name是什么，讲模型训练和部署解耦，更加方便。

我们先打开 inception_model 文件夹下,有文件如下:<br> 
`imagenet_2012_challenge_label_map_proto.pbtxt` <br>
`imagenet_synset_to_human_label_map.txt` <br>


简单说明：左侧文件中 target_class 后面的数字代表目标的分类，数值为 1~1000（inception 模型是用来做 1000 个分类的），target_class_string 后面的字符串值对应到右侧文件的第一列，右侧文件的第二列表示对第一列的描述，相当是对分类的描述，从而来表示属于哪一类。

在运行代码之前，先在在当前程序路径下新建 images 文件夹，网上找几张图片保存在 images 下。

代码中，程序的头读取了两个文件：

``` xml
    label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'   
    uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
```

类 `NodeLookup` 的目的就是建立两个文件之间的关系，将`imagenet_2012_challenge_label_map_proto.pbtxt`中的 target_class 对应于`imagenet_synset_to_human_label_map.txt`中的类。

最后的排序代码解释下：

``` python
#排序
top_k = predictions.argsort()[-5:][::-1]
print('top_k:', top_k)
node_lookup = NodeLookup()
for node_id in top_k:     
    #获取分类名称
    human_string = node_lookup.id_to_string(node_id)
    #获取该分类的置信度
    score = predictions[node_id]
    print('%s (score = %.5f)' % (human_string, score))
```

因为概率从小到大排序，所以如上第一行代码表示从倒数第 5 的位置开始取至倒数第 1 的位置，从而得到概率顺序从小到大的前 5 的概率值，再对这 5 个值做个倒序，进而得到从大到小的 5 个概率值。

最后的运行结果如下：

``` xml
images/lion.jpg

top_k: [190  11 206  85  30]
lion, king of beasts, Panthera leo (score = 0.96306)
cougar, puma, catamount, mountain lion, painter, panther, Felis concolor (score = 0.00161)
cheetah, chetah, Acinonyx jubatus (score = 0.00079)
leopard, Panthera pardus (score = 0.00057)
jaguar, panther, Panthera onca, Felis onca (score = 0.00033)
```

### 参考文献
- [tensorflow保存和恢复模型的两种方法介绍](https://zhuanlan.zhihu.com/p/31417693)
- [TensorFlow学习笔记：使用Inception v3进行图像分类](https://www.jianshu.com/p/cc830a6ed54b)
- [Google Inception Net介绍及Inception V3结构分析](https://blog.csdn.net/weixin_39881922/article/details/80346070)
- [深入浅出——网络模型中Inception的作用与结构全解析](https://blog.csdn.net/u010402786/article/details/52433324)
- [tensorflow+inceptionv3图像分类网络结构的解析与代码实现【附下载】](https://blog.csdn.net/k87974/article/details/80221215)
- [Tensorflow 模型保存与恢复（2）使用SavedModel](https://blog.csdn.net/JerryZhang__/article/details/85058005)