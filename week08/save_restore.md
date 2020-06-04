### 保存和加载模型

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

### 参考文献
-  tensorflow保存和恢复模型的两种方法介绍[]