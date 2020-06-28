 ### NLP 从 word embedding 到 bert

#### 1、word embedding 
 word embedding是根据上下文得到的词向量，但是它有一个问题就是它是静态的词向量，就是训练之后的词向量的表达是固定的，在使用的时候，不论这个单词
 在句子里面的含义是什么，它不会根据上下文的场景的变化而变化。但是ELMO的思想是先试用训练好的词向量，然后根据上下文的单词的语义去调整单词的词向量的
 表达，调整之后的word Embedding能够根据上下文含义，解决多义词的问题。


#### 2、ELMO
为什么可以解决多义词的问题，因为ELMO采用了两阶段训练过程，第一阶段是使用语言模型与训练，第二阶段是在下游任务的时候，
从与训练的网络中提取对应单词网络结构的word embedding作为新特征补充到下游任务，网络结构有两层双向LSTM，训练好网络之后，每输入一个句子，句子中的
每个单词对应3个word embedding，最底层的是word embedding，然后是第一层双向LSTM对应单词位置的embedding，还有第二层LSTM对应单词位置的
embedding。

ELMO缺点：
- 使用LSTM来作为特征提取，弱于transformer
- 拼接方式双向融合特征能力弱

#### 3、GPT
GPT(Generative Pre-Training),采用两阶段训练过程，和ELMO不同的是，特征抽取采用的是transformer，
以训练语言模型作为主要任务目标，但是采用的是单向的语言模型，只有上文，没有下文

GPT缺点：
- 语言模型是单向的而不是双向

#### 4、bert
Bert采用和GPT完全相同的两阶段模型，首先是语言模型预训练；其次是使用Fine-Tuning模式解决下游任务。和GPT的最主要不同在于在预训练阶段
采用了类似ELMO的双向语言模型。
Bert 主要是特征抽取使用的transformer，第二是与训练语言模型的时候使用了双向语言模型。