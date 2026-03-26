#!/usr/bin/env python
# coding: utf-8

# # 中文自然语言处理用于情感分析

# 本文内容大部分非原创，借用前人内容，只是在思路上向金融量化靠近。
# 
# 原创连接：https://www.youtube.com/watch?v=-mcrmLmNOXA
# 
# 需要的库
# 
# numpy
# 
# jieba 用于分词
# 
# gensim 用于加载预训练的词向量
# 
# tensorflow 用于构建神经网络
# 
# matplotlib

# In[1]:


# 首先加载必用的库
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")


# 导入预训练词向量
# 
# 本文使用了北京师范大学中文信息处理研究所与中国人民大学 DBIIR 实验室的研究者开源的"chinese-word-vectors" 
# 
# github链接为：
# https://github.com/Embedding/Chinese-Word-Vectors
# 
# 云盘连接：https://pan.baidu.com/s/1O6bmpFXM58G2_wVMQuZp8Q
# 
# 提取码：pp2z
# 
# 将下载的文件解压到自己指定的目录，由于聚宽云平台上存储空间有限，本文未使用金融行业词向量，因为这个要1.2G左右，所以使用知乎内容训练的词向量，如果在本机跑数据，直接更换即可。

# In[2]:


# 使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', 
                                          binary=False)


# **词向量模型**  
# 在这个词向量模型里，每一个词是一个索引，对应的是一个长度为300的向量，我们今天需要构建的LSTM神经网络模型并不能直接处理汉字文本，需要先进行分次并把词汇转换为词向量。
# 
# 第一步：使用jieba进行分词，将文章句子转化成切分成词语
# 
# 第二步：将词语索引化，加载的词向量模型中包括所有中文词语，按照使用频率大小排列，每个词语对应一个索引值
# 
# 第三步：词向量化。每个索引对应一个词向量，词向量就是用向量方式描述词语，成功实现由语言向数值的转换。一般常用的词向量都是300维的，也就是说一个汉语词语用一个300维的向量表示，向量数值已经标准化，取值在[-1,1]之间
# 
# 第四步：构建循环神经网络，可以使用RNN,GRU,LSTM具体那个方法好取决于不同的问题，需要多尝试。
# 
# 第五步：使用构建好的语料训练神经网络，最后用训练好的神经网络做预测。
# 
# <img src='flowchart.jpg' style='width:400px;'>

# In[3]:


# 由此可见每一个词都对应一个长度为300的向量，取值在[-1,1]之间
embedding_dim = cn_model['经济'].shape[0]
print('词向量的长度为{}'.format(embedding_dim))
cn_model['经济']


# 模型自带的api可以计算相似度，原理是计算两个向量的余弦相似度，即向量点乘后除以两个向量的模。

# In[5]:


# 计算相似度
cn_model.similarity('橘子', '橙子')


# In[6]:


# dot（'橘子'/|'橘子'|， '橙子'/|'橙子'| ）
np.dot(cn_model['橘子']/np.linalg.norm(cn_model['橘子']), 
cn_model['橙子']/np.linalg.norm(cn_model['橙子']))


# In[ ]:


# 找出最相近的10个词，余弦相似度,此功能可用于扩大同类词范围，在使用贝叶斯方法时，可以填充备选词库
cn_model.most_similar(positive=['牛市'], topn=10)


# In[36]:


# 找出不同的词
test_words = '老师 会计师 程序员 律师 医生 老人'
test_words_result = cn_model.doesnt_match(test_words.split())
print('在 '+test_words+' 中:\n不是同一类别的词为: %s' %test_words_result)


# In[ ]:


#
cn_model.most_similar(positive=['女人','女儿'], negative=['男人'], topn=1)


# **训练语料**  
# 
# 语料下载和上面词向量云盘下载在一个地方。
# 
# 解压后分别为pos和neg，每个文件夹里有2000个txt文件，每个文件内有一段评语，共有4000个训练样本，这样大小的样本数据在NLP中属于非常迷你的：

# In[38]:


# 获得样本的索引，样本存放于两个文件夹中，
# 分别为 正面评价'pos'文件夹 和 负面评价'neg'文件夹
# 每个文件夹中有2000个txt文件，每个文件中是一例评价
import os
pos_txts = os.listdir('pos')
neg_txts = os.listdir('neg')


# In[39]:


print( '样本总共: '+ str(len(pos_txts) + len(neg_txts)) )


# In[40]:


# 现在我们将所有的评价内容放置到一个list里

train_texts_orig = [] # 存储所有评价，每例评价为一条string

# 添加完所有样本之后，train_texts_orig为一个含有4000条文本的list
# 其中前2000条文本为正面评价，后2000条为负面评价

for i in range(len(pos_txts)):
    with open('pos/'+pos_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
for i in range(len(neg_txts)):
    with open('neg/'+neg_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)


# In[41]:


len(train_texts_orig)


# In[42]:


# 我们使用tensorflow的keras接口来建模
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


# **分词和tokenize**  
# 首先我们去掉每个样本的标点符号，然后用jieba分词，jieba分词返回一个生成器，没法直接进行tokenize，所以我们将分词结果转换成一个list，并将它索引化，这样每一例评价的文本变成一段索引数字，对应着预训练词向量模型中的词。

# In[43]:


# 进行分词和tokenize
# train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
train_tokens = []
for text in train_texts_orig:
    # 去掉标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 结巴分词
    cut = jieba.cut(text)
    # 结巴分词的输出结果为一个生成器
    # 把生成器转换为list
    cut_list = [ i for i in cut ]
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    train_tokens.append(cut_list)


# **索引长度标准化**  
# 因为每段评语的长度是不一样的，我们如果单纯取最长的一个评语，并把其他评填充成同样的长度，这样十分浪费计算资源，所以我们取一个折衷的长度。

# In[44]:


# 获得所有tokens的长度
num_tokens = [ len(tokens) for tokens in train_tokens ]
num_tokens = np.array(num_tokens)


# In[45]:


# 平均tokens的长度
np.mean(num_tokens)


# In[46]:


# 最长的评价tokens的长度
np.max(num_tokens)


# In[89]:


plt.hist(np.log(num_tokens), bins = 100)
plt.xlim((0,10))
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()


# In[48]:


# 取tokens平均值并加上两个tokens的标准差，
# 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens


# In[49]:


# 取tokens的长度为236时，大约95%的样本被涵盖
# 我们对长度不足的进行padding，超长的进行修剪
np.sum( num_tokens < max_tokens ) / len(num_tokens)


# **反向tokenize**  
# 我们定义一个function，用来把索引转换成可阅读的文本，这对于debug很重要。

# In[50]:


# 用来将tokens转换为文本
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text


# In[51]:


reverse = reverse_tokens(train_tokens[0])


# 以下可见，训练样本的极性并不是那么精准，比如说下面的样本，对早餐并不满意，但被定义为正面评价，这会迷惑我们的模型，不过我们暂时不对训练样本进行任何修改。

# In[52]:


# 经过tokenize再恢复成文本
# 可见标点符号都没有了
reverse


# In[53]:


# 原始文本
train_texts_orig[0]


# **准备Embedding Matrix**  
# 现在我们来为模型准备embedding matrix（词向量矩阵），根据keras的要求，我们需要准备一个维度为$(numwords, embeddingdim)$的矩阵，num words代表我们使用的词汇的数量，emdedding dimension在我们现在使用的预训练词向量模型中是300，每一个词汇都用一个长度为300的向量表示。  
# 注意我们只选择使用前50k个使用频率最高的词，在这个预训练词向量模型中，一共有260万词汇量，如果全部使用在分类问题上会很浪费计算资源，因为我们的训练样本很小，一共只有4k，如果我们有100k，200k甚至更多的训练样本时，在分类问题上可以考虑减少使用的词汇量。

# In[90]:


embedding_dim


# In[55]:


# 只使用前20000个词
num_words = 50000
# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, embedding_dim))
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 50000 * 300
for i in range(num_words):
    embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')


# In[56]:


# 检查index是否对应，
# 输出300意义为长度为300的embedding向量一一对应
np.sum( cn_model[cn_model.index2word[333]] == embedding_matrix[333] )


# In[57]:


# embedding_matrix的维度，
# 这个维度为keras的要求，后续会在模型中用到
embedding_matrix.shape


# **padding（填充）和truncating（修剪）**  
# 我们把文本转换为tokens（索引）之后，每一串索引的长度并不相等，所以为了方便模型的训练我们需要把索引的长度标准化，上面我们选择了236这个可以涵盖95%训练样本的长度，接下来我们进行padding和truncating，我们一般采用'pre'的方法，这会在文本索引的前面填充0，因为根据一些研究资料中的实践，如果在文本索引后面填充0的话，会对模型造成一些不良影响。

# In[58]:


# 进行padding和truncating， 输入的train_tokens是一个list
# 返回的train_pad是一个numpy array
train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                            padding='pre', truncating='pre')


# In[59]:


# 超出五万个词向量的词用0代替
train_pad[ train_pad>=num_words ] = 0


# In[60]:


# 可见padding之后前面的tokens全变成0，文本在最后面
train_pad[0]


# In[61]:


# 准备target向量，前2000样本为1，后2000为0
train_target = np.concatenate( (np.ones(2000),np.zeros(2000)) )


# In[62]:


# 进行训练和测试样本的分割
from sklearn.model_selection import train_test_split


# In[63]:


# 90%的样本用来训练，剩余10%用来测试
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.1,
                                                    random_state=12)


# In[64]:


# 查看训练样本，确认无误
print(reverse_tokens(X_train[35]))
print('class: ',y_train[35])


# 现在我们用keras搭建LSTM模型，模型的第一层是Embedding层，只有当我们把tokens索引转换为词向量矩阵之后，才可以用神经网络对文本进行处理。
# keras提供了Embedding接口，避免了繁琐的稀疏矩阵操作。   
# 在Embedding层我们输入的矩阵为：$$(batchsize, maxtokens)$$
# 输出矩阵为： $$(batchsize, maxtokens, embeddingdim)$$

# In[65]:


# 用LSTM对样本进行分类
model = Sequential()


# In[66]:


# 模型第一层为embedding
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_tokens,
                    trainable=False))


# In[67]:


model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))


# **构建模型**  
# 我在这个教程中尝试了几种神经网络结构，因为训练样本比较少，所以我们可以尽情尝试，训练过程等待时间并不长：  
# **GRU：**如果使用GRU的话，测试样本可以达到87%的准确率，但我测试自己的文本内容时发现，GRU最后一层激活函数的输出都在0.5左右，说明模型的判断不是很明确，信心比较低，而且经过测试发现模型对于否定句的判断有时会失误，我们期望对于负面样本输出接近0，正面样本接近1而不是都徘徊于0.5之间。  
# **BiLSTM：**测试了LSTM和BiLSTM，发现BiLSTM的表现最好，LSTM的表现略好于GRU，这可能是因为BiLSTM对于比较长的句子结构有更好的记忆，有兴趣的朋友可以深入研究一下。  
# Embedding之后第，一层我们用BiLSTM返回sequences，然后第二层16个单元的LSTM不返回sequences，只返回最终结果，最后是一个全链接层，用sigmoid激活函数输出结果。

# In[68]:


# GRU的代码
# model.add(GRU(units=32, return_sequences=True))
# model.add(GRU(units=16, return_sequences=True))
# model.add(GRU(units=4, return_sequences=False))


# In[69]:


model.add(Dense(1, activation='sigmoid'))
# 我们使用adam以0.001的learning rate进行优化
optimizer = Adam(lr=1e-3)


# In[70]:


model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[71]:


# 我们来看一下模型的结构，一共90k左右可训练的变量
model.summary()


# In[72]:


# 建立一个权重的存储点
path_checkpoint = 'sentiment_checkpoint.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                      verbose=1, save_weights_only=True,
                                      save_best_only=True)


# In[73]:


# 尝试加载已训练模型
try:
    model.load_weights(path_checkpoint)
except Exception as e:
    print(e)


# In[74]:


# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)


# In[75]:


# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1, min_lr=1e-5, patience=0,
                                       verbose=1)


# In[76]:


# 定义callback函数
callbacks = [
    earlystopping, 
    checkpoint,
    lr_reduction
]


# In[77]:


# 开始训练
model.fit(X_train, y_train,
          validation_split=0.1, 
          epochs=20,
          batch_size=128,
          callbacks=callbacks)


# **结论**  
# 我们首先对测试样本进行预测，得到了还算满意的准确度。  
# 之后我们定义一个预测函数，来预测输入的文本的极性，可见模型对于否定句和一些简单的逻辑结构都可以进行准确的判断。

# In[78]:


result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))


# In[79]:


def predict_sentiment(text):
    print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                           padding='pre', truncating='pre')
    # 预测
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)


# In[80]:


test_list = [
    '酒店设施不是新的，服务态度很不好',
    '酒店卫生条件非常不好',
    '床铺非常舒适',
    '房间很凉，不给开暖气',
    '房间很凉爽，空调冷气很足',
    '酒店环境不好，住宿体验很不好',
    '房间隔音不到位' ,
    '晚上回来发现没有打扫卫生',
    '因为过节所以要我临时加钱，比团购的价格贵'
]
for text in test_list:
    predict_sentiment(text)


# **错误分类的文本**
# 经过查看，发现错误分类的文本的含义大多比较含糊，就算人类也不容易判断极性，如index为101的这个句子，好像没有一点满意的成分，但这例子评价在训练样本中被标记成为了正面评价，而我们的模型做出的负面评价的预测似乎是合理的。

# In[81]:


y_pred = model.predict(X_test)
y_pred = y_pred.T[0]
y_pred = [1 if p>= 0.5 else 0 for p in y_pred]
y_pred = np.array(y_pred)


# In[82]:


y_actual = np.array(y_test)


# In[83]:


# 找出错误分类的索引
misclassified = np.where( y_pred != y_actual )[0]


# In[92]:


# 输出所有错误分类的索引
len(misclassified)
print(len(X_test))


# In[85]:


# 我们来找出错误分类的样本看看
idx=101
print(reverse_tokens(X_test[idx]))
print('预测的分类', y_pred[idx])
print('实际的分类', y_actual[idx])


# In[86]:


idx=1
print(reverse_tokens(X_test[idx]))
print('预测的分类', y_pred[idx])
print('实际的分类', y_actual[idx])

