#!/usr/bin/env python
# coding: utf-8

# # TensorFlow 学习之循环神经网络（RNN）

# 近期 [JoinQuant 金融终端](https://www.joinquant.com/default/index/jqclient)上线了**python3.6的版本**，并且为小伙伴们带来了诸多重要更新：
# 
# - 支持 pip 一键安装 TensorFlow、PyTorch、Keras 等深度学习库；
# - 研究示例文件增加了 TensorFlow、PyTorch 的安装教程；
# - 支持 Tick 回测功能；
# - 期权数据；
# - 组合优化更新：支持风险因子暴露限制、换手率限制、流动性限制、流动性限制、行业偏离度限制、追踪误差限制、换手率限制等；
# 
# ### 温馨提示：可以下载该文件，并在客户端的投资研究内直接运行
# 
# **目录：**
# 
# 1. 循环神经网络简介
# 2. 简单的循环神经网络
# 3. 用TensorFlow构建简单地循环神经网络
# 4. 各种不同的循环神经网络
# 
# 

# ## 1. 循环神经网络简介
# 
# **循环神经网络**（Recurrent Neural Network, RNN）是一类用于处理序列数据的神经网络。就像卷积网络是专门用于处理网格化数据（像一个图像），循环神经网络是专门用于处理序列数据的。
# 
# P.S. 为了方便理解，该篇文章尽量的删减了所需公式部分！
# 
# 讲到循环神经网络就不得不提起神经网络，而两者最大的不同就是循环神经网络可以做到根据过去做到未来。我们可以根据不同的训练准则，选择性的精确保留过去序列的某些方面。下图为神经网络与循环神经网络的对比。
# 
# ![1.png-15.1kB](https://image.joinquant.com/205241ae180cee6867a0684951bb1af4)
# 
# 在图中可以看到一个普通的神经网络就是将输入x通过函数f的计算再到状态h。而循环神经网络顾名思义就是一个循环过程，这个循环网络只处理来自输入X的信息，将其合并到经过时间向前传播的状态h，*f*是数据的处理函数。可以发现当前状态是可以影响其未来的状态。

# ## 2. 简单的循环神经网络
# 
# 我们将上面那个没有输出的循环神经网络扩展成一个完整的循环神经网络
# 
# ![2.png-24.2kB](https://image.joinquant.com/9aad1484da034dd09edb6149bf48de80)
# 
# 在图中损失L衡量每个${\hat y}$与相应的训练目标y的距离。从图中可以看出模型为向前传播。
# 
# 为了更好地解释上面的循环神经网络，我们先假设一些东西。在上图中没有指定隐藏单元的激活函数，我们假设激活函数为$f$。RNN从特定的起始状态$h^{(0)}$开始向前传播。从$t = 1$到$t = \tau$的每个时间步，我们应用以下更新方程：
# 
# $$
# \begin{array}{l}
# {a^{(t)}} = b + W{h^{(t - 1)}} + U{x^{(t)}}\\
# {h^{(t)}} = f({a^{(t)}})\\
# {{\hat y}^{(t)}} = c + V{h^{(t)}}\\
# \end{array}
# $$
# 
# 其中的参数的偏执向量b和c联通权重矩阵 **U、V、W**，分别对应于输入到隐藏、隐藏到输出和隐藏到隐藏的链接的权重。通过上面的公式就可以计算出估计的y值，但是上面这些权重到底该取多少合适，我们并不知道，所以这个时候就需要通过我们上述估计出的y和真实的y做对比，之后通过设置学习率和选择优化器来重新调整 **U、V、W**，一直重复这个过程直到误差符合要求为止。这个过程和普通的神经网络是大致相同的。

# ## 3. 用TensorFlow构建简单地循环神经网络
# 
# TensorFlow 是计算机图模型，即用图的形式来表示运算过程的一种模型。大概来说就是我们需要在图中先画出来各个参数设定，设定好之后再运行它。对于 TensorFlow 图里面的各个元素来说，设定好它是并不运行的，直到用命令运行图中的某个参数我们才会运行。
# 
# **下面是用 TensorFlow 建立一个循环神经网络模型，查看模型训练的效果。**
# 
# 以沪深300指数为例，我们想要预测下一天的指数收盘价。具体思路就是为了精准的预测输出值，就需要一个多次重复的一个过程，先随机的指定权重和偏置，经过一次次的循环，更新权重和偏置，最后找到最优的值来确定输出。其中每一次的计算流程如下图所示，我们希望得到L^{(t)},之后再根据其值得大小调整权重，最后得到训练的结果：
# 
# 
# ![3.png-39.8kB](https://image.joinquant.com/2ecb28f519363d61d7fe9a11807edba8)
# 
# 注：在用 TensorFlow 最重要的一点就是对数据维度、形状的把控~一不小心就会出现问题！
# 
# 第一步，我们需要载入用到的包。

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
#函数用于清除默认图形堆栈并重置全局默认图形
tf.reset_default_graph()


# 第二步，对输入值和输出值进行设定。

# In[2]:


#数据定义
#输入值x：以当天沪深300指数的开盘价、收盘价、最高价、最低价为输入数据
x_temp=get_price('000300.XSHG',start_date='2018-01-01',end_date='2018-12-31')[0:-1].iloc[:,:-2]
#将x调整为TensorFlow的格式（由于tf处理数据的格式需要改成float32）
x=tf.Variable(np.array(x_temp).astype(np.float32))
#输出值y：以第二天沪深300指数的收盘价作为输出数据
y_temp=get_price('000300.XSHG',start_date='2018-01-01',end_date='2018-12-31')[1:]['close']
#将y调整为TensorFlow的格式
y_temp=pd.DataFrame(y_temp)
y=tf.Variable(np.array(y_temp).astype(np.float32))


# 第三步：定义循环神经网络的构成图与一些参数
# 
# 1、隐藏层的层数：
# ![4.png-46.1kB](https://image.joinquant.com/00f80c4ffefec9f85ceb7d6697b72392)
# 
# 在图中显示为1层，但实际上层数多一点学习效果更好，可以想象为一个黑箱，这里设置为128层。
# 
# 2、学习率：学习率就是根据误差大小，选择学习的比率。这里设置为0.1

# In[3]:


#设置学习率为0.1
lr=0.1
#设置隐藏层的层数为128
hidden_units=128
#输入参数的维度
x_inputs=len(x_temp.columns)
#输入数据的长度
n_step=len(x_temp.index)
#输出的维度
y_output=len(y_temp.columns)


# 第四步：随机定义各个权重，这里需要设置的有x输入的权重U和输出的权重V，隐藏层的权重W可以不用自己设定

# In[4]:


#设置权重，权重为随机数
weights={
    'U':tf.Variable(tf.constant(0.25,shape=[x_inputs, y_output])),
    'V':tf.Variable(tf.random_normal([hidden_units, n_step])),
}
biases={
    'U':tf.Variable(tf.constant(0.0, shape=[n_step,y_output])),
    'V':tf.Variable(tf.constant(0.0, shape=[n_step,])),
}


# 第五步：计算的前期准备已经OK拉，接下来可以进行计算了~按照上面图中的步骤：
# 
# （1）计算图中输入数据进行加权处理后得到的数据
# 
# $ a(t)=b+Wh^{(t-1)}+Ux^{t} $ 根据权重和偏执的设置，用此公式得出的是x最终的输入数据

# In[5]:


x_in = tf.matmul(x, weights['U'])+biases['U']
x_in = tf.reshape(x_in, [-1, n_step, y_output])


# （2）定义隐藏层的处理方式和计算出输出值

# In[6]:


#定义隐藏层的处理方式（这里的函数是可以自行选择的~可以多去探索一下呦）
cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
#计算出隐藏层中每层的结果
outputs, states = tf.nn.dynamic_rnn(cell, x_in, dtype=tf.float32)
#outputs记录了隐藏层中每一层的输出，但是如果要计算出最后的输出，只需要最后一层即可
output = tf.transpose(outputs, [1,0,2])[-1]
#根据权重得到对y的估计值
results=tf.matmul(output, weights['V']) + biases['V']    


# （3）求出预测的误差，再根据误差大小和学习率用优化器调整权重设置。

# In[7]:


#算误差
cost=tf.reduce_mean(tf.square(results-y))
#根据误差大小和学习率用优化器调整权重设置
train_op = tf.train.AdamOptimizer(0.1).minimize(cost)


# 第六步：根据多遍运行，查看误差

# In[ ]:


#初始化
init = tf.global_variables_initializer()
#构建一个图
sess = tf.Session()

#先把图初始化
sess.run(init)
#运行30000次来调整权重，每1000次打印出误差和预测的y值
for i in range(30000):
    sess.run(train_op)
    if i % 1000 == 0:
        #由于学习出的结果过多，这里只显示最后一个值
        print("误差:",sess.run(cost),"，预测值:",sess.run(results[-1][-1]))

print("训练完成")


# 训练完成，我们可以看出，训练误差已经是非常的小了，这证明循环神经网络的学习能力还是很强的！我们可以通过调整里面的参数来建立自己的循环神经网络模型。
# 
# **综上所述这里提供了一个经过循环神经网络算法来预测股票收盘价的模型训练过程，提供了该算法的思路流程。这里并没有使用模型预测的步骤，文章只是想讲解RNN，因此未包含模型预测部分，请知悉~**
# 
# P.S. 注意！在实际应用中，训练中的误差不是越小越好~有的时候训练时样本误差很小，但到实验组时就会出现预测效率大大降低。这是由于学习过度也就是过拟合！所以说一定要掌握好一个合适的度！

# ## 4. 各种不同的循环神经网络
# 
# 所以你是不是以为循环神经网络就是这么简单么...NONONO！循环神经网络可以再各处进行循环，例如：
# ![7.png-26.9kB](https://image.joinquant.com/11b2a250896246381739068b1620173b)
# 这类RNN的唯一循环是从输出到隐藏层的反馈链接。
# 
# ![8.png-12.1kB](https://image.joinquant.com/8788a762b9f032d2e2fabcfbffe64c5e)
# 
# 这类 RNN 是在时间序列结束时，具有单个输出。
# 
# RNN 有很多不同的种类，我们可以根据需要自己构建不同的 RNN 的模型。
# 
# 循环神经网络就到这里就结束了！欢迎指正批评！

# In[ ]:




