#!/usr/bin/env python
# coding: utf-8

# # MACD简介
# 
# MACD（Moving Average Convergence and Divergence)是Geral Appel 于1979年提出的，利用收盘价的短期（常用为12日）指数移动平均线与长期（常用为26日）指数移动平均线之间的聚合与分离状况，对买进、卖出时机作出研判的技术指标。
# 
# 如下图：

# ![image.png](attachment:image.png)

# ## MACD指标的原理
# 
# MACD指标是根据均线的构造原理，对股票价格的收盘价进行平滑处理，求出算术平均值以后再进行计算，是一种趋向类指标。
# 
# MACD指标是运用快速（短期）和慢速（长期）移动平均线及其聚合与分离的征兆，加以双重平滑运算。而根据移动平均线原理发展出来的MACD，一则去除了移动平均线频繁发出假信号的缺陷，二则保留了移动平均线的效果，因此，MACD指标具有均线趋势性、稳重性、安定性等特点，是用来研判买卖股票的时机，预测股票价格涨跌的技术分析指标 。
# 
# MACD指标主要是通过EMA、DIF和DEA（或叫MACD、DEM）这三值之间关系的研判，DIF和DEA连接起来的移动平均线的研判以及DIF减去 DEM值而绘制成的柱状图（BAR）的研判等来分析判断行情，预测股价中短期趋势的主要的股市技术分析指标。其中，DIF是核心，DEA是辅助。DIF是快速平滑移动平均线（EMA1）和慢速平滑移动平均线（EMA2）的差。BAR柱状图在股市技术软件上是用红柱和绿柱的收缩来研判行情。
# 
# ## MACD指标的计算方法
# 
# MACD 是根据移动平均线较易掌握趋势变动的方向之优点所发展出来的，它是利用二条不同速度（一条变动的速率快──短期的移动平均线，另一条较慢──长期的移动平 均线）的指数平滑移动平均线来计算二者之间的差离状况（DIF）作为研判行情的基础，然后再求取其DIF之9日平滑移动平均线（MACD）记Signal。MACD实际就是运用快速与慢速移动平均线聚合与分离的征兆，来研判买进与卖进的时机和讯号。
# 
# 
# ### 计算方法
# 1. 计算平滑系数
# 
# > MACD一个最大的长处，即在于其指标的平滑移动，特别是对一某些剧烈波动的市场，这种平滑移动的特性能够对价格波动作较和缓的描绘，从而大为提高资料的实用性。不过，在计算EMA前，首先必须求得平滑系数。所谓的系数，则是移动平均周期之单位数，如几天，几周等等。其公式如下：
# > 平滑系数＝2÷（周期单位数＋1 ）
# > 如12日EMA的平滑系数＝2÷（12＋1）＝0．1538；
# > 26日EMA平滑系数为=2÷27＝0．0741
# 
# 2. 计算指数平均值（EMA）
# 
# > 一旦求得平滑系数后，即可用于EMA之运算，公式如下：
# > 今天的指数平均值＝平滑系数×（今天收盘指数－昨天的指数平均值）＋昨天的指数平均值。
# 
# > 依公式可计算出12日EMA
# > 12日EMA＝2÷13×（今天收盘指数一昨天的指数平均值）＋昨天的指数平均值＝（2÷13）×今天收盘指数＋（11÷13）×昨天的指数平均值。
# 
# > 同理，26日EMA亦可计算出：
# > 26日EMA＝（2÷27）×今天收盘指数＋（25÷27）×昨天的指数平均值。
# 
# > 由于每日行情震荡波动之大小不同，并不适合以每日之收盘价来计算移动平均值，于是有需求指数（Demand Index）之产生，乃轻需求指数代表每日的收盘指数。计算时，都分别加重最近一日的份量权数（两倍），即对较近的资料赋予较大的权值，其计算方法如下：
# > DI＝（C×2＋H＋L）÷4
# 
# > 其中，C为收盘价，H为最高价，L为最低价。
# > 所以，上列公式中之今天收盘指数，可以需求指数来替代。
# 
# 3. 计算指数平均的初值
# 
# 当开始要对指数平均值，作持续性的记录时，可以将第一天的收盘价或需求指数当作指数平均的初值。若要更精确一些，则可把最近几天的收盘价或需求指数平均，以其平均价位作为初值。此外。亦可依其所选定的周期单位数，来做为计算平均值的基期数据。
# 
# ## MACD指标的一般研判标准
# 
# > MACD指标是市场上绝大多数投资者熟知的分析工具，但在具体运用时，投资者可能会觉得MACD指标的运用的准确性、实效性、可操作性上有很多茫然的地方，有时会发现用从书上学来的MACD指标的分析方法和技巧去研判股票走势，所得出的结论往往和实际走势存在着特别大的差异，甚至会得出相反的结果。这其中的主要原因是市场上绝大多数论述股市技术分析的书中关于MACD的论述只局限在表面的层次，只介绍MACD的一般分析原理和方法，而对MACD分析指标的一些特定的内涵和分析技巧的介绍鲜有涉及。
# 
# > MACD指标的一般研判标准主要是围绕快速和慢速两条均线及红、绿柱线状况和它们的形态展开。一般分析方法主要包括DIF指标和MACD值及它们所处的位置、DIF和MACD的交叉情况、红柱状的收缩情况和MACD图形的形态这四个大的方面分析。
# 
# 1. DIF和MACD的值及线的位置
#     1. 当DIF和MACD均大于0（即在图形上表示为它们处于零线以上）并向上移动时，一般表示为股市处于多头行情中，可以买入或持股；
# 
#     2. 当DIF和MACD均小于0（即在图形上表示为它们处于零线以下）并向下移动时，一般表示为股市处于空头行情中，可以卖出股票或观望。
# 
#     3. 当DIF和MACD均大于0（即在图形上表示为它们处于零线以上）但都向下移动时，一般表示为股票行情处于退潮阶段，股票将下跌，可以卖出股票和观望；
# 
#     4. 当DIF和MACD均小于0时（即在图形上表示为它们处于零线以下）但向上移动时，一般表示为行情即将启动，股票将上涨，可以买进股票或持股待涨。
# 
# 2. DIF和MACD的交叉情况
# 
#     1. 当DIF与MACD都在零线以上，而DIF向上突破MACD时，表明股市处于一种强势之中，股价将再次上涨，可以加码买进股票或持股待涨，这就是MACD指标“黄金交叉”的一种形式。
# 
#     2. 当DIF和MACD都在零线以下，而DIF向上突破MACD时，表明股市即将转强，股价跌势已尽将止跌朝上，可以开始买进股票或持股，这是MACD指标“黄金交叉”的另一种形式。
# 
#     3. 当DIF与MACD都在零线以上，而DIF却向下突破MACD时，表明股市即将由强势转为弱势，股价将大跌，这时应卖出大部分股票而不能买股票，这就是MACD指标的“死亡交叉”的一种形式。
# 
#     4. 当DIF和MACD都在零线以上，而DIF向下突破MACD时，表明股市将再次进入极度弱市中，股价还将下跌，可以再卖出股票或观望，这是MACD指标“死亡交叉”的另一种形式。
# 
# 3. MACD指标中的柱状图分析
# 
#     1. 当红柱状持续放大时，表明股市处于牛市行情中，股价将继续上涨，这时应持股待涨或短线买入股票，直到红柱无法再放大时才考虑卖出。
# 
#     2. 当绿柱状持续放大时，表明股市处于熊市行情之中，股价将继续下跌，这时应持币观望或卖出股票，直到绿柱开始缩小时才可以考虑少量买入股票。
# 
#     3. 当红柱状开始缩小时，表明股市牛市即将结束（或要进入调整期），股价将大幅下跌，这时应卖出大部分股票而不能买入股票。
# 
#     4. 当绿柱状开始收缩时，表明股市的大跌行情即将结束，股价将止跌向上（或进入盘整），这时可以少量进行长期战略建仓而不要轻易卖出股票。
# 
#     5. 当红柱开始消失、绿柱开始放出时，这是股市转市信号之一，表明股市的上涨行情（或高位盘整行情）即将结束，股价将开始加速下跌，这时应开始卖出大部分股票而不能买入股票。
# 
#     6. 当绿柱开始消失、红柱开始放出时，这也是股市转市信号之一，表明股市的下跌行情（或低位盘整）已经结束，股价将开始加速上升，这时应开始加码买入股票或持股待涨。

# # 先说一下写代码的思路
# 
# - 需要使用talib模块中的MACDEXT函数返回macd对应的dif, dea, macd信息，封装好的方法为`MACD(close, fastperiod, slowperiod, signalperiod)`
# - 调用的标的是经常变化的，时间周期也是变化的，所以将对应标的信息的代码单独封装，封装好的方法为`get_macd(stock, count, end_date, unit)`
# - 为了查看macd的信息，将其图例化将是再好不过的，所谓一图胜万字,封装的方法为`show_macd(macd_list)`
# - 另外，使用macd信息的时候，最关心的点是dif与dea金叉死叉情况，这里也给出了方法，并与直接在图例上进行展示，封装的方法为`show_cross(macd_list)`

# # 导入必要的模块

# In[1]:


from jqdata import *
import talib as tl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


# # 封装公用函数

# In[2]:


# MACD 公用函数
def MACD(close, fastperiod, slowperiod, signalperiod):
    macdDIFF, macdDEA, macd = tl.MACDEXT(close, fastperiod=fastperiod, fastmatype=1, slowperiod=slowperiod, slowmatype=1, signalperiod=signalperiod, signalmatype=1)
    macd = macd * 2
    return macdDIFF, macdDEA, macd 

# 查寻一个时间段内某标的的macd信息
def get_macd(stock, count, end_date, unit):
    data = get_bars(security=stock, count=count, unit=unit,
                        include_now=False, 
                        end_dt=end_date, fq_ref_date=None)
    close = data['close']
    open = data['open']
    high = data['high']
    low = data['low']

    return MACD(close, 12, 26, 9)

# 将macd信息图例化
def show_macd(macd_list):
    macdDIFF, macdDEA, macd = macd_list
    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签  
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

    plt.figure(figsize=(20, 5))
    plt.title('MACD 金叉死叉示例图')
    plt.plot(macdDEA)
    plt.plot(macdDIFF)
    
    for i in range(0, len(macd)):
        plt.bar(i, macd[i], color='r' if macd[i] > 0 else 'g')
        
    plt.legend(['DEA', 'DIF'], loc=2)
    plt.show()

# 将macd的金叉与死叉
def show_cross(macd_list):
    macdDIFF, macdDEA, macd = macd_list
    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签  
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

    plt.figure(figsize=(20, 5))
    plt.title('MACD 金叉死叉示例图')
    plt.plot(macdDEA)
    plt.plot(macdDIFF)
    
    for i in range(0, len(macd)):
        plt.bar(i, macd[i], color='r' if macd[i] > 0 else 'g')
        
    for i in range(4, len(macd)):
        if macdDIFF[i-1] < macdDEA[i-1] and macdDIFF[i] > macdDEA[i]:
            # 用红圈标出金叉
            plt.scatter(i, macdDEA[i], color='', marker='o', edgecolors='r', s=500, linewidths=3)
        elif macdDIFF[i-1] > macdDEA[i-1] and macdDIFF[i] < macdDEA[i]:
            # 用绿圈标出死叉
            plt.scatter(i, macdDEA[i], color='', marker='o', edgecolors='g', s=500, linewidths=3)
            
    plt.legend(['DEA', 'DIF'], loc=2)
    plt.show()
    
# 将macd的0轴下的二次金叉显示出来
def show_two_low_cross(macd_list):
    macdDIFF, macdDEA, macd = macd_list
    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签  
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

    plt.figure(figsize=(20, 5))
    plt.title('MACD 金叉死叉示例图')
    plt.plot(macdDEA)
    plt.plot(macdDIFF)
    
    for i in range(0, len(macd)):
        plt.bar(i, macd[i], color='r' if macd[i] > 0 else 'g')
    
    cross_list = []
    for i in range(4, len(macd)):
        if macdDIFF[i-1] < macdDEA[i-1] and macdDIFF[i] > macdDEA[i]:
            gold_cross = {'id':i,
                          'name':'gold_cross',
                          'dif':macdDIFF[i],
                          'dea':macdDEA[i]}
            cross_list.append(gold_cross)
        elif macdDIFF[i-1] > macdDEA[i-1] and macdDIFF[i] < macdDEA[i]:
            death_cross = {'id':i,
                          'name':'death_cross',
                          'dif':macdDIFF[i],
                          'dea':macdDEA[i]}
            cross_list.append(death_cross)
            
    df = pd.DataFrame(cross_list, columns=['id', 'name', 'dif', 'dea']).sort_values(by="id",ascending=False)  
    for index in df.index:
        if index < 3:
            break
        if df.loc[index]['name']=='gold_cross' and df.loc[index]['dif']<0:
            if df.loc[index-1]['name']=='death_cross' and df.loc[index-1]['dif']<0:
                if df.loc[index-2]['name']=='gold_cross' and df.loc[index-2]['dif']<0:
                    if df.loc[index-3]['name']=='death_cross' and df.loc[index-3]['dif']>0:
                        plt.scatter(df.loc[index-2]['id'], df.loc[index-2]['dea'], color='', marker='o', edgecolors='r', s=500, linewidths=3)
                        plt.scatter(df.loc[index]['id'], df.loc[index]['dea'], color='', marker='o', edgecolors='r', s=500, linewidths=3)  
    
    plt.legend(['DEA', 'DIF'], loc=2)
    plt.show()
    
# 将macd的0轴下多次金叉显示出来
def show_low_cross(macd_list):
    macdDIFF, macdDEA, macd = macd_list
    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签  
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

    plt.figure(figsize=(20, 5))
    plt.title('MACD 金叉死叉示例图')
    plt.plot(macdDEA)
    plt.plot(macdDIFF)
    
    for i in range(0, len(macd)):
        plt.bar(i, macd[i], color='r' if macd[i] > 0 else 'g')
    
    cross_list = []
    for i in range(4, len(macd)):
        if macdDIFF[i-1] < macdDEA[i-1] and macdDIFF[i] > macdDEA[i]:
            gold_cross = {'id':i,
                          'name':'gold_cross',
                          'dif':macdDIFF[i],
                          'dea':macdDEA[i]}
            cross_list.append(gold_cross)
        elif macdDIFF[i-1] > macdDEA[i-1] and macdDIFF[i] < macdDEA[i]:
            death_cross = {'id':i,
                          'name':'death_cross',
                          'dif':macdDIFF[i],
                          'dea':macdDEA[i]}
            cross_list.append(death_cross)
            
    df = pd.DataFrame(cross_list, columns=['id', 'name', 'dif', 'dea']).sort_values(by="id",ascending=False)  
    for index in df.index:
        if index < 2:
            break
        if df.loc[index]['name']=='gold_cross' and df.loc[index]['dif'] < 0:
            if df.loc[index-1]['name']=='death_cross' and df.loc[index-1]['dif'] < 0:
                        plt.scatter(df.loc[index]['id'], df.loc[index]['dea'], color='', marker='o', edgecolors='r', s=500, linewidths=3)  
    
    plt.legend(['DEA', 'DIF'], loc=2)
    plt.show()
    

# 判断是否是低们二金叉
def is_second_low_gold_cross(macd_list):
    macdDIFF, macdDEA, macd = macd_list
    
    cross_list = []
    # 判断当前是否发生金叉，并且位置在0轴之下
    if macdDIFF[-2] < macdDEA[-2] and macdDIFF[-1] > macdDEA[-1] and macdDIFF[-1] < 0:
        for i in range(4, len(macd)):
            if macdDIFF[i-1] < macdDEA[i-1] and macdDIFF[i] > macdDEA[i]:
                gold_cross = {'id':i,
                              'name':'gold_cross',
                              'dif':macdDIFF[i],
                              'dea':macdDEA[i]}
                cross_list.append(gold_cross)
            elif macdDIFF[i-1] > macdDEA[i-1] and macdDIFF[i] < macdDEA[i]:
                death_cross = {'id':i,
                              'name':'death_cross',
                              'dif':macdDIFF[i],
                              'dea':macdDEA[i]}
                cross_list.append(death_cross)

        df = pd.DataFrame(cross_list, columns=['id', 'name', 'dif', 'dea']).sort_values(by="id",ascending=False)
        index_list = []
        if len(df.index) > 4:
            index_list = df.index[0:4] 
        else:
            return False
            
        result1 = df.loc[index_list[0]]['name']=='gold_cross' and df.loc[index_list[0]]['dif'] < 0
        result2 = df.loc[index_list[1]]['name']=='death_cross' and df.loc[index_list[1]]['dif'] < 0
        result3= df.loc[index_list[2]]['name']=='gold_cross' and df.loc[index_list[2]]['dif'] < 0
        result4 = df.loc[index_list[3]]['name']=='death_cross' and df.loc[index_list[3]]['dif'] > 0                                            
        if result1 and result2 and result3 and result4:
            return True
        else:
            return False
    else:
        return False      


# # 日级另MACD信息展示

# In[3]:


print('获取一只标的的日级别macd信息')
macd_day = get_macd('000001.XSHE', 200, datetime.datetime.now().date(), '1d')
print('最近5条DIF', macd_day[0][-5:])
print('最近5条DEA', macd_day[1][-5:])
print('最近5条MACD', macd_day[2][-5:])


# In[4]:


print('图例展示')
show_macd(macd_day)


# In[5]:


print('金叉死叉展示')
show_cross(macd_day)


# # 周级另MACD信息展示

# In[6]:


print('获取一只标的的周级别macd信息')
macd_week = get_macd('000001.XSHE', 200, datetime.datetime.now().date(), '1w')
print('最近5条DIF', macd_week[0][-5:])
print('最近5条DEA', macd_week[1][-5:])
print('最近5条MACD', macd_week[2][-5:])


# In[7]:


print('图例展示')
show_macd(macd_week)


# In[8]:


print('金叉死叉展示')
show_cross(macd_week)


# # 月级另MACD信息展示

# In[9]:


print('获取一只标的的月级别macd信息')
macd_month = get_macd('000001.XSHE', 200, datetime.datetime.now().date(), '1M')
print('最近5条DIF', macd_month[0][-5:])
print('最近5条DEA', macd_month[1][-5:])
print('最近5条MACD', macd_month[2][-5:])


# In[10]:


print('图例展示')
show_macd(macd_month)


# In[11]:


print('金叉死叉展示')
show_cross(macd_month)


# # 分钟级另MACD信息展示

# In[12]:


print('获取一只标的的60分钟级别macd信息')
macd_60m = get_macd('000001.XSHE', 200, datetime.datetime.now().date(), '60m')
print('最近5条DIF', macd_60m[0][-5:])
print('最近5条DEA', macd_60m[1][-5:])
print('最近5条MACD', macd_60m[2][-5:])


# In[13]:


print('图例展示')
show_macd(macd_60m)


# In[14]:


print('金叉死叉展示')
show_cross(macd_60m)


# ## 还可以将周、日级别的MACD同时展示，查看级别之间的共振情况

# In[15]:


macd_week = get_macd('000001.XSHE', 50, datetime.datetime.now().date(), '1w')
macd_day = get_macd('000001.XSHE', 50*5, datetime.datetime.now().date(), '1d')
show_cross(macd_week)
show_cross(macd_day)


# # 示例：求0轴下二次金叉

# In[16]:


macd_week = get_macd('000001.XSHE', 200, datetime.datetime.now().date(), '1w')
show_two_low_cross(macd_week)
value = is_second_low_gold_cross(macd_week)
if value:
    print('当前金叉是0轴下二次金叉')
else:
    print('当前金叉不是0轴下二次金叉')


# In[ ]:




