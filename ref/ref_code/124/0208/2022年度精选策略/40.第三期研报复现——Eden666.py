#!/usr/bin/env python
# coding: utf-8

# # 引言

# ## 研究目的

# ### 参考国泰君安证券研报《20181128-基于CCK模型的股票市场羊群效应研究》，对市场上存在的个别股票的涨跌引起相关股票收益率联动的现象进行分析探究。根据研报构建CCK模型，并进行改良，寻找更多联动信号，并正确分析市场趋势。

# ## 研究思路

# ### 1.根据研究报告，计算市场回报率并进行改良
# ### 2.在CCK模型的基础上，增加过滤指标进行分析和改良
# ### 3.通过改良后，并在此基础上进行回测分析。

# ## 研究结论

# ### 1.在本文进行改良后，模型信号出现的次数更多，但是纯度下降了。
# ### 2.不管标的指数是宽基还是行业，做多策略都明显优于做空策略，但是策略的收益率不是很出众。
# ### 3.与研报相同的是，策略有效性和标的指数的市值风格和风格纯度有关，市值越高出效果越好。
# ### 4.通过回测判断，板块间确实存在羊群效应，而且CCK模型也能很好分析出羊群效应出现的时间点，但是不足的是在区分市场方向上并不是很好，甚至会出现错误区分市场方向导致大额亏损的状况
# ### 5.最后，羊群效应确实如研报所言，确实存在，不过本文认为羊群效应不管多空都发生在短期，且人为区分信号所反映的市场方向更好
# 

# 本文中所涉及的A股综合日市场回报率指标资料源自国泰安数据库，以下是国泰安数据库(CSMAR)的网址
# 
# http://www.gtarsc.com/Login/index?ReturnUrl=%2fHome%2fIndex  
# 
# 下文代码中涉及国泰安数据的都会用CSMAR进行前缀标记
# 
# （下文中提到的CSMAR的市场回报率全称为）
#      
#     （——考虑现金红利再投资的A股综合日市场回报率（流通市值加权平均法））

# In[207]:


from jqdata import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import *
import time
import statsmodels.api as sm 
from sklearn.preprocessing import scale
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# In[208]:


#展示CSMAR的A股综合日市场回报率表格
TRD = pd.read_excel('TRD20090101-20101231.xlsx')
CSMAR_market = TRD[TRD['综合市场类型']==5]
CSMAR_market.index= CSMAR_market.datetime
del CSMAR_market['Market_return_rate.1']
CSMAR_market_sample = CSMAR_market[(CSMAR_market.datetime>'2010-10-12'
                                   )&(CSMAR_market.datetime<='2010-11-06')
                                  ].sort_values('datetime',ascending=True)
print('国泰安数据库提供的考虑现金红利再投资的A股综合日市场回报率（流通市值加权平均法）')
print('')
display(CSMAR_market_sample.tail())


#     国泰安的数据库仅能免费提供2009年1月1日到2010年12月31日综合日市场回报率的数据，在文章仅在刚开始引用国泰安库的A股综合日市场的市场回报率。如果要购买所有年份的数据的话至少要543元才能获得，基于本文作者穷苦生活现状分析，购买这一指标并不现实所有改为由自己构建指标进行回测。
# 

# In[110]:


###展示医药生物指数收盘价、相关龙头股收盘价及离散程度
t0 = time.time()
"获取申万一级医药生物行业在10年11月6号之前的成份分股代码"
pharmaceutical_industry_stock = get_industry_stocks('801150',date='2010-11-06')

###根据成份股，获取出该段时间内每日每股的股价###
pharmaceutical_industry = pd.DataFrame([])
for i in pharmaceutical_industry_stock:
    close = get_price(i,start_date='2010-10-13',end_date='2010-11-06',fields='close')
    pharmaceutical_industry[i]=close.close
print('')
print('医药生物行业在10年10月13到11月06这段时间内每股每日的收盘价')
display(pharmaceutical_industry.tail())

    
"获取医药生物龙头股通化东宝收盘价"
A600867 = get_price('600867.XSHG',start_date='2010-10-13',end_date='2010-11-06')
CriValue_600867=finance.run_query(query(finance.SW1_DAILY_PRICE
                          ).filter(finance.SW1_DAILY_PRICE.code=='801150',
                                    finance.SW1_DAILY_PRICE.date < '2010-11-06',
                                    finance.SW1_DAILY_PRICE.date > '2010-10-12'))
###统一日期索引###
CriValue_600867.index=A600867.index  
    
    
"计算医药生物板块指数与市场回报率的截面绝对离散程度"

###利用申万一级行业的医药生物板块的所有成分股，先输出每只成分股的收益率数据，再进行绝对加总运算。

###如果仅使用医药生物行业的指数收益率作为T时刻股票组合的截面收益率的话，会收到指数加权算法的影响，
###导致较大的误差

pharmaceutical_industry_rate = pharmaceutical_industry.pct_change(1).fillna(0)###获取医药生物板块横截面收益率
CSAD_list = pd.DataFrame([0]*len(pharmaceutical_industry_rate))
for i in range(len(pharmaceutical_industry_rate)):
    CSAD = abs(pharmaceutical_industry_rate.iloc[i,:]-CSMAR_market_sample.Market_return_rate[i]).sum(
                                )/len(pharmaceutical_industry_rate.iloc[i,:])
    CSAD_list.iloc[i,:] = CSAD
    
    
###运用sklearn的模块进行归一化###
A600867_close = scale(A600867.close)
pharmaceutical_industry = scale(CriValue_600867.close)
CSAD = scale(CSAD_list)
market_sample_rate = scale(CSMAR_market_sample.Market_return_rate)

dataframe = pd.DataFrame([A600867_close,pharmaceutical_industry,CSAD
                 ],index=['A600867_close','pharmaceutical_industry','CSAD']).T

###统一日期索引###
dataframe.index=CriValue_600867.index 
print('')
print('医药生物指数收盘价、相关龙头股收盘价及离散程度的数据表格')
display(dataframe.tail())
    

t1 = time.time()
print('获取数据完成 耗时 %s 秒' %round((t1-t0),3))  
print('')
fig = plt.subplots(figsize=(13,8))
plt.plot(dataframe.A600867_close,'r')
plt.plot(dataframe.pharmaceutical_industry,'blue')
plt.plot(dataframe.CSAD,'g')
plt.grid(True)
plt.title('医药生物行业指数收盘价及相关龙头股票收盘价',fontsize=20)
plt.legend(['A600867_close','pharmaceutical_industry','CSAD'],fontsize=15) 
plt.show()
    


# 2010年10月，医药生物行业龙头股通化东宝大涨，随后行业指数也跟着跟风上涨。
# 
# 上图可以很直观地看出，在10月15日通化东宝大涨后，医药板块指数收益率与上证指数收益率的离散程度明显变大，而在之后医药板块股跟风上涨后，离散程度逐渐减少。
# 

# # 对A股综合日市场回报率指标进行构建

# 本文构建的市场回报率计算公式为：
# 
#     上证A股指数的收益率*上证A股的总流通市值占比
#                      +
#     深证A股指数的收益率*深证A股的总流通市值占比
# 而总流通市值 = 当日指数成份股流通市值的总和

# In[159]:


###计算深证A股每日的流通市值
t0 = time.time()
def get_SZ_CriValue():
    A399107 = get_index_stocks('399107.XSHE',date='2019-09-01')
    A399107_market_value = pd.DataFrame([])
    for i in A399107:
        q = query(valuation.circulating_market_cap).filter(valuation.code==i)
        panel = get_fundamentals_continuously(q, end_date='2019-09-01', count=3100)
        stock_market = panel.minor_xs(i)
    ###删除后缀的原因是本文作者常用数据库时Mongodb，存入数据库的columns不能有点###
        stock = i[0:6]
        A399107_market_value[stock]=stock_market.circulating_market_cap

    ###计算每日的总流通市值###
    A399107_market_value = A399107_market_value.fillna(0)
    A399107_market_value['market_cap']=[0]*len(A399107_market_value)
    for i in range(len(A399107_market_value.index)):
        A399107_market_value['market_cap'][i]=A399107_market_value.iloc[i,:].sum()
        
    return A399107_market_value

A399107 = get_SZ_CriValue()
t1 = time.time()
print('获取数据完成 耗时 %s 秒' %round((t1-t0),3))  
print('')    
print('表4————深证A股07年1月至29年9月01的成份股流通市值每日数据')
print("")
display(A399107.tail())
    


# In[111]:


##计算上证A股每日的流通市值
t0 = time.time()
def get_SH_CriValue():
    A000002 = get_index_stocks('000002.XSHG',date='2019-09-01')
    A000002_market_value = pd.DataFrame([])
    for i in A000002:
        q = query(valuation.circulating_market_cap).filter(valuation.code==i)
        panel = get_fundamentals_continuously(q, end_date='2019-09-01', count=3100)
        stock_market = panel.minor_xs(i)
        stock = i[0:6]
        A000002_market_value[stock]=stock_market.circulating_market_cap

    A000002_market_value = A000002_market_value.fillna(0)
    A000002_market_value['market_cap'] = [0]*len(A000002_market_value)
    for i in range(len(A000002_market_value.index)):
        A000002_market_value['market_cap'][i]=A000002_market_value.iloc[i,:].sum()
        
    return A000002_market_value
        
A000002 = get_SH_CriValue()
t1 = time.time()
print('获取数据完成 耗时 %s 秒' %round((t1-t0),3))  
print('')    
print('表5————上证A股07年1月至29年9月01的成份股流通市值每日数据')
print("")
display(A000002.tail())


# In[109]:


###运用流通市值加权平均法计算Rm###
sz = pd.DataFrame(A399107.market_cap)
sh = pd.DataFrame(A000002.market_cap)
real_market = pd.concat([sz,sh],axis=1)
real_market.columns = ['深证','上证']
real_market.index = A000002.index
real_market['all'] = real_market['深证'] + real_market['上证']


ASH_market_return = get_price('000002.XSHG',start_date='2006-12-06',end_date='2019-09-01',fields='close'
                             ).pct_change(1).fillna(0)
ASZ_market_return = get_price('399107.XSHE',start_date='2006-12-06',end_date='2019-09-01',fields='close'
                             ).pct_change(1).fillna(0)
real_market['R深证'],real_market['R上证'] = ASZ_market_return,ASH_market_return


Rm = (real_market['深证']/real_market['all'] * real_market['R深证']
                               )+(
      real_market['上证']/real_market['all'] * real_market['R上证'])


Rm = Rm[Rm.index>='2007-01-01']


# 下文对本文构建的Rm与数据库的Rm进行方差对比分析

# In[113]:


Rm_sample = Rm[(Rm.index>'2008-12-31')&(Rm.index<='2010-12-31')]

###计算构建的A股综合日市场回报率指标的方差###
Rm_sample_Var = np.var(Rm_sample)
print('本文构建的Rm的方差=%s'%Rm_sample_Var)
print('')

###计算国泰安数据库的A股综合日市场回报率指标的方差###
CSMAR_Var = np.var(CSMAR_market.Market_return_rate)
print('国泰安数据库的Rm的方差=%s'%CSMAR_Var)
print('')

###计算两个指标的相关性###
corr = Rm_sample.corr(CSMAR_market.Market_return_rate)
print('两个指标的相关性=%s'%corr)


#     根据两个指标的方差，本文构建的综合日市场回报率指标比CSMAR的指标更平稳，但是还是存在较大误差，不能很好的复现出CSMAR的指标。
#     
#     
# 于此，本文开始寻找改良本文构建的Rm的方法，寻找改良方法的原因如下：
#     
#         第一：国泰安数据库能免费提供的数据时间过短，只免费提供能到10年12月份，要获取最新的数据成本过大。
#     
#         第二：本文构建的Rm占用内存过多，文件太大，两个dataframe保存到excel的话文件大小是60M。其次基于本文作者穷苦生活现状分析，用积分去升级配置有点得不偿失，积分用的过于浪费，所以寻求更简便的方法计算Rm。
#     
#         第三：信号分布上和原有研报的分布还是存在差异，猜测是构建的指标方差与原文的方差差距过大。而方差过大的原因，本文分析是因为幸存者偏差，因为在获取上证和深证的成份股列表时，是使用2019年9月1日的上深两的成份股信息，由于指数成份股是会定期调整的，所有当前的成份股不代表就是过去成份股。而不获取过去成份股信息的原因是数据量太大，不好去整合分析，所以下文寻求替代方法。
#         
#         第四：目前，本文构建的Rm不仅存在幸存者偏差，而且没有考虑到现金红利再投资的影响，从本文求出的A股两市流通市值和真实的流通市值对比就可以发现差异，所以需要进行改良。
# （替代方法的详情会在下两节在原有研报思路改进上会详细说明。）
# 
# （需要注意的是数据量的问题在聚宽的回测引擎中中是可以解决的。在回测引擎回测时输出当日的成份股，再得到收益率，计算出横截面收益率，这样就不需要保存数据在文件中，方便很多。）

# # 计算替代A股综合日市场回报率

# 计算公式=（上A指数收益率+深A指数收益率）/2
# 
# 意思就是不对流通市值进行加权平均。

# In[6]:


ASH_market_rate_sample = get_price('000002.XSHG',start_date='2008-12-31',end_date='2010-12-31',fields='close'
                                  ).pct_change(1).dropna()
ASZ_market_rate_sample = get_price('399107.XSHE',start_date='2008-12-31',end_date='2010-12-31',fields='close'
                                  ).pct_change(1).dropna()

Sub_Rm_sample = (ASZ_market_rate_sample+ASH_market_rate_sample)/2

###计算构建的A股综合日市场回报率指标的方差###
Sub_Rm_Var = np.var(Sub_Rm_sample.close)
print('本文构建的改良Rm的方差=%s'%Sub_Rm_Var)
print('')

###计算国泰安数据库的A股综合日市场回报率指标的方差###
CSMAR_Var = np.var(CSMAR_market.Market_return_rate)
print('国泰安数据库的Rm的方差=%s'%CSMAR_Var)


# 上文代码可以看出在09年1月1日到10年12月31日期间
# 
#     国泰安数据库的A股综合日市场回报率的方差与构建的改良上证A股和深证A股的平均市场回报率的方差相差不大，比原有构建的指标方差差距更小了，相关性也是几乎等于1。惊喜，可以用更少内存占比得出更接近CSMAR的Rm的指标！！！
# 
#     证明，改良的市场回报率指标能更好复现CSMAR的市场回报率，可以运用上A和深A的平均市场回报率代替数据库的市场回报率的。
# 
#     但是其缺点就是替代市场回报率没有对上A和深A的流通市值进行加权平均去计算市场回报率，所以会与数据库的市场回报率相比会有所偏差。
# 
# （下文的市场回报率都将使用改良的市场回报率—Sub_Rm）

# # 研报中对模型改进章节的复现

# In[209]:


def get_stocks_interval(code,interval_time):
    """简述get_stocks_interval（）函数需要输入的参数和返回值
参数：code = 标的指数的代码（有后缀），不能是单只股票，因为要根据code去获取标的指数的成份股。
     interval_time = 间隔时间。每个标的指数的成份股会定期调整一次，所以需要输入时间列表去获取调整后的成份股。
返回值：stock_timelist = 标的指数code在每个时间段内（interval_time）的成份股表格
    """
    datetime = interval_time
    stock_timelist = []
    if len(code)==11:
        for h in datetime:
            index_list = get_index_stocks(code,date=h)
            stock_timelist.append(index_list)
        stock_timelist = pd.DataFrame(stock_timelist).T
        ###将列名改为每年的1月1日###
        stock_timelist.columns = datetime
        
    elif len(code)==6:
        for h in datetime:
            industry_list = get_industry_stocks(code,date=h)
            stock_timelist.append(industry_list)
        stock_timelist = pd.DataFrame(stock_timelist).T
        ###将列名改为每年的1月1日###
        stock_timelist.columns = datetime
        
        
    return stock_timelist


# In[210]:


def get_stocks_section(stocks,interval_time):
    """简述get_stocks_section()函数需要输入的参数和返回值
参数：stocks = dataframe，内容是get_stocks_interval的返回值，输入每年的成份股数据表格
     interval_time = 同上文的interval_time，回测的时间间隔，每个时间段间相隔一年。         
返回值：stock_CSAD_section = 返回一个dataframe，包含每天所有成份股的横截面绝对离散程度CSAD。
    """
    datetime = interval_time
    start = datetime[0]
    end = datetime[-1]
    stocks_timelist = stocks
    
    """使用改良的A股综合日市场回报率指标"""
    ASH_market_rate = get_price('000002.XSHG',start_date=start,end_date=end,fields='close'
                               ).pct_change(1).fillna(0)
    ASZ_market_rate = get_price('399107.XSHE',start_date=start,end_date=end,fields='close'
                               ).pct_change(1).fillna(0)
    Sub_Rm = (ASZ_market_rate+ASH_market_rate)/2
    

    """获取每个时间段的成份股数据"""
    
    
    """计算标的指数每天的横截面绝对离散程度"""
    stocks_CSAD = pd.DataFrame([],columns=['CSAD'])
    for i in range(len(stocks_timelist.columns)-1):###？？？为何要减1请看下文###
        
        ###用每年标的指数成份股获得当年内的所有成份股的收益率###
        year = stocks_timelist.columns[i+1]
        
        ###过滤掉None，因为在有些标的指数每个时间段内的成份股数量是不同的，返回的len也不同，合在一起时就会出现None###
        current_stock = list(filter(None,stocks_timelist[year]))  
        stocks_rate = pd.DataFrame([],columns=['CSAD'])
        
        
        """第一个for循环是对stocks_rate这个表格的每一列输入每只股票的收盘价再转换成收益率"""
        for k in current_stock:
            if stocks_timelist.columns[i] != end :
                
                ###当stocks_timelist得第i个值等于最后一个column时，第i+1个值是不存在###
                stock_close = get_price(k,start_date = stocks_timelist.columns[i]
                                          ,end_date = stocks_timelist.columns[i+1]
                                          ,fields='close')
                stocks_rate[k] = stock_close.close
            elif stocks_timelist.columns[i] == end:
                pass 
        stocks_rate = stocks_rate.pct_change(1).fillna(0)
        
        
        """第二个for循环是对stocks_rate表格每一行即每天所有成份股的收益率矢量操作减去Sub_Rm
        最后求出当天标的指标的横截面绝对离散程度"""
        for j in range(len(stocks_rate)):
            
            ###iloc返回的是Series，Series的name等于一个交易日，格式是timestamp###
            times = stocks_rate.iloc[j,:].name.strftime('%Y-%m-%d')
            
            ###计算横截面绝对离散程度###
            stocks_section_CASD = abs((stocks_rate.iloc[j,1:]-Sub_Rm.loc[times].values)).sum(
                                        )/len(stocks_rate.iloc[j,1:])

            stocks_rate.CSAD[j] = stocks_section_CASD
            
        CSAD_data = pd.DataFrame(stocks_rate.CSAD)
        stocks_CSAD = stocks_CSAD.append(CSAD_data)
    
    return stocks_CSAD


# In[211]:


###获取07年至19年的标的指数每年的股票列表去计算当年每天得横截面绝对离散程度，避免幸存者偏差###
interval_time = ['2007-01-01','2008-01-01','2009-01-01','2010-01-01'
                ,'2011-01-01','2012-01-01','2013-01-01','2014-01-01'
                ,'2015-01-01','2016-01-01','2017-01-01','2018-01-01'
                ,'2019-01-01','2019-09-01']
t0 = time.time()
###用沪深300进行分析###
stocks = get_stocks_interval('000300.XSHG',interval_time)
HS300_CSAD_section = get_stocks_section(stocks,interval_time)
t1 = time.time()
print('获取数据完成 耗时 %s 秒' %round((t1-t0),3))
print('')    
display(HS300_CSAD_section.tail())


# In[147]:


def get_stocks_belta(target_CSAD,interval_time):  
    """简述get_target_belta()的参数和返回值
参数: target_CSAD = 必须是get_stocks_section的返回值
    interval_time = 最好等于输入进get_target_section的interval_time。不过该函数内只会用到第一个值和最后一个值
                    当作start_time和end_time,所以也可以输入进一个包含两个时间字符串的列表。  
返回值： belta = 用OLS回归后得到的二次项系数，格式是dataframe
    """

    ###额，好吧。改良市场回报率指标我打包放进了get_industry_section函数里面##
    ###为了方便大家调出，这里再打包计算一次###
    datetime = interval_time
    start = datetime[0]
    end = datetime[-1]
    target_CSAD = target_CSAD
    
    """计算改良A股综合日市场回报率指标"""
    ASH_market_rate = get_price('000002.XSHG',start_date=start,end_date=end,fields='close'
                                   ).pct_change(1).fillna(0)
    ASZ_market_rate = get_price('399107.XSHE',start_date=start,end_date=end,fields='close'
                                   ).pct_change(1).fillna(0)
    Sub_Rm = (ASZ_market_rate+ASH_market_rate)/2

    
    """计算二次项系数β2"""
    CSAD = target_CSAD.CSAD
    x1=[]
    x2=[]
    market_data = abs(Sub_Rm)
    for i in range(len(CSAD))[22:]:

        x=market_data.iloc[i-22:i]
        y=CSAD.iloc[i-22:i]
        X = np.column_stack((x**2, x))
        X = sm.add_constant(X)

        model = sm.OLS(y,X)
        results = model.fit()
        x1.append(results.params[1])
        x2.append(results.params[2])
        #print(results.summary())
    
    belta = pd.DataFrame(x1,index = CSAD.index[22:],columns=['x1'])
    return belta


# In[6]:


HS300_belta = get_stocks_belta(HS300_CSAD_section,interval_time)

fig=plt.subplots(figsize=(13,8))
plt.plot(HS300_belta,'b')
plt.grid(True)
plt.title('07年至19年9月β2折线走势图',fontsize=20)
plt.legend(['β2'],fontsize=15)
plt.xlabel('时间',fontsize=15)
plt.ylabel('β2数值',fontsize=15)
plt.show()


# 解读β2显著为负
# 
#     样本不在总体的95%的置信区间内，则判断样本∈拒绝域。而置信区间存在双侧置信区间和单侧置信区间，当用于解读显著为负时，我们只需要考虑单侧的情况，单侧置信区间则为（θ_ ,+∞），则应该运用单侧置信区间的下限——θ_，去判断是否显著为负，若当日的β2∉（θ_ ,+∞），则判断β2显著为负。
#     
#     而计算单侧置信区间的下限，则运用分位数去计算。
#     即总体的2.5%的分位数既是单侧置信区间的下限。
#     当当日β值小于总体（180天的β值）的单侧置信区间下限时则判断该样本显著为负。

# In[148]:


def get_analysis_frame(target_code,belta_frame,interval_time):
    """ 简述get_analysis_frame（）函数的参数和返回值
参数：target_code = 标的指数的代码,格式为字符串
      belta_frame = get_stocks_belta的返回值，      interval_time = 同上文
返回值：belta = 一个dataframe，包含标的指数的belta，MA5,MA10,MA20,滚动22天平均收益率，滚动180天2.5%，5%，10%的分位数。
"""   
    code= target_code
    belta = belta_frame
    belta = pd.DataFrame(belta,columns=[code])
    start = interval_time[0]
    end = interval_time [-1]
    
    
    if len(code)==11:
        target = get_price(code,start_date=start,end_date=end,fields='close')
        R_target = target.pct_change(1).fillna(0)
    elif len(code)==6:
        q = finance.run_query(query(finance.SW1_DAILY_PRICE
                                          ).filter(finance.SW1_DAILY_PRICE.code==code,
                                                   finance.SW1_DAILY_PRICE.date > start,
                                                   finance.SW1_DAILY_PRICE.date < end))
        
        target = pd.DataFrame(list(q.close),index=q.date,columns=['close'])
        R_target = target.pct_change(1).fillna(0)
        
    R_MA22 = R_target.rolling(22,min_periods=22).mean()
    MA30 = target.rolling(30,min_periods=30).mean()
    MA10 = target.rolling(10,min_periods=10).mean()
    MA5 = target.rolling(5,min_periods=5).mean()
    
    belta['MA5'] = MA5
    belta['MA10']= MA10
    belta['MA30']= MA30
    belta['R_MA22'] = R_MA22
    belta['quantile-0.1'] = belta[code].rolling(180,min_periods=180).quantile(0.1)
    belta['quantile-0.05'] = belta[code].rolling(180,min_periods=180).quantile(0.05)
    belta['quantile-0.025'] = belta[code].rolling(180,min_periods=180).quantile(0.025)
    
    """为了避免过拟合的情况，分位数的计算方式为向前滚动计算180天的分位数
    180天即半年，是许多标的指数成份股的调整周期
    """
    
    belta = belta.fillna(method = 'bfill')
    
    return belta


# In[10]:


name = '000300.XSHG'
HS300_belta.columns=[name]
x1_negative = get_analysis_frame(name,HS300_belta[name],interval_time)
R_300 = get_price('000300.XSHG',start_date='2007-01-01',end_date='2019-09-01',fields=['low','close'])


# In[11]:


###单侧置信区间下限为百分之2.5时belta信号的分布状况
signal_01 = pd.DataFrame([])

for i in range(len(x1_negative)):
    if x1_negative[name][i]<x1_negative['quantile-0.025'][i    ###筛选出小于单侧置信区间2.5%下限的值###
        ] and x1_negative.R_MA22[i]>0:                         ###筛选出22天平均收益率大于0的值###
        
        signl = pd.DataFrame(x1_negative.iloc[i,:]).T
        signal_01 = signal_01.append(signl)

display(signal_01.tail())
print('')
print('单侧置信区间下限为百分之2.5的belta信号总共出现了%s次'%len(signal_01))
print('')

###信号坐标##
Y = R_300.loc[signal_01.index].low
X = signal_01.index

fig = plt.subplots(figsize=(13,8))
plt.plot(R_300.close,'b')
plt.grid(True)
plt.title('图四单侧置信区间下限2.5%时的β信号分布图',fontsize=20)


"""本来是完整的复现研报那样画出长竖直线的，但是发现最细的线条也是挺宽的"""
"""再加上信号点有点多,全部放在图片上的话有点小丑，所以就缩短直线的长度，尽可能美观点"""
for x in X:
    loc = int(Y.loc[x])
    c = range(loc-400,loc+400)
    plt.scatter([x]*len(c),c,color='r',marker='.',s=1)
plt.legend(['000300.XSHG'],fontsize=20)
plt.xlabel('时间',fontsize=15)
plt.ylabel('SZ50指数收盘价',fontsize=15)
plt.show()


# # 在原有研报的思路上进行尝试和改进

# 第一小节:关于单侧置信区间下限的改进
#     
#     对于原始的置信区间，2.5%的单侧置信区间下限去计算时，会发现在07年至19年间信号量少得只有85次，策略总体的频率较低，有一大部分时间策略是处于空仓状态。
#     因此，本小节以增加信号量，提高策略开仓频率为目的进行改进。
#     改进的方法为提高优化单侧置信区间下限，得到更多的β2显著为负的信号。
#     **优化后的单侧置信区间下限为10%，单侧置信区间为90%。

# In[12]:


###单侧置信区间下限为百分之10时belta信号的分布状况

signal_02 = pd.DataFrame([])
for i in range(len(x1_negative)):
    if x1_negative[name][i]<x1_negative['quantile-0.1'][i   ###筛选出小于单侧置信区间10%下限的值###
        ] and x1_negative.R_MA22[i]>0:                      ###筛选出22天平均收益率大于0的值###
        
        signl = pd.DataFrame(x1_negative.iloc[i,:]).T
        signal_02 = signal_02.append(signl)
print('单侧置信区间下限为百分之10的belta信号总共出现了%s次'%len(signal_02))
print('')


###增加过滤指标后的信号坐标###
Y = R_300.loc[signal_02.index].low
X = signal_02.index


fig = plt.subplots(figsize=(13,8))
plt.plot(R_300.close,'b')
plt.grid(True)
plt.title('图五单侧置信区间下限为5%时的β信号分布图',fontsize=20)

for x in X:
    loc = int(Y.loc[x])
    c = range(loc-400,loc+400)
    plt.scatter([x]*len(c),c,color='r',marker='.',s=1)
plt.legend(['000300.XSHG'],fontsize=20)
plt.xlabel('时间',fontsize=15)
plt.ylabel('HS300指数收盘价',fontsize=15)
plt.show()


#         可以看出在提高单侧置信区间下限后，我们发现得出的信号更多了，期间总共出现了257次显著性信号。
#         事实上，策略的频率提高了，显著性信号增加了，但出现的问题是信号更加混杂了。
#         在08年熊市市场大幅度，出现了下跌羊群效应，但是判断市场趋势的指标却判断为上涨市场，这样会使得策略的回撤大大增加，收益率减少。
#         为此，下一小节会着重于对市场趋势指标的改进。

# 第三小节：关于判断市场趋势的过滤指标的改进
# 
#         在基于上一小节改进的基础上，本小节对市场趋势的过滤指标进行改进。原有研报是使用22日内指数平均收益率的正负区间去判断市场趋势是涨还是跌。但是这就会出现一个问题，在上一小节改进置信区间后，在08年7月份出现1次，8月份出现3次异常信号，这时候市场是处于熊市下跌的趋势的，但是MA22平均收益率却是显示为正。像这种情况在提高单侧置信区间下限后就会出现，原因是信号更多更驳杂了，原有的判断市场趋势的指标不能很好的判断趋势，所以该趋势判断指标并不是很合理。于是基于市场趋势的判断指标，而且不改变原来研报的核心思路下,我采用最原始的均线去判断市场状态，过滤掉异常情况。
#     
#         即指数当日MA30大于30日前的MA30,则判断为上涨市场，反之则为下跌市场。
#     

# In[13]:


###单侧置信区间下限为百分之10，并增加过滤指标时belta信号的分布状况
signal_03 = pd.DataFrame([])
for i in range(len(x1_negative)):
    if x1_negative[name][i]<x1_negative['quantile-0.1'][i   ###筛选出小于单侧置信区间10%下限的值###
    ] and x1_negative.MA30[i]>x1_negative.MA30[i-30         ###筛选出大于前30天MA30的值###
    ] and x1_negative.R_MA22[i]>0:                          ###筛选出22天平均收益率大于0的值###
        eden =  pd.DataFrame(x1_negative.iloc[i,:]).T                                                                                                
        signal_03 = signal_03.append(eden)
        
        
###改良后的信号坐标###
Y = R_300.loc[signal_03.index].low
X = signal_03.index

print('单侧置信区间下限为百分之10并改进后的belta信号总共出现了%s次'%len(signal_03))
print('')


fig = plt.subplots(figsize=(13,8))
plt.plot(R_300.close,'b',label='A000300.XSHG')   
plt.grid(True)
plt.title('图六改进的β信号分布图',fontsize=20)
plt.legend(['000300.XSHG'],fontsize=20)


for x in X:
    loc = int(Y.loc[x])
    c = range(loc-400,loc+400)
    plt.scatter([x]*len(c),c,color='r',marker='.',s=1)
    
plt.xlabel('时间',fontsize=15)
plt.ylabel('SZ50指数收盘价',fontsize=15)
plt.show()


#     到这里本文发现在运用均线过滤异常情况后，可以直观地看出已经过滤掉了大部分下跌市场时的信号，提高了策略的成功率和收益率。信号量缩小到195次，且信号的纯度更高。
#     

# ### 展示多空的因子信号分布状况

# 本文基于原有研报的基础上进行改良。研报显示市场存在下跌趋势时，羊群效应策略效果相对不显著的原因可能是卖空限制导致投资者可抛售的股票数量有限，因此下跌时羊群效应发生后下跌趋势持续时间较短。于此，本文对做空信号的因子过滤的均线选择为短期均线MA5.

# In[14]:


###基于上文改良，belta信号的多空分布状况
long_signal = pd.DataFrame([])
for i in range(len(x1_negative)):
    if x1_negative[name][i]<x1_negative['quantile-0.1'][i ###筛选出小于单侧置信区间10%下限所在值###                
        ] and x1_negative.MA30[i]>x1_negative.MA30[i-30   ###筛选出大于前30天MA30的值###
        ] and x1_negative.R_MA22[i]>0:                    ###筛选出22天平均收益率大于0的值###
        
        long =  pd.DataFrame(x1_negative.iloc[i,:]).T                                                                                                
        long_signal = long_signal.append(long)
        
        
short_signal = pd.DataFrame([])
for i in range(len(x1_negative)):
    
    if x1_negative[name][i]<x1_negative['quantile-0.1'][i ###筛选出小于单侧置信区间10%下限所在值###             
        ] and x1_negative.MA30[i]<x1_negative.MA30[i-5    ###筛选出大于前 5天MA30的值###
        ] and x1_negative.R_MA22[i]<0:                    ###筛选出22天平均收益率小于0的值###
        
        short =  pd.DataFrame(x1_negative.iloc[i,:]).T                                                                                                
        short_signal = short_signal.append(short)
      
    
###做多的信号坐标###
Y_long = R_300.loc[long_signal.index].low
X_long = long_signal.index
###做空的信号坐标###
Y_short = R_300.loc[short_signal.index].low
X_short = short_signal.index


fig = plt.subplots(figsize=(13,8))
plt.plot(R_300.close,'b',label='000300.XSHG')
plt.title('图七  改进后多空的β信号分布图',fontsize=20)
plt.grid(True)


for long in X_long:
    loc = int(Y_long.loc[long])
    c = range(loc-400,loc+400)
    plt.scatter([long]*len(c),c,color='r',marker='.',s=1)
    
for short in X_short:
    loc = int(Y_short.loc[short])
    c = range(loc-600,loc+600)
    plt.scatter([short]*len(c),c,color='green',marker='.',s=1)
    
plt.legend(['000300.XSHG'],fontsize=20)
plt.xlabel('时间',fontsize=15)
plt.ylabel('SZ50指数收盘价',fontsize=15)
plt.show()


# 从上图中可以很明显的看出，做空信号的发生除了在08年崩盘的之外，其他时候基本都发生在下跌时某个阶段的末段，这与研报的分析的相同，在下跌时羊群效应发生后下跌趋势持续时间较短。

# (代码的打包操作放于回测部分)

# # 策略思路以及策略回测

#         选择投资者较为熟悉的六种宽基指数，上证综指、上证50、沪深300、中证500、中小板综合指数、创业板指数
#         策略标的：以上六种宽基指数
#         股票组合：当日标的指数的成分股，剔除ST股等非正常交易状态股票        
#         回测时间：2017年1月1日至2019年9月1日        
#         手续费用：在此回测暂时忽略，但在回测引擎上设置为双边千三 
#         
#         策略步骤：（做多）
#                      向前22日计算每天（包括当天）的成份股组合截面绝对离散度CSAD，OLS估计CCK模型中Rm平方的系数β2。
#                      向前计算180天内的样本单侧置信区间10%下限。
#                      向前计算22日内标的指数平均收益率的正负区间，运用正负判断市场趋势
#                      向前计算MA30
#                  开仓：当β2小于单侧置信区间下限，标的指数平均收益率为正，当日MA30大于30天前的MA30时，买入标的指数。
#                  清仓：持有22天后卖出。（为了接近研报，暂时不加入止损止盈，但是之后会在评论区发布）
#                  -----------------------------------------------------------------------------------------------------
#                   （做空）
#                      所需计算的指标同上文做多策略。
#                  开仓：当β2小于单侧置信区间下限，标的指数平均收益率为正，当日MA30小于5天前的MA30时，买入标的指数。
#                  清仓：持有7天后卖出。（为了接近研报，暂时不加入止损止盈，但是之后会在评论区发布）

# ## 羊群效应策略在宽基指数上的应用

# In[149]:


def get_signal(industry_code,direction,interval_time):
    """简述get_signal的参数和返回值：
参数： industry = 标的指数的列表，列表可以包含多个标的或单个标的指数，但是industry的格式必须是列表。
      direction = 仅能输入‘buy’和 ‘sell’    interval_time = 同上文的一样，是个时间间隔的列表。
返回值：signal_group = 返回一个dict，里面包含每个标的指数在interval_time的时间内出现的所有信号。
                        返回的dict的一级字典键为标的指数的代码，每个键对应值的格式为dataframe。
    """
    industry = industry_code
    time = interval_time
    
    all_belta = pd.DataFrame([])
    for i in industry:
        blanket_stocks = get_stocks_interval(i,time)
        stocks_section = get_stocks_section(blanket_stocks,time)
        belta  = get_stocks_belta(stocks_section,time)    ###获取每个指数的每天的belta信号###
        all_belta[i] = belta.x1                           ###存入all_belta这dataframe里面###
        
        
    signal_group={} 
    if direction == 'buy':
        for k in industry:
            signal = pd.DataFrame([])
            frame = get_analysis_frame(k,all_belta[k],time)    
            for i in range(len(frame)):
                if frame[k][i]<frame['quantile-0.1'][i    ###筛选出小于单侧置信区间10%下限所在值###
                    ] and frame.MA30[i]>frame.MA30[i-30   ###筛选出大于前30天MA30的值###
                    ] and frame.R_MA22[i]>0:              ###筛选出22天平均收益率大于0的值###

                    long =  pd.DataFrame(frame.iloc[i,:]).T                                                                        
                    signal = signal.append(long)
            signal_group[k] = signal   ###将每个指数出现的做多信号以dict形式存储起来###
            
    elif direction =='sell':
        for k in industry:
            signal = pd.DataFrame([])
            frame = get_analysis_frame(k,all_belta[k],time)
            for i in range(len(frame)):
                if frame[k][i]<frame['quantile-0.1'][i    ###筛选出小于单侧置信区间10%下限所在值###
                    ] and frame.MA30[i]<frame.MA30[i-5    ###筛选出大于前5天MA30的值###
                    ] and frame.R_MA22[i]<0:              ###筛选出22天平均收益率大于0的值###

                    short =  pd.DataFrame(frame.iloc[i,:]).T                                                                                                
                    signal = signal.append(short)
            signal_group[k] = signal   ###将每个指数出现的做空信号以dict形式存储起来###
            
    return signal_group


# In[150]:


"""简述get_back_analysis_data（）函数的参数和返回值

参数：daliy_rate = 每日的收益率，格式必须是一列dataframe     code = 回测的标的的代码，最好只是一个字符串。
      cum_rate = 每日收益率的累加，格式也是必须一列dataframe
      days = 回测的天数（日历日，非工作日。因为计算年化要用日历日）
返回值： 一个1×1的dataframe，包含标的的最终收益率，年化收益率，最大回撤，夏普比率
"""

def get_back_analysis_data(daliy_rate,cum_rate,code,days):
    rate = cum_rate
    Daliy_rate = np.array(daliy_rate)
    
    ###计算最大回撤###
    re = []
    for k in range(len(rate)):
        retreat = max(rate.iloc[k,0]-rate.iloc[k:,0])
        re.append(retreat)
    max_retreat = max(re)  
    
    ###计算收益率###
    earn = rate.iloc[-1,-1]-1 
    
    ###计算年化收益率###
    annual_return = (earn)*365/days      
    
    ###计算夏普比率###
    ex_pct_close = Daliy_rate - 0.04/252
    sharpe = (ex_pct_close.mean() * math.sqrt(252)
                     )/ex_pct_close.std()
    
    analysis_datas = [earn,annual_return,max_retreat,sharpe]
    back_analysis_data = pd.DataFrame(analysis_datas,
                                      index =['收益率','年化收益率','最大回撤','夏普比率'
                                      ],columns = [code]).T
    
    return  back_analysis_data


# In[151]:


def get_backtest_gain(code_list,direction,interval_time):   
    """参数：code_list = 同get_analysis_frame函数一样，必须是个包含标的指数字符串代码的列表，数量不限。
          direction = 同get_signal函数一样，必须是‘buy’或者‘sell’        
          interval_time = 同上文
       返回值：gain_dict = 字典，包含code_list里面每个标的指数在interval_time内每天的收益率
           back_revenue = 一个dataframe，包含每个code_list回测的数据分析，收益、年化、回撤、夏普
    """

    to = time.time()
    signal_group = get_signal(code_list,direction,interval_time)
    color = ['red','orange','yellow','green','blue','purple']
    back_revenue = pd.DataFrame()       ###用于存放每个标的的回测收益数据###
    start = interval_time[0]
    end = interval_time[-1]


    gain_dict={}
    keys = list(signal_group.keys())
    for i in range(len(keys)):
        stock = keys[i]
        
        ###判断是行业指数还是宽基指数###
        if len(stock)==11:
            index = get_price(stock,start_date=start,end_date=end,fields='close')
            back_data = index.pct_change(1)

        elif len(stock)==6:
            q = finance.run_query(query(finance.SW1_DAILY_PRICE
                                          ).filter(finance.SW1_DAILY_PRICE.code==stock,
                                                   finance.SW1_DAILY_PRICE.date > start,
                                                   finance.SW1_DAILY_PRICE.date < end))
            
            change_index_format = get_price('000001.XSHG',start_date=start,
                                                          end_date=end,fields='close')
            
            industry = pd.DataFrame(list(q.close),
                                    index=change_index_format.index,
                                    columns=['close'])
            back_data = industry.pct_change(1)
  

        ###取出信号出现的日期索引###
        signal_time = signal_group[stock].index
        ###创建一个NaN列表，在对应行存入对应的收益率，再向前填充NaN##
        backtest = [NaN]*len(back_data.index)     
        

        
        for j in signal_time:
            
            ###信号所在行数###
            row = int(np.where(back_data.index == j)[0])
            ###避免未来函数，信号出现的第二天才开始计算收益率###
            if direction=='buy':
                backtest[row+1:row+23] = back_data.iloc[row+1:row+23].close
            elif direction=='sell':
                backtest[row+1:row+8] = back_data.iloc[row+1:row+8].close*-1
                
                
        backtest = pd.DataFrame(backtest,index=back_data.index,columns=['b_s'])
        backtest = backtest.fillna(0)
        R_backtest=backtest
        gain_dict[keys[i]] = R_backtest

            
        """生成回测数据"""
        ###回测数据预处理###
        R_backtest.iloc[0,0]=1    ###累积收益计算完成后第一个值再变回零###
        R_cum = R_backtest.cumsum()
        R_backtest.iloc[0,0]=0    ###第一个值再变回零是为了谨慎计算sharpe
                                  ###不然计算时会把第一天的收益当成100%去计算###
        
        days = len(pd.date_range(start,end))   ###计算interval_time列表中首末日期相隔的日历日###
        revenue = get_back_analysis_data(R_backtest,R_cum,keys[i],days)
        back_revenue = back_revenue.append(revenue)    ###把标的的回测收益数据存入back_revenue###
    
    
    return gain_dict,back_revenue


# In[126]:


index_code = ['000001.XSHG','000016.XSHG','000300.XSHG'
              ,'000905.XSHG','399101.XSHE','399006.XSHE']
interval_time = ['2007-01-01','2008-01-01','2009-01-01','2010-01-01'
                ,'2011-01-01','2012-01-01','2013-01-01','2014-01-01'
                ,'2015-01-01','2016-01-01','2017-01-01','2018-01-01'
                ,'2019-01-01','2019-09-01']


# In[25]:


###宽基指数多空回测###
t0 = time.time()

long_back_rate,long_back_revenue = get_backtest_gain(index_code,'buy',interval_time)
short_back_rate,short_back_revenue = get_backtest_gain(index_code,'sell',interval_time)

back_rate = [long_back_rate,short_back_rate]
back_revenue = [long_back_revenue,short_back_revenue]

index_colors = ['red','orange','yellow','green','blue','purple']

for k in range(0,2):
    
    fig=plt.figure(figsize=(13,8))
    ax1=fig.add_subplot(1,1,1)

    keys = list(back_rate[k].keys())
    for i in range(len(keys)):
        target = keys[i]
        target_back_rate = back_rate[k][target]
        target_back_rate.iloc[0,0]=1
        target_back_rate_cum = target_back_rate.cumsum()
        
        ax1.plot(target_back_rate_cum,index_colors[i])

    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('value ratio') 
    #ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
    ax1.axhline(1.0, linestyle='-', color='black', lw=1)
    plt.legend(index_code,loc=2,fontsize=10) 

    if k == 0:
        print('宽基指数做多净值组合回测数据表格')
        plt.title('宽基指数做多净值组合走势图',fontsize=20) 
        
    elif k ==1 :
        print('宽基指数做空净值组合回测数据表格')
        plt.title('宽基指数做空净值组合走势图',fontsize=20) 
    display(back_revenue[k])
    print('')
    plt.show()
    
    
t1 = time.time()   
print('回测完成 总耗时 %s 秒' %round((t1-t0),3))  
print('')    


#     做多策略时，策略在宽基指数上的平均表现为上图，可以看出在每轮牛市时，羊群效应策略效果相对较为显著，在沪深300和上证50等高市值指数回测结果优于低市值指数。标的为上证50时，收益率达到73.3%，年化5.7%，最大回撤略大43.2%，夏普0.14。
#     总的来说，上证50，沪深300的表现均优于中证500，中小板综，创业板指，上证综指。
#     
#     做空策略时，策略在宽基指数上的表现一般，表现最好最明显的是沪深300，收益率达到40.5%。但是回测结果最差的却是创业板指数，亏损40%。
#     在所有策略中，尽管加入过过滤指标改良，但是还是不可避免的出现羊群效应发生时错误的市场判断。

# In[53]:


###上涨时羊群效应策略分日累计收益率
fig=plt.figure(figsize=(13,5))
ax2=fig.add_subplot(1,1,1)
long_keys = list(long_back_rate.keys())

for i in range(len(long_keys)):
    sample = long_back_rate[keys[i]].loc['2009-06-30':'2009-07-30']
    sample = sample.reset_index(drop=True)
    cum_sample = sample.cumsum()
    ax2.plot(cum_sample,index_colors[i])
    
plt.title('上涨时羊群效应策略分日累计收益率',fontsize=20)
plt.grid(True)
plt.xlabel('time',fontsize=15)
plt.legend(long_keys,fontsize=12)
plt.show()


# In[54]:


###下跌时羊群效应策略分日累计收益率
fig=plt.figure(figsize=(13,5))
ax2=fig.add_subplot(1,1,1) 
short_keys = list(short_back_rate.keys())

for i in range(len(short_keys)):
    sample = short_back_rate[keys[i]].loc['2008-09-30':'2008-10-29']
    sample = sample.reset_index(drop=True)
    cum_sample = sample.cumsum()
    ax2.plot(cum_sample,index_colors[i])
    
plt.title('下跌时羊群效应策略分日累计收益率',fontsize=20)
plt.grid(True)
plt.xlabel('time',fontsize=15)
plt.legend(short_keys,fontsize=12)
plt.show()


# ## 羊群效应策略在行业指数上的应用

#         申万一级行业指数，并分成4大类，金融型，成长型，消费型，周期型
#         策略标的：以上4大类行业
#         股票组合：当日标的指数的成分股，剔除ST股等非正常交易状态股票        
#         回测时间：2014年3月1日至2019年9月1日        
#         手续费用：在此回测暂时忽略，但在回测引擎上设置为双边千三 
#         
#         策略步骤：（做多）
#                      向前22日计算每天（包括当天）的成份股组合截面绝对离散度CSAD，OLS估计CCK模型中Rm平方的系数β2。
#                      向前计算180天内的样本单侧置信区间10%下限。
#                      向前计算22日内标的指数平均收益率的正负区间，运用正负判断市场趋势
#                      向前计算MA30
#                  开仓：当β2小于单侧置信区间下限，标的指数平均收益率为正，当日MA30大于30天前的MA30时，做多标的指数。
#                  清仓：持有22天后卖出。（为了接近研报，暂时不加入止损止盈，但是之后会在评论区发布）
#                  -----------------------------------------------------------------------------------------------------
#                   （做空）
#                      所需计算的指标同上文做多策略。
#                  开仓：当β2小于单侧置信区间下限，标的指数平均收益率为负，当日MA30小于5天前的MA30时，做空标的指数。
#                  清仓：持有7天后卖出。（为了接近研报，暂时不加入止损止盈，但是之后会在评论区发布）

# In[76]:


Cycle = ['801020','801030','801040','801050','801710',
         '801720','801730','801890','801170','801180','801160']

Consumption = ['801180','801110','801130','801200',
               '801120','801150','801210','801140','801010']

Growth = ['801080','801770','801760','801750','801740']

Finance = ['801780','801790']

times  = ['2014-03-01','2015-01-01','2016-01-01','2017-01-01'
          ,'2018-01-01','2019-01-01','2019-09-01']


# 在进行股票获取的时候发现，聚宽申万一级提供的行业成份股中，金融型【‘801780’，‘801790’】,还有电子，传媒等行业在14年2月10日之前是没有成份股的，
# 
# 所以在行业指数上的回测时间从14年3月1日至19年9月1日

# In[77]:


###获取四大类内每个行业回测的没日收益率和回测分析表格

t0=time.time()
Cycle_long_back_rate,Cycle_long_back_revenue = get_backtest_gain(Cycle,'buy',times)
Cycle_short_back_rate,Cycle_short_back_revenue = get_backtest_gain(Cycle,'sell',times)

Growth_long_back_rate,Growth_long_back_revenue = get_backtest_gain(Growth,'buy',times)
Growth_short_back_rate,Growth_short_back_revenue = get_backtest_gain(Growth,'sell',times)


Consumption_long_back_rate,Consumption_long_back_revenue = get_backtest_gain(Consumption,'buy',times)
Consumption_short_back_rate,Consumption_short_back_revenue = get_backtest_gain(Consumption,'sell',times)


Finance_long_back_rate,Finance_long_back_revenue = get_backtest_gain(Finance,'buy',times)
Finance_short_back_rate,Finance_short_back_revenue = get_backtest_gain(Finance,'sell',times)

t1=time.time()
print('获取回测数据完毕 总耗时 %s 秒' %round((t1-t0),3))  
print('')   
print('buy:周期型各行业回测数据表格')
display(Cycle_long_back_revenue)
print('')   
print('buy:消费型各行业回测数据表格')
display(Growth_long_back_revenue)
print('')   
print('buy:成长型各行业回测数据表格')
display(Consumption_long_back_revenue)
print('')   
print('buy:金融型各行业回测数据表格')
display(Finance_long_back_revenue)


#     由于数据过多，不方便一起展示，所以上面四个表格只显示了四大类行业每个行业做多的收益分析表格，没有显示做空的。
#     在回测中，周期型行业的801720-建筑装饰行业和801040-钢铁行业的做多盈利达到62%和47%，高于周期型的其他行业。
#              消费型行业的801740-通信行业的做多盈利达到43%，在消费型行业中较为突出，是其中唯一一个收益率为正的行业。
#              成长型行业的801140-轻工制造行业的做多盈利达到24.3%。
#              金融型行业的801790-非银行金融行业的做多盈利达到49.5%。
#              在所有行业中表现最差的是消费型行业中801760-传媒行业和801080-电子行业，做多亏损率分别为-44.7%和-43%。
#              

# In[78]:


###通过每个行业的收益率计算四大类的平均收益率

industry_long_list = [Cycle_long_back_rate,Growth_long_back_rate
                      ,Consumption_long_back_rate,Finance_long_back_rate]

industry_short_list = [Cycle_short_back_rate,Growth_short_back_rate
                      ,Consumption_short_back_rate,Finance_short_back_rate]

for industry in industry_long_list:
    keys=list(industry.keys())
    industry['avg_rate'] = industry[keys[0]]
    for i in range(len(keys)-1):
        industry['avg_rate'] += industry[keys[i+1]] ###累加计算出每个大类做多的总的收益率
                                                    ###并在下文代码除以每个大类的行业数求得平均收益率
        
for industry in industry_short_list:
    keys=list(industry.keys())
    industry['avg_rate'] = industry[keys[0]]
    for i in range(len(keys)-1):
        industry['avg_rate'] += industry[keys[i+1]] ###累加计算出每个大类做多的总的收益率
                                                    ###并在下文代码除以每个大类的行业数求得平均收益率
            

Cycle_avg_long_rate = Cycle_long_back_rate['avg_rate']/11
Consumption_avg_long_rate = Consumption_long_back_rate['avg_rate']/5
Growth_avg_long_rate = Growth_long_back_rate['avg_rate']/9
Finance_avg_long_rate = Finance_long_back_rate['avg_rate']/2


# In[80]:


###四大类行业做多回测累计收益率###
industry_colors = ['magenta','lime','darkorchid','darkorange']

industry_avg_long_list = [Cycle_avg_long_rate,Consumption_avg_long_rate
                          ,Growth_avg_long_rate,Finance_avg_long_rate]

classfiy_name = ['Cycle','Consumption','Growth','Finance']
industry_long_data = pd.DataFrame()

for avg in range(0,4):
    analysis = get_back_analysis_data(industry_avg_long_list[avg]
                                      ,industry_avg_long_list[avg].cumsum()+1
                                      ,classfiy_name[avg],2011)
                                        ###行业指数回测总日历天数为2011天
    industry_long_data = industry_long_data.append(analysis)
    
print('四大类行业指数回测数据分析表格')
display(industry_long_data)


fig=plt.figure(figsize=(13,8))
ax3=fig.add_subplot(1,1,1)   
for avg in range(0,4):
    ax3.plot(industry_avg_long_list[avg].cumsum()+1,industry_colors[avg])
plt.title('四大类行业指数做多净值组合走势图',fontsize=20)
plt.grid(True)
ax3.axhline(1.0, linestyle='-', color='black', lw=1)
plt.xlabel('time',fontsize=15)
plt.legend(classfiy_name,fontsize=12)
plt.show()


#     在做多策略中，羊群效应策略效果不能说太好，但是有一个缺点就所有行业指数的收益大部分来自大牛市之中
#     在大牛市过后的市场震荡阶段策略表现明显欠佳。
#     表现最好的是市值较大的金融型行业，在大牛市中行业指数涨幅较大，收益较高，但是在之后的震荡市之中错误分辨信号导致亏损。

# In[81]:


###行业指数上涨时羊群效应策略分日累计收益率
fig=plt.figure(figsize=(13,6))
ax4=fig.add_subplot(1,1,1)  

for i in range(len(industry_avg_long_list)):
    sample = industry_avg_long_list[i].loc['2014-11-20':'2014-12-21']
    sample = sample.reset_index(drop=True)
    cum_sample = sample.cumsum()
    ax4.plot(cum_sample,industry_colors[i])
plt.title('上涨时羊群效应策略分日累计收益率',fontsize=20)
plt.grid(True)
plt.xlabel('time',fontsize=15)
plt.legend(classfiy_name,fontsize=12)
plt.show()


#     下跌时羊群效应的分日累计本文选取2014年11月20到12月21日，期间市场是处于大牛市阶段，市场出现较大的涨幅。
#     该阶段表现最好的是金融型行业，最差的是成长型行业。

# In[82]:


###四大类行业做空回测累计收益率###
Cycle_avg_short_rate = Cycle_short_back_rate['avg_rate']/11
Consumption_avg_short_rate = Consumption_short_back_rate['avg_rate']/5
Growth_avg_short_rate = Growth_short_back_rate['avg_rate']/9
Finance_avg_short_rate = Finance_short_back_rate['avg_rate']/2

industry_avg_short_list = [Cycle_avg_short_rate,Consumption_avg_short_rate
                          ,Growth_avg_short_rate,Finance_avg_short_rate]

industry_short_data = pd.DataFrame()

for avg in range(0,4):
    analysis = get_back_analysis_data(industry_avg_short_list[avg]
                                      ,industry_avg_short_list[avg].cumsum()+1
                                      ,classfiy_name[avg],2011)
    industry_short_data = industry_short_data.append(analysis)
    
print('四大类行业指数回测数据分析表格')
display(industry_short_data)


fig=plt.figure(figsize=(13,8))
ax3=fig.add_subplot(1,1,1)   
for avg in range(0,4):
    ax3.plot(industry_avg_short_list[avg].cumsum()+1,industry_colors[avg])
plt.title('四大类行业指数做空净值组合走势图',fontsize=20)
plt.grid(True)
ax3.axhline(1.0, linestyle='-', color='black', lw=1)
plt.xlabel('time',fontsize=15)
plt.legend(classfiy_name,fontsize=12)
plt.show()


#     在做空策略中，即使把策略清仓时间缩短到7天，羊群效应策略在行业指数上的效果也是相对不显著。
#     总收益率方面，所以行业均为负收益。
#     大部分亏损发生于15年下半年后，当时市场处于牛市变熊市阶段且触摸到阶段性底部。
#     由于此时波动较大，发出了错误方向信号，导致亏损。

# In[83]:


###行业指数下跌时羊群效应策略分日累计收益率
fig=plt.figure(figsize=(13,6))
ax5=fig.add_subplot(1,1,1)  

for i in range(len(industry_avg_short_list)):
    sample = industry_avg_short_list[i].loc['2017-03-25':'2017-04-29']
    sample = sample.reset_index(drop=True)
    cum_sample = sample.cumsum()
    ax5.plot(cum_sample,industry_colors[i])
plt.title('下跌时羊群效应策略分日累计收益率',fontsize=20)
plt.grid(True)
plt.xlabel('time',fontsize=15)
plt.legend(classfiy_name,fontsize=12)
ax5.axhline(0, linestyle='-', color='black', lw=1)
plt.show()


#      下跌时羊群效应的分日累计本文选取2014年3月25到4月29日，期间市场是处于慢牛回调阶段，市场出现较大的跌幅。
#      该阶段表现最好的消费性行业，最差的是金融型行业。

# # 结论
# ## 1.标的指数的市值越高，策略效果越明显
# ## 2.通过研报和本文分析，羊群效应确实存在，但是重要的是市场方向的区分，而本文虽有在这方面进行改进，但还是会出现错误的判断。所以羊群效应要想有好的表现就必须要正确区分信号所代表的市场方向。
# ## 3.单从本文的回测结果看，该策略明显不足，收益率低，回撤大，且大部分策略收益都在大牛市，在震荡期间策略表现不明显，所以策略还不成熟，有待改进
