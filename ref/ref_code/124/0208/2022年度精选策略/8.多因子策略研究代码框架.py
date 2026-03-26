#!/usr/bin/env python
# coding: utf-8

# # 第一模块 数据准备

# In[ ]:


import pandas as pd
import numpy as np
import time
import datetime
import statsmodels.api as sm
import pickle
import warnings
from jqdata import *
warnings.filterwarnings('ignore')


# In[2]:


#此代码在自己机器上运行时使用，在聚宽平台上跑数据不需要
#from jqdatasdk import *
#auth('用户名','密码')


# In[3]:



start_date = '2013-01-01'
end_date = '2014-01-01'


all_trade_days = (get_trade_days(start_date=start_date,end_date=end_date)).tolist() #所有交易日
trade_days = all_trade_days[::20] #每隔20天取一次数据，基本面数据更新频率较慢，数据获取频率尽量与之对应

securities = get_all_securities()
start_data_dt = datetime.datetime.strptime(start_date,'%Y-%m-%d').date()
securities_after_start_date = securities[(securities['start_date']<start_data_dt)] #选择起始时间之前上市的股票
all_stocks = list(securities_after_start_date.index)

INDUSTRY_NAME = 'sw_l1'

ttm_factors = []


# # 基本面数据及缺失值填充函数

# In[4]:


'''
基本面因子映射
'''
fac_dict = {
    'MC':valuation.market_cap, # 总市值
    'GP':indicator.gross_profit_margin * income.operating_revenue, # 毛利润
    'OP':income.operating_profit,
    'OR':income.operating_revenue, # 营业收入
    'NP':income.net_profit, # 净利润
    'EV':valuation.market_cap + balance.shortterm_loan+balance.non_current_liability_in_one_year+balance.longterm_loan+balance.bonds_payable+balance.longterm_account_payable - cash_flow.cash_and_equivalents_at_end,
    
    'TOE':balance.total_owner_equities, # 股东权益合计(元)
    'TOR':income.total_operating_revenue, # 营业总收入
    'EBIT':income.net_profit+income.financial_expense+income.income_tax_expense,
    
    'TOC':income.total_operating_cost,#营业总成本
    'NOCF/MC':cash_flow.net_operate_cash_flow / valuation.market_cap, #经营活动产生的现金流量净额/总市值
    'OTR':indicator.ocf_to_revenue, #经营活动产生的现金流量净额/营业收入(%) 
    
    
    'GPOA':indicator.gross_profit_margin * income.operating_revenue / balance.total_assets,  #毛利润 / 总资产 = 毛利率*营业收入 / 总资产
    'GPM':indicator.gross_profit_margin, # 毛利率
    'OPM':income.operating_profit / income.operating_revenue, #营业利润率
    'NPM':indicator.net_profit_margin, # 净利率
    'ROA':indicator.roa, # ROA
    'ROE':indicator.roe, # ROE
    'INC':indicator.inc_return, # 净资产收益率(扣除非经常损益)(%)
    'EPS':indicator.eps, # 净资产收益率(扣除非经常损益)(%)
    'AP':indicator.adjusted_profit, # 扣除非经常损益后的净利润(元)
    'OP':indicator.operating_profit, # 经营活动净收益(元)
    'VCP':indicator.value_change_profit, # 价值变动净收益(元) = 公允价值变动净收益+投资净收益+汇兑净收益
    
    'ETTR':indicator.expense_to_total_revenue, # 营业总成本/营业总收入(%)
    'OPTTR':indicator.operation_profit_to_total_revenue, # 营业利润/营业总收入(%)
    'NPTTR':indicator.net_profit_to_total_revenue, # 净利润/营业总收入(%)
    'OETTR':indicator.operating_expense_to_total_revenue, # 营业费用/营业总收入
    'GETTR':indicator.ga_expense_to_total_revenue, # 管理费用/营业总收入(%)
    'FETTR':indicator.financing_expense_to_total_revenue, # 财务费用/营业总收入(%)	
    
    'OPTP':indicator.operating_profit_to_profit, # 经营活动净收益/利润总额(%)
    'IPTP':indicator.invesment_profit_to_profit, # 价值变动净收益/利润总额(%)
    'GSASTR':indicator.goods_sale_and_service_to_revenue, # 销售商品提供劳务收到的现金/营业收入(%)
    'OTR':indicator.ocf_to_revenue, # 经营活动产生的现金流量净额/营业收入(%)
    'OTOP':indicator.ocf_to_operating_profit, # 经营活动产生的现金流量净额/经营活动净收益(%)
    
    'ITRYOY':indicator.inc_total_revenue_year_on_year, # 营业总收入同比增长率(%)
    'ITRA':indicator.inc_total_revenue_annual, # 营业总收入环比增长率(%)
    'IRYOY':indicator.inc_revenue_year_on_year, # 营业收入同比增长率(%)
    'IRA':indicator.inc_revenue_annual, # 营业收入环比增长率(%)
    'IOPYOY':indicator.inc_operation_profit_year_on_year, # 营业利润同比增长率(%)
    'IOPA':indicator.inc_operation_profit_annual, # 营业利润环比增长率(%)
    'INPYOY':indicator.inc_net_profit_year_on_year, # 净利润同比增长率(%)
    'INPA':indicator.inc_net_profit_annual, # 净利润环比增长率(%)
    'INPTSYOY':indicator.inc_net_profit_to_shareholders_year_on_year, # 归属母公司股东的净利润同比增长率(%)
    'INPTSA':indicator.inc_net_profit_to_shareholders_annual, # 归属母公司股东的净利润环比增长率(%)
    'INPTSA':indicator.inc_net_profit_to_shareholders_annual, # 归属母公司股东的净利润环比增长率(%)
    
    
    'ROIC':(income.net_profit+income.financial_expense+income.income_tax_expense)/(balance.total_owner_equities+balance.shortterm_loan+balance.non_current_liability_in_one_year+balance.longterm_loan+balance.bonds_payable+balance.longterm_account_payable),
    'OPTT':income.operating_profit / income.total_profit, # 营业利润占比
    'TP/TOR':income.total_profit / income.total_operating_revenue, #利润总额/营业总收入
    'OP/TOR':income.operating_profit / income.total_operating_revenue,
    'NP/TOR':income.net_profit / income.total_operating_revenue,

    'NP':income.net_profit, # 净利润
    
    'TA':balance.total_assets, # 总资产

    'DER':balance.total_liability / balance.equities_parent_company_owners, # 产权比率 = 负债合计/归属母公司所有者权益合计
    'FCFF/TNCL':(cash_flow.net_operate_cash_flow - cash_flow.net_invest_cash_flow) / balance.total_non_current_liability, #自由现金流比非流动负债
    'NOCF/TL': cash_flow.net_operate_cash_flow / balance.total_liability, # 经营活动产生的现金流量净额/负债合计
    'TCA/TCL':balance.total_current_assets / balance.total_current_liability, # 流动比率

    'PE':valuation.pe_ratio, # PE 市盈率
    'PB':valuation.pb_ratio, # PB 市净率
    'PR':valuation.pcf_ratio, # PR 市现率
    'PS':valuation.ps_ratio, # PS 市销率
    
    'TOR/TA':income.total_operating_revenue / balance.total_assets, #总资产周转率
    'TOR/FA':income.total_operating_revenue / balance.fixed_assets, #固定资产周转率
    'TOR/TCA':income.total_operating_revenue / balance.total_current_assets, #流动资产周转率
    'LTL/OC':balance.longterm_loan / income.operating_cost, #长期借款/营业成本
    
    'TL/TA':balance.total_liability / balance.total_assets, #总资产/总负债
    'TL/TOE':balance.total_liability / balance.total_owner_equities,#负债权益比
    
    }

adjust_factors = {
    'TOR/TA':income.total_operating_revenue / balance.total_assets, #总资产周转率
    'TOR/FA':income.total_operating_revenue / balance.fixed_assets, #固定资产周转率
    'TOR/TCA':income.total_operating_revenue / balance.total_current_assets, #流动资产周转率
    'LTL/OC':balance.longterm_loan / income.operating_cost, #长期借款/营业成本
    
    'TL/TA':balance.total_liability / balance.total_assets, #总资产/总负债
    'TL/TOE':balance.total_liability / balance.total_owner_equities,#负债权益比
    
    'DER':balance.total_liability / balance.equities_parent_company_owners, # 产权比率 = 负债合计/归属母公司所有者权益合计
    'FCFF/TNCL':(cash_flow.net_operate_cash_flow - cash_flow.net_invest_cash_flow) / balance.total_non_current_liability, #自由现金流比非流动负债
    'NOCF/TL': cash_flow.net_operate_cash_flow / balance.total_liability, # 经营活动产生的现金流量净额/负债合计
    'TCA/TCL':balance.total_current_assets / balance.total_current_liability, # 流动比率
    
    'ROIC':(income.net_profit+income.financial_expense+income.income_tax_expense)/(balance.total_owner_equities+balance.shortterm_loan+balance.non_current_liability_in_one_year+balance.longterm_loan+balance.bonds_payable+balance.longterm_account_payable),
    'OPTT':income.operating_profit / income.total_profit, # 营业利润占比
    'TP/TOR':income.total_profit / income.total_operating_revenue, #利润总额/营业总收入
    'OP/TOR':income.operating_profit / income.total_operating_revenue,
    'NP/TOR':income.net_profit / income.total_operating_revenue,
    
    'NOCF/MC':cash_flow.net_operate_cash_flow / valuation.market_cap, #经营活动产生的现金流量净额/总市值
    'GPOA':indicator.gross_profit_margin * income.operating_revenue / balance.total_assets,  #毛利润 / 总资产 = 毛利率*营业收入 / 总资产
    'OPM':income.operating_profit / income.operating_revenue, #营业利润率
    'EBIT':income.net_profit+income.financial_expense+income.income_tax_expense,

}
#获取所有因子列表
factor_list = list(fac_dict.keys())


# In[5]:


def get_fundamental_data(securities,factor_list,ttm_factors, date):
    '''
    获取基本面数据,横截面数据，时间、股票、因子三个参数确定
    获取的数据中含有Nan值，一般用行业均值填充
    输入：
    factor_list:list, 普通因子
    ttm_factors:list, ttm因子，获取过去四个季度财报数据的和
    date:str 或者 datetime.data, 获取数据的时间
    securities：list,查询的股票
    输出：
    DataFrame,普通因子和ttm因子的合并，index为股票代码，values为因子值
    '''
    if len(factor_list) == 0:
        return 'factors list is empty, please input data'
    #获取查询的factor list
    q = query(valuation.code)
    for fac in factor_list:
        q = q.add_column(fac_dict[fac])
    q = q.filter(valuation.code.in_(securities))
    fundamental_df = get_fundamentals(q,date)
    fundamental_df.index = fundamental_df['code']
    fundamental_df.columns = ['code'] + factor_list

    if type(date) == str:
        year = int(date[:4])
        month_day = date[5:]
    elif type(date) == datetime.date:
        date = date.strftime('%Y-%m-%d')
        year = int(date[:4])
        month_day = date[5:]
    else:
        return 'input date error'
    
    if month_day < '05-01':
        statdate_list = [str(year-2)+'q4', str(year-1)+'q1', str(year-1)+'q2', str(year-1)+'q3']
    elif month_day >= '05-01' and month_day < '09-01':
        statdate_list = [str(year-1)+'q1', str(year-1)+'q2', str(year-1)+'q3',str(year)+'q1']
    elif month_day >= '09-01' and month_day < '11-01':
        statdate_list = [str(year-1)+'q2', str(year-1)+'q3', str(year)+'q1', str(year)+'q2']
    elif month_day >= '11-01':
        statdate_list = [str(year-1)+'q4', str(year)+'q1', str(year)+'q2', str(year)+'q3']
            
    ttm_fundamental_data = ''
   
    ttm_q = query(valuation.code)
    for fac in ttm_factors:
        ttm_q = ttm_q.add_column(fac_dict[fac])
    ttm_q = ttm_q.filter(valuation.code.in_(securities))  
                             
    for statdate in statdate_list:
        if type(ttm_fundamental_data) == str:
            fundamental_data = get_fundamentals(ttm_q, statDate=statdate)
            fundamental_data.index = fundamental_data['code']
            ttm_fundamental_data = fundamental_data
        else:
            fundamental_data = get_fundamentals(ttm_q, statDate=statdate)
            fundamental_data.index = fundamental_data['code']
            ttm_fundamental_data.iloc[:,1:] += fundamental_data.iloc[:,1:]
    ttm_fundamental_data.columns = ['code'] + ttm_factors
    results = pd.merge(fundamental_df,ttm_fundamental_data,on=['code'],how='inner')
    results = results.sort_values(by='code')
    results.index = results['code']
    results = results.drop(['code'],axis=1)
    #删除非数值列
    columns = list(results.columns)
    for column in columns:
        if not(isinstance(results[column][0],int) or isinstance(results[column][0],float)):
            results = results.drop([column],axis=1)
    return results


# In[6]:


def get_all_fundamentals(securities, date):
    '''
    获取所有基本面因子
    输入：
    securies:list,查询的股票代码
    date:str or datetime，查询的时间
    输出：
    fundamentals:dataframe,index为股票代码，values为因子值
    '''
    q = query(valuation,balance,cash_flow,income,indicator).filter(valuation.code.in_(securities))
    fundamentals = get_fundamentals(q,date)
    fundamentals.index = fundamentals['code']
    #删除非数值列
    columns = list(fundamentals.columns)
    for column in columns:
        if not(isinstance(fundamentals[column][0],int) or isinstance(fundamentals[column][0],float)):
            fundamentals = fundamentals.drop([column],axis=1)
    fundamentals = fundamentals.sort_index()
    return fundamentals
all_fundamentals = get_all_fundamentals(all_stocks,start_date)


# In[7]:


def get_stock_industry(industry_name,date,output_csv = False):
    '''
    获取股票对应的行业
    input：
    industry_name: str, 
    "sw_l1": 申万一级行业
    "sw_l2": 申万二级行业
    "sw_l3": 申万三级行业
    "jq_l1": 聚宽一级行业
    "jq_l2": 聚宽二级行业
    "zjw": 证监会行业
    date:时间
    output: DataFrame,index 为股票代码，columns 为所属行业代码
    '''
    industries = list(get_industries(industry_name).index)
    all_securities = get_all_securities(date=date)   #获取当天所有股票代码
    all_securities['industry_code'] = 1
    for ind in industries:
        industry_stocks = get_industry_stocks(ind,date)
        #有的行业股票不在all_stocks列表之中
        industry_stocks = set(all_securities) & set(industry_stocks)
        all_securities['industry_code'][industry_stocks] = ind
    stock_industry = all_securities['industry_code'].to_frame()
    if output_csv == True:
        stock_industry.to_csv('stock_industry.csv') #输出csv文件，股票对应行业
    return stock_industry


# In[8]:


def fillna_with_industry(data,date,industry_name='sw_l1'):
    '''
    使用行业均值填充nan值
    input:
    data：DataFrame,输入数据，index为股票代码
    date:string,时间必须和data数值对应时间一致
    output：
    DataFrame,缺失值用行业中值填充，无行业数据的用列均值填充
    '''
    stocks = list(data.index)
    stocks_industry = get_stock_industry(industry_name,date)
    stocks_industry_merge = data.merge(stocks_industry, left_index=True,right_index=True,how='left')
    stocks_dropna = stocks_industry_merge.dropna()
    columns = list(data.columns)
    select_data = []
    group_data = stocks_industry_merge.groupby('industry_code')
    group_data_mean = group_data.mean()
    group_data = stocks_industry_merge.merge(group_data_mean,left_on='industry_code',right_index=True,how='left')
    for column in columns:

        if type(data[column][0]) != str:

            group_data[column+'_x'][pd.isnull(group_data[column+'_x'])] = group_data[column+'_y'][pd.isnull(group_data[column+'_x'])]
            
            group_data[column] = group_data[column+'_x']
            #print(group_data.head())
            select_data.append(group_data[column])
            
    result = pd.concat(select_data,axis=1)
    #行业均值为Nan,用总体均值填充
    mean = result.mean()
    for i in result.columns:
        result[i].fillna(mean[i],inplace=True)
    return result


# # 工具函数

# In[9]:


#获取日期列表
def get_tradeday_list(start,end,frequency=None,count=None):
    '''
    input:
    start:str or datetime,起始时间，与count二选一
    end:str or datetime，终止时间
    frequency:
        str: day,month,quarter,halfyear,默认为day
        int:间隔天数
    count:int,与start二选一，默认使用start
    '''
    if isinstance(frequency,int):
        all_trade_days = get_trade_days(start,end)
        trade_days = all_trade_days[::frequency]
        days = [datetime.datetime.strftime(i,'%Y-%m-%d') for i in trade_days]
        return days
    
    if count != None:
        df = get_price('000001.XSHG',end_date=end,count=count)
    else:
        df = get_price('000001.XSHG',start_date=start,end_date=end)
    if frequency == None or frequency =='day':
        days = df.index
    else:
        df['year-month'] = [str(i)[0:7] for i in df.index]
        if frequency == 'month':
            days = df.drop_duplicates('year-month').index
        elif frequency == 'quarter':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month']=='01') | (df['month']=='04') | (df['month']=='07') | (df['month']=='10') ]
            days = df.drop_duplicates('year-month').index
        elif frequency =='halfyear':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month']=='01') | (df['month']=='06')]
            days = df.drop_duplicates('year-month').index
    trade_days = [datetime.datetime.strftime(i,'%Y-%m-%d') for i in days]
    return trade_days
tl = get_tradeday_list(start_date,end_date,frequency='month')


# In[10]:


def get_date_list(begin_date, end_date):
    '''
    得到datetime类型时间序列
    '''
    dates = []
    dt = datetime.datetime.strptime(begin_date,"%Y-%m-%d")
    date = begin_date[:]
    while date <= end_date:
        dates.append(date)
        dt += datetime.timedelta(days=1)
        date = dt.strftime("%Y-%m-%d")
    return dates

#去极值函数
#mad中位数去极值法
def filter_extreme_MAD(series,n): #MAD: 中位数去极值 
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n*new_median
    min_range = median - n*new_median
    return np.clip(series,min_range,max_range)

#进行标准化处理
def winsorize(factor, std=3, have_negative = True):
    '''
    去极值函数 
    factor:以股票code为index，因子值为value的Series
    std为几倍的标准差，have_negative 为布尔值，是否包括负值
    输出Series
    '''
    r=factor.dropna().copy()
    if have_negative == False:
        r = r[r>=0]
    else:
        pass
    #取极值
    edge_up = r.mean()+std*r.std()
    edge_low = r.mean()-std*r.std()
    r[r>edge_up] = edge_up
    r[r<edge_low] = edge_low
    return r

#标准化函数：
def standardize(s,ty=2):
    '''
    s为Series数据
    ty为标准化类型:1 MinMax,2 Standard,3 maxabs 
    '''
    data=s.dropna().copy()
    if int(ty)==1:
        re = (data - data.min())/(data.max() - data.min())
    elif ty==2:
        re = (data - data.mean())/data.std()
    elif ty==3:
        re = data/10**np.ceil(np.log10(data.abs().max()))
    return re

#数据去极值及标准化
def winsorize_and_standarlize(data,qrange=[0.05,0.95],axis=0):
    '''
    input:
    data:Dataframe or series,输入数据
    qrange:list,list[0]下分位数，list[1]，上分位数，极值用分位数代替
    '''
    if isinstance(data,pd.DataFrame):
        if axis == 0:
            q_down = data.quantile(qrange[0])
            q_up = data.quantile(qrange[1])
            index = data.index
            col = data.columns
            for n in col:
                data[n][data[n] > q_up[n]] = q_up[n]
                data[n][data[n] < q_down[n]] = q_down[n]
            data = (data - data.mean())/data.std()
            data = data.fillna(0)
        else:
            data = data.stack()
            data = data.unstack(0)
            q = data.quantile(qrange)
            index = data.index
            col = data.columns
            for n in col:
                data[n][data[n] > q[n]] = q[n]
            data = (data - data.mean())/data.std()
            data = data.stack().unstack(0)
            data = data.fillna(0)
            
    elif isinstance(data,pd.Series):
        name = data.name
        q = data.quantile(qrange)
        data = np.clip(data,q.values[0],q.values[1])
        data = (data - data.mean())/data.std()
    return data
    
def neutralize(data,date,market_cap,industry_name='sw_l1'):
    '''
    中性化，使用行业和市值因子中性化
    input：
    data：DataFrame,index为股票代码，columns为因子，values为因子值
    name:str,行业代码
    "sw_l1": 申万一级行业
    "sw_l2": 申万二级行业
    "sw_l3": 申万三级行业
    "jq_l1": 聚宽一级行业
    "jq_l2": 聚宽二级行业
    "zjw": 证监会行业
    date:获取行业数据的时间
    maket_cap:市值因子
    '''
    industry_se = get_stock_industry(industry_name,date)
    columns = list(data.columns)
    if isinstance(industry_se,pd.Series):
        industry_se = industry_se.to_frame()
    if isinstance(market_cap,pd.Series):
        market_cap = market_cap.to_frame()
        
    index = list(data.index)
    industry_se = np.array(industry_se.ix[index,0].tolist())
    industry_dummy = sm.categorical(industry_se,drop=True)
    industry_dummy = pd.DataFrame(industry_dummy,index=index)
    market_cap = np.log(market_cap.loc[index])
    x = pd.concat([industry_dummy,market_cap],axis=1)
    model = sm.OLS(data,x)
    result = model.fit()
    y_fitted =  result.fittedvalues
    neu_result = data - y_fitted
    return neu_result


# # 计算收益

# In[11]:


def get_month_profit(stocks,start_date,end_date,month_num=1,cal_num=3):
    '''
    获取月收益率数据，数据为本月相对于上月的增长率
    input:
    stocks:list 股票代码
    start_date:str, 初始日期
    end_date:str,终止日期
    month_num:计算几个月的收益率，默认为1，即一个月的收益率
    cal_num:int，计算每月最后n天的收盘价均值，默认为3
    
    '''
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    start_month = int(start_date[5:7])
    end_month = int(end_date[5:7])
    len_month = (end_year - start_year)*12 + (end_month - start_month)
    price_list = []
    #获取初始时间之前一个月的价格数据
    if start_month == 1:
        last_date = str(start_year-1)+'-'+'12'+'-'+'01'
    else:
        last_date = str(start_year-1)+'-'+str(start_month-1)+'-'+'01'
    last_price = get_price(stocks,fields=['close'],count=cal_num,end_date=last_date)['close']
    last_price = last_price.mean().to_frame()
    last_price.columns = [last_date]
    price_list.append(last_price)
    #计算给定时间段内的月度价格数据
    for i in range(len_month):
        date = str(start_year+i//12)+'-'+str(start_month+i%12)+'-'+'01'
        price = get_price(stocks,fields=['close'],count=cal_num,end_date=date)['close']
        price_mean = price.mean().to_frame()
        price_mean.columns = [date]
        price_list.append(price_mean)
    month_profit = pd.concat(price_list,axis=1)
    #计算月度收益率
    month_profit_pct = month_profit.pct_change(month_num,axis=1).dropna(axis=1,how='all')
    return month_profit_pct


# In[12]:


def get_profit_depend_timelist(stocks,timelist,month_num=1,cal_num=3):
    '''
    input:
    stocks:list 股票代码
    timelist: 时间序列
    month_num:计算几个月的收益率，默认为1，即一个月的收益率
    cal_num:int，计算每月最后n天的收盘价均值，默认为3
    '''
    price_list = []
    for date in timelist:
        price = get_price(stocks,fields=['close'],count=cal_num,end_date=date)['close']
        price_mean = price.mean().to_frame()
        price_mean.columns = [date]
        price_list.append(price_mean)
    profit = pd.concat(price_list,axis=1)
    profit_pct = profit.pct_change(month_num,axis=1).dropna(axis=1,how='all')
    return profit_pct


# In[13]:


def get_day_profit_forward(stocks,end_date,start_date=None,count=-1,pre_num=20):
    '''
    获取收益率,pre_num为计算时间差，在时间轴上的当期值是未来计算周期内的收益率，
    例如：pre_num=3,2013-01-01对应的收益率是2013-01-04的收益率与01-01日收益率之差
    input：
    stocks:list or Series,股票代码
    start_date:开始时间
    end_date：结束时间
    count:与start_date二选一，向前取值个数
    pre_num:int,向后计算的天数
    output:
    profit:dataframe,index为日期，columns为股票代码，values为收益率
    '''
    if count == -1:
        price = get_price(stocks,start_date,end_date,fields=['close'])['close']
        date_list = get_trade_days(start_date=start_date,end_date=end_date)
        price.index = date_list

    else:
        price = get_price(stocks,end_date=end_date,count=count,fields=['close'])['close']
        date_list = get_trade_days(end_date=end_date,count=count)
        price.index = date_list
    profit = price.pct_change(periods=pre_num).shift(-pre_num).dropna()
    return profit


# # 获取数据

# In[14]:


def get_one_day_data(stocks,factor_list,ttm_factors,date,neu=False):
    '''
    获取一天的基本面数据
    input:
    stocks:list,股票列表
    factor_list:list,普通因子列表
    ttm_factors:list,ttm因子列表
    date:str or datetime， 获取数据时间
    neu:bool,是否进行中性化处理，使用市值和行业进行中性化，默认不进行中性化
    '''
    fund_data = get_fundamental_data(stocks,factor_list,ttm_factors,date)
    fillna_data = fillna_with_industry(fund_data,date)
    if neu == False:
        results = winsorize_and_standarlize(fillna_data)
    elif 'MC' in fillna_data.columns:
        neu_data = neutralize(fillna_data,date,fillna_data['MC'])
        results = winsorize_and_standarlize(neu_data)
    elif 'market_cap' in fillna_data.columns:
        neu_data = neutralize(fillna_data,date,fillna_data['market_cap'])
        results = winsorize_and_standarlize(neu_data)
    else:
        print("error: please input 'market_cap' for neutralize")
        return None
    return results


# In[15]:


def get_timelist_data(stocks,factor_list,ttm_factors,timelist,neu=False):
    dic = {}
    for date in timelist:
        fund_date = get_one_day_data(stocks,factor_list,ttm_factors,date,neu=neu)
        dic[date] = fund_date
    return dic


# In[16]:


fund_data = get_timelist_data(all_stocks,factor_list,ttm_factors,tl)
fund_data_neu = get_timelist_data(all_stocks,factor_list,ttm_factors,tl,neu=True)
profit =  get_profit_depend_timelist(all_stocks,tl,month_num=2,cal_num=3)
res = []
res.append(fund_data)
res.append(fund_data_neu)
res.append(profit)


# In[18]:


#将数据输出到pickle文件
with open('fundamental_data.pkl','wb') as pk_file:
    pickle.dump(res,pk_file)


# # 第二模块 因子选取

# In[7]:


import numpy as np
import pandas as pd
import pickle
import datetime
import statsmodels.api as sm
import warnings
from sklearn.feature_selection import RFE,SelectKBest,SelectPercentile,SelectFromModel,f_classif
import lightgbm as lgb 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import gc
from sklearn.metrics import accuracy_score,recall_score

warnings.filterwarnings('ignore')


# In[8]:


with open('fundamental_data.pkl','rb') as pk_file:
    data_pk = pickle.load(pk_file)
fund_data = data_pk[0]
fund_data_neu = data_pk[1]
profit = data_pk[2]
keys = list(fund_data.keys())


# In[1]:


#截面数据，将profit与基本面数据对齐，基本面数据对应下一月的profit
def get_fund_profit_data(fund_data,profit):
    '''
    input:
    fund_data:dic key为日期，values为dataframe，基本面数据，index为股票代码，columns为因子
    profit:dataframe,index为股票代码，columns为时间
    注意：此函数针对于fund_data keys日期与profit日期在位置上已经对应
    '''
    keys = list(fund_data.keys())
    columns = profit.columns.tolist()
    l = min(len(keys),len(columns))
    fund_profit = {}
    for i in range(l):
        fd = fund_data[keys[i]].copy() #复制新的dataframe，否则fund_profit为引用，在fund_profit上修改值会直接影响到fund_data
        p = profit[columns[i]].to_frame()
        p.columns = ['profit']
        fd_merge = pd.merge(fd,p,left_index=True,right_index=True,how='inner')
        fund_profit[keys[i]] = fd_merge
    return fund_profit


# In[2]:


def get_fund_profit_class_data(fund_data,profit):
    '''
    profit不再是数值，而是类别，大于0标记为1，小于0标记为0
    input:
    fund_data:dic key为日期，values为dataframe，基本面数据，index为股票代码，columns为因子
    profit:dataframe,index为股票代码，columns为时间
    output:
    fund_profit:dic, 在fund_data每个dataframe后面加了profit列
    注意：此函数针对于fund_data keys日期与profit日期在位置上已经对应
    '''
    pf = profit.copy(deep=True)
    pf[pf>0] = 1
    pf[pf<0] = 0
    keys = list(fund_data.keys())
    columns = pf.columns.tolist()
    l = min(len(keys),len(columns))
    fund_profit = {}
    for i in range(l):
        fd = fund_data[keys[i]].copy() #复制新的dataframe，否则fund_profit为引用，在fund_profit上修改值会直接影响到fund_data
        p = pf[columns[i]].to_frame()
        p.columns = ['profit']
        fd_merge = pd.merge(fd,p,left_index=True,right_index=True,how='inner')
        fund_profit[keys[i]] = fd_merge
    return fund_profit


# In[9]:


fund_profit_data = get_fund_profit_data(fund_data,profit)
fund_profit_data_neu = get_fund_profit_data(fund_data_neu,profit)
fund_profit_class_data = get_fund_profit_class_data(fund_data,profit)
fund_profit_class_data_neu = get_fund_profit_class_data(fund_data_neu,profit)


# # 因子评判函数

# In[23]:


#使用稳健回归（sm.RLM）robust linear model
'''
用回归系数作为因子有效性的指标，如果因子与收益之间非线性，则此指标不准确，此指标作为参考之一
'''
def get_RLM_res(fund_profit_data):
    '''
    input:
    fund_profit_data:dic, keys为日期，values为dataframe，基本面数据，index为股票代码，columns为因子,columns最后一列为profit
    output:
    f: dataframe, index为因子，columns为时间，values为稳健回归系数
    t: dataframe,index为因子，columns为时间，values为稳健回归系数的t值
    '''
    keys = fund_profit_data.keys()
    
    f_dic = {}
    t_dic = {}
    f = []
    t = []
    for k in keys:
        col = fund_profit_data[k].columns
        f_list = []
        t_list = []
        for c in col[:-1]:
            df = fund_profit_data[k]
            rlm_model = sm.RLM(df[col[-1]],df[c],M=sm.robust.norms.HuberT()).fit()
            f_list.append(rlm_model.params)
            t_list.append(rlm_model.tvalues)
        f_list = [f_list[i].values[0] for i in range(len(f_list))]
        t_list = [t_list[i].values[0] for i in range(len(t_list))]
        f_df_k = pd.DataFrame(f_list,index=list(col[:-1]),columns=[k])
        t_df_k = pd.DataFrame(t_list,index=col[:-1],columns=[k])
        f.append(f_df_k)
        t.append(t_df_k)
    f = pd.concat(f,axis=1)
    t = pd.concat(t,axis=1)
    return f,t


# In[24]:


rlm_res = get_RLM_res(fund_profit_data)
rlm_neu_res = get_RLM_res(fund_profit_data_neu)


# In[25]:


'''
信息系数IC值，可以有效的观察到某个因子收益率预测的稳定性和动量特征，以便在组合优化时用作筛选的指标。常见的IC值计算方法有两种：
相关系数（Pearson Correlation）和秩相关系数（Spearman Rank Correlation）,此例中IC值统计用到的是秩相关系数，
与IC相关的用来判断因子的有效性和预测能力指标如下：
IC值的均值
IC值的标准差
IC值大于0的比例
IC绝对值大于0.02的比例
IR （IC均值与IC标准差的比值）
参考：https://www.joinquant.com/post/16105?tag=algorithm
'''
def get_IC(fund_profit_data):
    '''
    input:
    fund_profit_data:dic, keys为日期，values为dataframe，基本面数据，index为股票代码，columns为因子,columns最后一列为profit
    output:
    p: dataframe, index为因子，columns为时间，values为pearson相关系数
    s: dataframe,index为因子，columns为时间，values为spearman相关系数
    '''
    keys = fund_profit_data.keys()
    p_dic = {}
    s_dic = {}
    p = []
    s = []
    for k in keys:
        columns = fund_profit_data[k].columns
        p_list = []
        s_list = []
        for c in columns[:-1]:
            df = fund_profit_data[k]
            p_value = df[c].corr(df[columns[-1]],method='pearson')
            s_value = df[c].corr(df[columns[-1]],method='spearman')
            p_list.append(p_value)
            s_list.append(s_value)

        p_df_k = pd.DataFrame(p_list,index=list(columns[:-1]),columns=[k])
        s_df_k = pd.DataFrame(s_list,index=columns[:-1],columns=[k])
        p.append(p_df_k)
        s.append(s_df_k)
    p = pd.concat(p,axis=1)
    s = pd.concat(s,axis=1)
    return p,s


# In[26]:


ic_res = get_IC(fund_profit_data)
ic_res_neu = get_IC(fund_profit_data_neu)


# In[27]:


#好不容易跑完的数据，赶紧保存一下子
res = []
res.append(rlm_res)
res.append(rlm_neu_res)
res.append(ic_res)
res.append(ic_res_neu)
with open('judge_data.pkl','wb') as pf:
    pickle.dump(res,pf)


# In[28]:


#计算IC的各个指标
def cal_IC_indicator(data):
    '''
    input:
    data:dataframe,index为因子，columns为日期
    output:
    res:dataframe，index为因子，输出计算好的各个评判标准
    '''
    data = data.stack().unstack(0)
    data_mean = data.mean()
    data_std = data.std()
    data_ir = data_mean / data_std
    
    data_mean_df = data_mean.to_frame()
    data_mean_df.columns = ['mean']
    data_std_df = data_std.to_frame()
    data_std_df.columns = ['std']
    data_ir_df = data_ir.to_frame()
    data_ir_df.columns = ['IR']
    
    data_nagative = (data[data > 0]).count() / len(data)
    data_nagative_df = data_nagative.to_frame()
    data_nagative_df.columns = ['IC正值比例']
    data_abs_dayu = (data[data.abs() > 0.02]).count() / len(data)
    data_abs_dayu_df = data_abs_dayu.to_frame()
    data_abs_dayu_df.columns = ['IC绝对值>0.02']
    res = pd.concat([data_mean_df,data_std_df,data_nagative_df,data_abs_dayu_df,data_ir_df],axis=1)
    return res    
    


# In[29]:


ic_indicator_pearson = cal_IC_indicator(ic_res[0])


# # 特征选择方法

# In[ ]:


class FeatureSelection():
    '''
    特征选择：
    identify_collinear：基于相关系数，删除小于correlation_threshold的特征
    identify_importance_lgbm：基于LightGBM算法，得到feature_importance,选择和大于p_importance的特征
    filter_select:单变量选择，指定k,selectKBest基于method提供的算法选择前k个特征，selectPercentile选择前p百分百的特征
    wrapper_select:RFE，基于estimator递归特征消除，保留n_feature_to_select个特征
    embedded_select： 基于estimator，
    
    '''
    def __init__(self):
        self.supports = None #bool型，特征是否被选中
        self.columns = None  #选择的特征
        self.record_collinear = None #自相关矩阵大于门限值
        self.feature_importances = None #lgbm算法保存特征重要性值
        
    def identify_collinear(self, data, correlation_threshold):
        """
        Finds collinear features based on the correlation coefficient between features. 
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal. 

        Using code adapted from: https://gist.github.com/Swarchal/e29a3a1113403710b6850590641f046c
        
        Parameters
        --------

        data : dataframe
            Data observations in the rows and features in the columns

        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation cofficient for identifying correlation features

        """
        columns = data.columns
        self.correlation_threshold = correlation_threshold

        # Calculate the correlations between every column
        corr_matrix = data.corr()
        
        self.corr_matrix = corr_matrix
    
        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
        obtain_columns = [column for column in columns if column not in to_drop]
        self.columns = obtain_columns
        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop
        for column in to_drop:

            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]    

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                             'corr_feature': corr_features,
                                             'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index = True)

        self.record_collinear = record_collinear
        return data[obtain_columns]
     
        
    def identify_importance_lgbm(self, features, labels,p_importance=0.8, eval_metric='auc', task='classification', 
                                 n_iterations=10, early_stopping = True):
       

        # One hot encoding
        data = features
        features = pd.get_dummies(features)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        features = np.array(features)
        labels = np.array(labels).reshape((-1, ))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))
        
        print('Training Gradient Boosting Model\n')
        
        # Iterate through each fold
        for _ in range(n_iterations):

            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=100, learning_rate = 0.05, verbose = -1)

            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=100, learning_rate = 0.05, verbose = -1)

            else:
                raise ValueError('Task must be either "classification" or "regression"')
                
            # If training using early stopping need a validation set
            if early_stopping:
                
                train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size = 0.15)

                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric = eval_metric,
                          eval_set = [(valid_features, valid_labels)],
                           verbose = -1)
                
                # Clean up memory
                gc.enable()
                del train_features, train_labels, valid_features, valid_labels
                gc.collect()
                
            else:
                model.fit(features, labels)

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
        
        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)
        
        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
        #obtain feature importance
        self.feature_importances = feature_importances
        select_df = feature_importances[feature_importances['cumulative_importance']<=p_importance]
        select_columns = select_df['feature']
        self.columns = list(select_columns.values)
        res = data[self.columns]
        return res
        
    def filter_select(self, data_x, data_y, k=None, p=50,method=f_classif):
        columns = data_x.columns
        if k != None:
            model = SelectKBest(method,k)
            res = model.fit_transform(data_x,data_y)
            supports = model.get_support()
        else:
            model = SelectPercentile(method,p)
            res = model.fit_transform(data_x,data_y)
            supports = model.get_support()
        self.support_ = supports
        self.columns = columns[supports]
        return res
    
    def wrapper_select(self,data_x,data_y,n,estimator):
        columns = data_x.columns
        model = RFE(estimator=estimator,n_features_to_select=n)
        res = model.fit_transform(data_x,data_y)
        supports = model.get_support() #标识被选择的特征在原数据中的位置
        self.supports = supports
        self.columns = columns[supports]
        return res
    
    def embedded_select(self,data_x,data_y,estimator,threshold=None):
        columns = data_x.columns
        model = SelectFromModel(estimator=estimator,prefit=False,threshold=threshold)
        res = model.fit_transform(data_x,data_y)
        supports = model.get_support()
        self.supports = supports
        self.columns = columns[supports]
        return res


# In[ ]:


#使用特征选择方法选择因子值
test_fund = fund_profit_class_data_neu[keys[0]] #取一期数据测试
test_fund1 = fund_profit_class_data_neu[keys[1]]
#test_fund = pd.concat([test_fund0,test_fund1])
columns = test_fund.columns
fs = FeatureSelection()
x = test_fund[columns[:-1]]
y = test_fund[columns[-1]]

lgbm = fs.identify_importance_lgbm(x,y) #使用特征选择方法选择有效因子
fs.feature_importances #各个因子重要性


# # 使用机器学习算法探索选股或择时策略（示例）

# In[ ]:


'''
以下代码是示例代码，简单走一遍机器学习探索策略及调参，具体有效的策略请大家自己探索，不做分享
'''

fund_data_train = fund_profit_class_data_neu[keys[1]]
columns_s  = lgbm.columns
col_s = columns_s[:-1]
fund_data_train_y = fund_data_train[fund_data_train.columns[-1]]
lgbm_x_train,lgbm_x_test,lgbm_y_train,lgbm_y_test = train_test_split(fund_data_train[col_s],fund_data_train_y,test_size=0.3)


# In[ ]:



lgbm_svm = SVC(max_iter=1000)
param_grid = {'C':[0.1,1,3],'kernel':['rbf','sigmoid','linear','poly'],'gamma':np.arange(0.3,0.8,0.1)}
lgbm_model = GridSearchCV(estimator=lgbm_svm,param_grid=param_grid,scoring='accuracy')
lgbm_model.fit(lgbm_x_train,lgbm_y_train)
lgbm_test_res = lgbm_model.predict(lgbm_x_test)
accuracy = accuracy_score(lgbm_y_test,lgbm_test_res)
print('accuracy is: %0.5f'%accuracy)
print(lgbm_model.best_params_)
print(lgbm_model.best_score_)


# In[ ]:


gbdt = GradientBoostingClassifier()
gbdt_params_grid = {'max_depth':[4,6,8],'min_samples_split':[10,20,30]}
gbdt_model = GridSearchCV(estimator=gbdt,param_grid=gbdt_params_grid)
gbdt_model.fit(lgbm_x_train,lgbm_y_train)
gbdt_test_res = gbdt_model.predict(lgbm_x_test)
gbdt_accuracy = accuracy_score(lgbm_y_test,gbdt_test_res)
print('accuracy is: %0.5f'%gbdt_accuracy)
print(gbdt_model.best_params_)
print(gbdt_model.best_score_)


# In[ ]:


fund_for_pre = fund_profit_class_data_neu[keys[3]] #取一期的截面数据验证
columns_for_pre = fund_for_pre.columns
x_for_pre = fund_for_pre[col_s]
y_for_pre = fund_for_pre[columns_for_pre[-1]]
prediction = lgbm_model.predict(x_for_pre)
accuracy_for_pre = accuracy_score(y_for_pre,prediction)
print(accuracy_for_pre)
print(len(prediction))

