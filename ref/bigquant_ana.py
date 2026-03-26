### 因子分析工具
import pandas as pd 
import numpy as np
import warnings
import empyrical
import dai
import bigcharts
warnings.filterwarnings('ignore')
from biglearning.api import tools as T
print('导入包完成!')
class AlphaMiner(object):
    def __init__(self, params, factor_data):

        # params: 字典格式。 形如  {'group_num':10, 'factor_field':'hf_close_netinflow_rate_small_order_act', 'instruments':'中证500', 'factor_direction':-1, 'benchmark':'中证500'}
        # group_num:分组数量 参数类型：int
        # factor_field:因子在表中所对应的字段名称 参数类型：str
        # instruments:标的池,支持选项：沪深300、中证500、中证1000、全市场 参数类型：str
        # factor_direction：因子方向，字符串格式，取值为1、-1。1表示因子方向为正，因子值越大越好，-1表示因子值为负，因子值越小越好。 参数类型：int
        # benchmark:基准对比指数，支持选项：沪深300、中证500、中证1000 参数类型：str

        # factor_data:pandas.DataFrame格式,形如
        #   instrument	date	    hf_fz_ykws
        # 0	000001.SZ	2017-01-03	1.564644
        # 1	000001.SZ	2017-01-04	1.521567
        # 2	000001.SZ	2017-01-05	1.519973
        # 3	000001.SZ	2017-01-06	1.553225
        # 4	000001.SZ	2017-01-09	1.367971
        # 其中, instrument:str ,以股票代码+.sh（沪市） +.SZ（深市）
        #     date:datetime64 
        #     hf_fz_ykws:float64

        self.params = params
        self.top_n_ins = 5 # 默认5只
        self.factor_data = factor_data.rename(columns={self.params['factor_field']:'factor'}) 
        self.factor_data['factor'] *= self.params['factor_direction']

        if self.params['data_process'] == True:
            self.factor_data = self.factor_data_process('factor')
            print("数据预处理完成")

        # 检查因子数据格式
        try:
            self.check_data_format(self.factor_data)
            print("数据格式检查通过")
        except ValueError as e:
            print("数据格式检查失败：" + str(e))

        # 进行数据池过滤 
        self.stock_pool_filter() 
        self.start_date = self.factor_data.date.min().strftime('%Y-%m-%d')
        self.end_date =  self.factor_data.date.max().strftime('%Y-%m-%d')

        self.price_data =  self.get_daily_ret(self.start_date, self.end_date) # 日收益率数据
        print('个股日收益率计算完成')
        
        self.merge_data = pd.merge(self.factor_data.sort_values(['date', 'instrument']), \
                                   self.price_data.sort_values(['date', 'instrument']), on=['date','instrument'], how='left')
        self.group_data = self.get_group_data()  # 分组数据
        print('因子分组完成')
        self.bm_ret = self.get_bm_ret(self.params['benchmark'])
        print('基准日收益率计算完成')
        self.group_cumret = self.get_group_cumret()  # 分组累积收益率
        print('分组收益率计算完成')
        self.whole_perf = self.get_whole_perf()   # 整体绩效指标 
        print('整体绩效计算完成')
        self.yearly_perf =  self.get_yearly_perf() # 按年度绩效指标
        print('年度绩效计算完成')
        self.ic = self.get_IC_data('all')  # ic指标
        print('IC计算完成')


    def factor_data_process(self, col):
        """因子数据预处理函数，包括去极值、标准化、中性化"""
        
        def zscore(df, train_col):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[train_col] = scaler.fit_transform(df[train_col])
            return df

        def remove_extreme_and_cut_zscore(df, col):
            col_list = [col]
            for fac in col_list:
                n = 3
                mean = df[fac].mean()  # 计算因子值的均值
                std = df[fac].std() # 计算因子值的标准差
                lower_bound = mean - n * std  # 计算下边界
                upper_bound = mean + n * std  # 计算上边界
                df.loc[df[fac]<lower_bound, fac] = lower_bound
                df.loc[df[fac]>upper_bound, fac] = upper_bound
        
            df = zscore(df, col_list)
            return df

        factor_data = self.factor_data.groupby('date').apply(remove_extreme_and_cut_zscore, col=col)
        factor_data = factor_data.fillna(0) # 用0进行填充 

        start_date = factor_data.date.min().strftime("%Y-%m-%d")
        end_date = factor_data.date.max().strftime("%Y-%m-%d")

        sql = """
        SELECT date, instrument, industry_level1_code
        FROM cn_stock_industry_component 
        WHERE date >= '{0}' and date <= '{1}' and industry =='sw2021'
        ORDER BY instrument, date;
        """.format(start_date, end_date)

        df_industry = dai.query(sql).df()
        df_industry = df_industry.dropna()

        factor_data = factor_data.merge(df_industry, on=['date', 'instrument'])
        industry_list = df_industry['industry_level1_code'].unique()

        sql = """
        SELECT date, instrument, total_market_cap
        FROM cn_stock_valuation 
        WHERE date >= '{0}' and date <= '{1}' 
        ORDER BY instrument, date;
        """.format(start_date, end_date)

        df_market_cap = dai.query(sql).df()
        factor_data = factor_data.merge(df_market_cap, on=['date', 'instrument'])

        factor_data['log_cap'] = np.log(factor_data.total_market_cap)
            
        #截面中性化
        def neutralize(df, col):
            import warnings
            warnings.filterwarnings('ignore')
            import statsmodels.api as sm

            col_list = [col]
            for fac in col_list:
                ind_dummies = pd.get_dummies(df['industry_level1_code'], prefix='industry_level1_code')
                mkcap = df['log_cap']
                train = pd.concat([ind_dummies,mkcap],axis=1)
                X = sm.add_constant(train)
                y = df[fac]
                model = sm.OLS(y, X).fit()
                df[fac] = model.resid
            return df

        res =  factor_data.groupby('date').apply(neutralize, col=col)
        res.sort_values(by='date', inplace=True)
        res.reset_index(inplace=True, drop=True)
        return res

    def check_data_format(self, df):
        # 检查date列是否是日期型类型
        if df['date'].dtype != 'datetime64[ns]':
            raise ValueError("date列的数据格式应为datetime格式")
        # 检查instrument列是否是以SZ\SH结尾
        if not all(df['instrument'].str.endswith('.SH') | df['instrument'].str.endswith('.SZ') | df['instrument'].str.endswith('.BJ')):
            raise ValueError("instrument列的数据格式应为以.SH或.SZ或.BJ结尾的字符串")
        # 检查factor列是否是浮点型数值
        if df['factor'].dtype != 'float64':
            raise ValueError("factor列的数据格式应为浮点型")


    def stock_pool_filter(self):
        pools = self.params['instruments']
        if pools == "沪深300":
            index_code = '000300.SH'
        elif pools == "中证500":
            index_code = '000905.SH'
        elif pools == "中证1000":
            index_code = '000852.SH'
        elif pools  == "全市场":
            return
        else: 
            print('请检查输入的指数池是否正确')

        index_com_df  = dai.query("select * from cn_stock_index_component where date >= '2015-01-01' and instrument == '%s' order by date, instrument "%index_code).df()
        factor_df = self.factor_data

        merge_df = pd.merge(factor_df, index_com_df, how='inner', left_on=['date','instrument'], right_on=['date', 'member_code'])[['instrument_x','date','factor']]
        merge_df.rename(columns={'instrument_x':'instrument'}, inplace=True)
        self.factor_data = merge_df

    def get_daily_ret(self, start_date, end_date):
        """计算收益率. T0的因子对应的收益率是T+1日开盘买入,T+2开盘卖出"""
        sql = f"SELECT instrument,date, (m_lead(open, 2)/ m_lead(open, 1) - 1) AS daily_ret from cn_stock_bar1d ORDER BY date, instrument;"
        
        from datetime import datetime, timedelta
        ten_days_ago_start_date = pd.Timestamp(self.start_date) - timedelta(days=10) # 往前多取10天数据
        ten_days_ago_start_date = ten_days_ago_start_date.strftime('%Y-%m-%d')

        price_data = dai.query(sql, filters={"date": [ten_days_ago_start_date, self.end_date]}).df()
        return price_data

    def get_group_data(self):
        """因子分组，因子值越大，组数越大，默认的多头组合是因子数值最大的组合"""
        def cut(df, group_num=10):
            """分组"""
            df = df.drop_duplicates('factor') # 删除重复值
            df['group'] = pd.qcut(df['factor'], q=group_num, labels=False, duplicates='drop')
            df = df.dropna(subset=['group'], how='any')
            df['group'] =  df['group'].apply(int).apply(str)
            return df

        group_data = self.merge_data.groupby('date', group_keys=False).apply(cut, group_num=self.params['group_num'])
        return group_data

    def get_bm_ret(self, benchmark):

        if benchmark == "沪深300":
            bm_code = '000300.SH'
        elif benchmark == "中证500":
            bm_code = '000905.SH'
        elif benchmark == "中证1000":
            bm_code = '000852.SH'
        else: 
            print('请检查输入的基准代码是否正确')

        # 获取基准日收益率数据
        bm_sql = """
        SELECT 
            date,instrument, (close - m_Lag(close,1))  / m_LAG(close, 1) as benchmark_ret
        FROM cn_stock_index_bar1d
        WHERE instrument = '%s'
        AND date >= '%s' and date <='%s' ;"""%(bm_code, self.start_date, self.end_date)

        bm_ret = dai.query(bm_sql).df()
        return bm_ret

    def get_group_cumret(self):
        # 分组收益率
        groupret_data = self.group_data[['date','group','daily_ret']].groupby(['date','group'], group_keys=False).apply(lambda x:np.nanmean(x)).reset_index()
        groupret_data.rename(columns={0:'g_ret'}, inplace=True)

        groupret_pivotdata = groupret_data.pivot(index='date', values='g_ret', columns='group')
        groupret_pivotdata['ls'] = groupret_pivotdata[str(self.params['group_num']-1)] - groupret_pivotdata['0']  # 日收益率

        bm_ret = self.bm_ret.set_index('date') # 基准收益率
        groupret_pivotdata['bm'] = bm_ret['benchmark_ret'] 
        groupret_pivotdata = groupret_pivotdata.shift(1) # 首日为nan，最后一日有值
        self.groupret_pivotdata = groupret_pivotdata
        
        groupcumret_pivotdata = groupret_pivotdata.add(1).cumprod() - 1 # 尝试用复利下的累积收益率
        return groupcumret_pivotdata.round(4) # 数值型数据都是保留到小数点后四位 

    def get_Performance(self, data_type):
        def get_stats(series, bm_series):
            """
            series是日收益率数据, pandas.series 
            data_type是组合类型, 'long'、'short'、'long_short'
            """
            ##复利口径
            return_ratio =  series.add(1).prod() - 1 # 总收益
            annual_return_ratio = (1 + return_ratio) ** (242/len(series)) - 1  #  年度收益

            ex_return_ratio =   ((1 + series) / (1 + bm_series)).prod() - 1 # 超额总收益
            ex_annual_return_ratio =  (1 + ex_return_ratio) ** (242 / len(series)) - 1  #  超额年度收益
            
            sharp_ratio = empyrical.sharpe_ratio(series, 0.035/242)
            return_volatility = empyrical.annual_volatility(series)
            max_drawdown  = empyrical.max_drawdown(series)
            information_ratio=series.mean()/series.std()
            win_percent = len(series[series>0]) / len(series)
            trading_days = len(series)

            series = series.fillna(0)  #  单利口径
            ret_3 = series.rolling(3).sum().iloc[-1]
            ret_10 = series.rolling(10).sum().iloc[-1]
            ret_21 = series.rolling(21).sum().iloc[-1]
            ret_63 = series.rolling(63).sum().iloc[-1]
            ret_126 = series.rolling(126).sum().iloc[-1]
            ret_252 = series.rolling(252).sum().iloc[-1]

            return {
                    'return_ratio': return_ratio,
                    'annual_return_ratio': annual_return_ratio,
                    'ex_return_ratio': ex_return_ratio,
                    'ex_annual_return_ratio': ex_annual_return_ratio,
                    'sharp_ratio': sharp_ratio,
                    'return_volatility': return_volatility,
                    'information_ratio':information_ratio,
                    'max_drawdown': max_drawdown,
                    'win_percent':win_percent,
                    'trading_days':trading_days,
                    'ret_3':ret_3,
                    'ret_10':ret_10,
                    'ret_21':ret_21,
                    'ret_63':ret_63,
                    'ret_126':ret_126,
                    'ret_252':ret_252
                    }

        if data_type == 'long':
            perf = get_stats(self.groupret_pivotdata[str(self.params['group_num']-1)], self.groupret_pivotdata['bm'])
        elif data_type =='short':
            perf = get_stats(self.groupret_pivotdata['0'], self.groupret_pivotdata['bm'])
        elif data_type =='long_short':
            perf = get_stats(self.groupret_pivotdata['ls'], self.groupret_pivotdata['bm'])
        return perf

    def get_IC_data(self, data_type):
            # IC
            def cal_ic(df):
                return df['daily_ret'].corr(df['factor'], method='spearman')

            if data_type == 'all':
                groupIC_data = self.group_data[['date','daily_ret','factor']].groupby('date', group_keys=False).apply(lambda x:cal_ic(x)).reset_index()
                groupIC_data.rename(columns={0:'g_ic'}, inplace=True)
                groupIC_data = groupIC_data.shift(1) # 首日为nan，最后一日有值
                groupIC_data['ic_cumsum'] = groupIC_data['g_ic'].cumsum()
                groupIC_data['ic_roll_ma'] = groupIC_data['g_ic'].rolling(22).mean()
                return groupIC_data.round(4).dropna() 

            elif data_type == 'long':
                data = self.group_data[self.group_data['group'] == str(self.params['group_num']-1)][['date','daily_ret','factor']]  
                groupIC_data = data.groupby('date', group_keys=False).apply(lambda x:cal_ic(x)).reset_index()
            elif data_type == 'short':
                data = self.group_data[self.group_data['group'] == '0'][['date','daily_ret','factor']]  
                groupIC_data = data.groupby('date', group_keys=False).apply(lambda x:cal_ic(x)).reset_index()
            elif data_type == 'long_short':
                data = self.group_data[self.group_data['group'].isin(['0',str(self.params['group_num']-1)])][['date','daily_ret','factor']]  
                groupIC_data = data.groupby('date', group_keys=False).apply(lambda x:cal_ic(x)).reset_index()
        
            IC_data = groupIC_data.rename(columns={0:'g_ic'}).dropna()
            
            ic_mean = np.nanmean(IC_data['g_ic'])
            ir = np.nanmean(IC_data['g_ic']) / np.nanstd(IC_data['g_ic'])
            ic_3 = IC_data['g_ic'].tail(3).mean()
            ic_10 = IC_data['g_ic'].tail(10).mean()
            ic_21 = IC_data['g_ic'].tail(21).mean()
            ic_63 = IC_data['g_ic'].tail(63).mean()
            ic_126 = IC_data['g_ic'].tail(126).mean()
            ic_252 = IC_data['g_ic'].tail(252).mean()

            return {
                    'ic':ic_mean,
                    'ir':ir,
                    'ic_3':ic_3,
                    'ic_10':ic_10,
                    'ic_21':ic_21,
                    'ic_63':ic_63,
                    'ic_126':ic_126,
                    'ic_252':ic_252
                    }

    def get_Turnover_data(self, data_type):

        def cal_turnover(df):
            # 求每天instrument和上一日的重复元素数量
            def count_repeat(s):
                if s.name > 0:
                    return len(set(s['instrument']).intersection(set(df.loc[s.name - 1, 'instrument'])))
                else:
                    return 0

            s = df.groupby('date').apply(lambda x:x.instrument.tolist())
            df = pd.DataFrame(s, columns = ['instrument']).reset_index()
            # 求每天instrument有多少元素
            df['instrument_count'] = df['instrument'].apply(len)

            df['repeat_count'] = df.apply(count_repeat, axis=1)
            df['turnover'] = 1 - df['repeat_count'] / df['instrument_count']
            return np.nanmean(df['turnover'])

        if data_type == 'long':
            df = self.group_data[self.group_data['group'] == str(self.params['group_num']-1)]
            return {'turnover':cal_turnover(df)}
        elif data_type == 'short':
            df = self.group_data[self.group_data['group'] == '0']
            return {'turnover':cal_turnover(df)}
        elif data_type == 'long_short':
            long_df = self.group_data[self.group_data['group'] == str(self.params['group_num']-1)]
            short_df = self.group_data[self.group_data['group'] == '0']
            return {'turnover':cal_turnover(long_df) + cal_turnover(short_df)}

    ## 总体绩效计算
    def get_whole_perf(self):
        summary_df = pd.DataFrame() 
        for _type in ['long', 'short', 'long_short']:   
            dict_merged = {} 
            dict1 = self.get_IC_data(_type)
            dict2 = self.get_Performance(_type)
            dict3 = self.get_Turnover_data(_type)

            dict_merged.update(dict1)
            dict_merged.update(dict2)
            dict_merged.update(dict3)
            df = pd.DataFrame.from_dict(dict_merged, orient='index', columns=['value']).T
            df['portfolio'] = _type 
            summary_df = summary_df.append(df)
        
        summary_df.index = range(len(summary_df))
        return summary_df.round(4)

    # 按年绩效计算
    def get_yearly_perf(self):
        # 计算年度绩效指标
        year_df = self.groupret_pivotdata.reset_index('date')
        year_df['year'] = year_df['date'].apply(lambda x:x.year)
        
        def cal_Performance(data):
            series = data[str(self.params['group_num']-1)] # 只看多头组合
            bm_series = data['bm']

            return_ratio =  series.add(1).prod() - 1 # 总收益
            annual_return_ratio = (1 + return_ratio) ** (242/len(series)) - 1  #  年度收益
            ex_return_ratio =  ((1 + series) / (1 + bm_series)).prod() - 1 # 总收益
            ex_annual_return_ratio = (1 + ex_return_ratio) ** (242 / len(series)) - 1  #  年度收益

            sharp_ratio = empyrical.sharpe_ratio(series,0.035/242)
            return_volatility = empyrical.annual_volatility(series)
            max_drawdown  = empyrical.max_drawdown(series)
            information_ratio=series.mean()/series.std()
            win_percent = len(series[series>0]) / len(series)
            trading_days = len(series)
            perf =  pd.DataFrame({
                    'return_ratio': [return_ratio],
                    'annual_return_ratio': [annual_return_ratio],
                    'ex_return_ratio': [ex_return_ratio],
                    'ex_annual_return_ratio': [ex_annual_return_ratio],
                    'sharp_ratio': [sharp_ratio],
                    'return_volatility': [return_volatility],
                    'max_drawdown': [max_drawdown],
                    'win_percent':[win_percent],
                    'trading_days':[int(trading_days)],
                    })
            return perf
        yearly_perf = year_df.groupby(['year'], group_keys=True).apply(cal_Performance) 
        yearly_perf = yearly_perf.droplevel(1).round(4)   # 去掉一个level

        # 计算年度IC
        data = self.group_data[self.group_data['group'] == str(self.params['group_num']-1)][['date','daily_ret','factor']] # 只看多头组合
        def cal_ic(df):
            return df['daily_ret'].corr(df['factor'])
        groupIC_data = data.groupby('date', group_keys=False).apply(lambda x:cal_ic(x)).reset_index()       
        IC_data = groupIC_data.rename(columns={0:'g_ic'}).dropna()
        IC_data['year'] = IC_data['date'].apply(lambda x:x.year)
        yearly_IC = IC_data.groupby('year').apply(lambda x:np.nanmean(x['g_ic']))

        yearly_perf['ic'] = yearly_IC.round(4)
        yearly_perf = yearly_perf.reset_index()
        yearly_perf['year'] = yearly_perf['year'].apply(str) 
        return yearly_perf

    def render(self):
        """图表展示因子分析结果"""
        from bigcharts import opts 

        fields = ['portfolio','ic', 'ir', 'turnover', 'return_ratio', 'annual_return_ratio','ex_return_ratio', 'ex_annual_return_ratio', 'sharp_ratio', 'return_volatility', 'information_ratio', 'max_drawdown', 'win_percent', 'ic_252', 'ret_252']
        whole_perf = self.whole_perf[fields] 
        c1 = bigcharts.Chart(
            data=whole_perf,
            type_="table",
            chart_options=dict(
                title_opts=opts.ComponentTitleOpts(title="整体绩效指标")
            ),
            y=list(whole_perf.columns))

        fields = ['year','ic', 'return_ratio', 'annual_return_ratio', 'ex_return_ratio', 'ex_annual_return_ratio', 'sharp_ratio', 'return_volatility',
        'max_drawdown', 'win_percent', 'trading_days']
        yearly_perf = self.yearly_perf[fields]
        c2 = bigcharts.Chart(
            data=yearly_perf,
            type_="table",
            chart_options=dict(
                title_opts=opts.ComponentTitleOpts(title="年度绩效指标(多头组合)")
            ),
            y=list(yearly_perf.columns))

        # 绘制累积收益图
        c3 = bigcharts.Chart(
            data=self.group_cumret,
            type_="line",
            x=self.group_cumret.index,
            y=self.group_cumret.columns)

        _IC = np.nanmean(alpha_instance.ic['g_ic'])
        _IR = np.nanmean(alpha_instance.ic['g_ic']) / np.nanstd(alpha_instance.ic['g_ic'])
        abs_IC = alpha_instance.ic['g_ic'].abs()
        significant_ic_ratio = abs_IC[abs_IC>=0.02].shape[0] / abs_IC.shape[0]
        c4 = bigcharts.Chart(
                    data=pd.DataFrame({'IC':[np.round(_IC,4)], '|IC|>0.02':[np.round(significant_ic_ratio,4)], 'IR':[np.round(_IR,4)]}),
                    type_="table",
                    chart_options=dict(
                        title_opts=opts.ComponentTitleOpts(title="IC分析指标")
                    ),
                    y=['IC','|IC|>0.02','IR'],
                )

        # 绘制每期IC时序图
        c5 = bigcharts.Chart(
            data=self.ic,
            type_="bar",
            x='date',
            y=['g_ic', 'ic_roll_ma'],
            chart_options=dict(
                title_opts=opts.TitleOpts(
                    title="IC曲线",
                    subtitle="每日IC、累计IC、近22日IC均值",
                    pos_left="center",
                    pos_top="top",
                ),
                legend_opts=opts.LegendOpts(
                    is_show=False,  # 不显示图例
                ),
                extend_yaxis=[opts.AxisOpts()]
                )
        )

        # 绘制IC累计曲线图 
        c6 = bigcharts.Chart(
            data=self.ic,
            type_="line",
            x='date',
            y=['ic_cumsum'],
            chart_options=dict(
                title_opts=opts.TitleOpts(
                    title="IC累积曲线",
                    pos_left="center",
                    pos_top="top",
                ),
                legend_opts=opts.LegendOpts(
                    is_show=False,  # 不显示图例
                )
                ),
            series_options={"ic_cumsum": {"yaxis_index": 1}}
        )
        c5_6 = bigcharts.Chart(data = [c5, c6], type_ = "overlap",)


        top_factor_df = self.factor_data[self.factor_data['date'] == self.end_date].round(4) # 最后一天因子数据
        top_factor_df['date'] = top_factor_df['date'].apply(lambda x:x.strftime('%Y-%m-%d'))
        # 按照 factor 列升序排序，获取最小的10行数据
        df_sorted_min = top_factor_df.sort_values('factor').head(self.top_n_ins)
        # 按照 factor 列降序排序，获取最大的10行数据
        df_sorted_max = top_factor_df.sort_values('factor', ascending=False).head(self.top_n_ins)

        c7 = bigcharts.Chart(
                    data=df_sorted_max,
                    type_="table",
                    chart_options=dict(
                        title_opts=opts.ComponentTitleOpts(title="因子值最大的%s只标的"%self.top_n_ins)
                    ),
                    y=['date','instrument','factor'],
                )
        c8 = bigcharts.Chart(
                    data=df_sorted_min[['date','instrument','factor']],
                    type_="table",
                    chart_options=dict(
                        title_opts=opts.ComponentTitleOpts(title="因子值最小的%s只标的"%self.top_n_ins)
                    ),
                    y=['date','instrument','factor'],
                )

        c_set = bigcharts.Chart([c1, c2, c3, c4, c5_6, c7, c8], type_="page").render(display=False)
        from IPython.display import display
        display(c_set)
        return c_set.data
alpha_instance = AlphaMiner(params=params, factor_data=factor_data)
print(alpha_instance.price_data)
report_html = alpha_instance.render()