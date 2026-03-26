# 克隆自聚宽文章：https://www.joinquant.com/post/19809
# 标题：价值选股与RSRS择时 py3版本
# 作者：nbnc003


# 克隆自聚宽文章：https://www.joinquant.com/post/15002
# 标题：兄台且慢，去天台排队不如看看这个策略先
# 作者：K线放荡不羁

# 导入函数库
import statsmodels.api as sm
#from pandas.stats.api import ols

# 初始化函数，设定基准等等
def initialize(context):
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')
    set_parameter(context)
    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    
    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG') 
      # 开盘时运行
    run_daily(market_open, time='9:30', reference_security='000300.XSHG')
      # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')
    
'''
==============================参数设置部分================================
'''
def set_parameter(context):
    # 设置RSRS指标中N, M的值
    #统计周期
    g.N = 18
    #统计样本长度
    g.M = 1100
    #首次运行判断
    g.init = False
    #持仓股票数
    g.stock_num = 10
    #风险参考基准
    g.security = '000300.XSHG'
    # 设定策略运行基准
    set_benchmark(g.security)
    #记录策略运行天数
    g.days = 0
    # 买入阈值
    g.buy = 0.7
    g.sell = -0.7
    #用于记录回归后的beta值，即斜率
    g.ans = []
    #用于计算被决定系数加权修正后的贝塔值
    g.ans_rightdev= []
    g.blacklist = []   
    # 计算2005年1月5日至回测开始日期的RSRS斜率指标
    # 获得回测前一天日期，千万避免未来数据
    previous_date = context.current_dt - datetime.timedelta(days=1)
    prices = get_price(g.security, '2005-01-05', previous_date, '1d', ['high', 'low'])
    prices[np.isnan(prices)] = 0
    prices[np.isinf(prices)] = 0
    highs = prices.high
    lows = prices.low
    g.ans = []
    for i in range(len(highs))[g.N:]:
        data_high = highs.iloc[i-g.N+1:i+1]
        data_low = lows.iloc[i-g.N+1:i+1]
        X = sm.add_constant(data_low)
        model = sm.OLS(data_high,X)
        results = model.fit()
        g.ans.append(results.params[1])
        #计算r2
        g.ans_rightdev.append(results.rsquared)
    
## 开盘前运行函数     
def before_market_open(context):
    g.days += 1
    g.statsDate = context.current_dt.date()
    # 给微信发送消息（添加模拟交易，并绑定微信生效）
    send_message('策略正常，运行第%s天~'%g.days)

## 开盘时运行函数
def market_open(context):
#def handle_data(context, data):

    times = context.current_dt.month
    security = g.security
    # 填入各个日期的RSRS斜率值
    beta=0
    r2=0
    if g.init:
        g.init = False
    else:
        #RSRS斜率指标定义
        prices = attribute_history(security, g.N, '1d', ['high', 'low'])#获取过去g.n（18）天的最高价、最低价
        prices[np.isnan(prices)] = 0
        prices[np.isinf(prices)] = 0
        highs = prices.high#最高价
        lows = prices.low#最低价
        X = sm.add_constant(lows)#线性回归X
        model = sm.OLS(highs, X)#线性回归结果
        beta = model.fit().params[1]
        g.ans.append(beta)#回归后的beta值，即斜率
        #计算r2
        r2=model.fit().rsquared    
        g.ans_rightdev.append(r2)#用于计算被决定系数加权修正后的贝塔值
    # 计算标准化的RSRS指标
    section = g.ans[-g.M:]
    # 计算均值序列
    mu = np.mean(section)
    # 计算标准化RSRS指标序列
    sigma = np.std(section)
    zscore = (section[-1]-mu)/sigma  
    #计算右偏RSRS标准分
    zscore_rightdev= zscore*beta*r2
    # 如果上一时间点的RSRS斜率大于买入阈值, 则全仓买入
    if zscore_rightdev > g.buy:# and corrcoef(array([trade_vol10['volume'], g.RSRS_stdratio_rev_list[:context.previous_date].tail(10)]))[0,1] > 0and MA20>MA20_2:
        # 记录这次买入
        log.info("市场风险在合理范围")
        trade_func(context)
    # 如果上一时间点的RSRS斜率小于卖出阈值, 则空仓卖出
    elif (zscore_rightdev < g.sell) and (len(context.portfolio.positions.keys()) > 0):
        # 记录这次卖出
        log.info("市场风险过大，保持空仓状态")
        # 卖出所有股票,使这只股票的最终持有量为0
        for s in context.portfolio.positions.keys():
            order_target(s, 0)

#策略选股买卖部分    
def trade_func(context):
    #获取股票池
    df = get_fundamentals(query(valuation.code,valuation.market_cap,valuation.pb_ratio,indicator.roe))
    #进行pb,roe大于0筛选
    #进行pb,roe大于0筛选
    df = df[(df['roe']>0) & (df['pb_ratio']>0)].sort_values('pb_ratio')
    #以股票名词作为index
    df.index = df['code'].values
    #取roe倒数
    df['1/roe'] = 1/df['roe']
    #获取综合得分
    df['point'] = df[['pb_ratio','1/roe']].rank().T.apply(f_sum)
    df = df.sort_values('point')[:g.stock_num*10]
    stock_list = df.index
    positions_list = context.portfolio.positions.keys()
    stock_list = unpaused(context,stock_list,positions_list)
    stock_list = remove_st(context,stock_list, g.statsDate)
    ## 过滤正在涨停的股票,不过滤持仓股
    stock_list = remove_limit_up(context, stock_list, positions_list) 
    pool = []
    for stock in stock_list:
        if stock not in g.blacklist:
            pool.append(stock)    
    pool = pool[:g.stock_num]
    log.info('总共选出%s只股票'%len(pool))
    if len(pool)>0:
        #得到每只股票应该分配的资金
        cash = context.portfolio.total_value/len(pool)
    # 将不在股票池中的股票卖出
    sell_list = set(positions_list) - set(pool)
    for stock in sell_list:
        order_target_value(stock, 0)    
    
    #买入股票
    for s in pool:
        order_target_value(s,cash)
        
#打分工具
def f_sum(x):
    return sum(x)

## 收盘后运行函数  
def after_market_close(context):
    #得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    #打印账户总资产
    log.info('今日账户总资产：%s'%round(context.portfolio.total_value,2))
    #log.info('##############################################################')
    # 计算当前仓位
    record(P=(100-math.ceil(context.portfolio.available_cash / context.portfolio.total_value * 100)))
# 剔除上市时间较短的产品
def fun_delNewShare(context, equity, deltaday):
    deltaDate = context.current_dt.date() - dt.timedelta(deltaday)

    tmpList = []
    for stock in equity:
        if get_security_info(stock).start_date < deltaDate:
            tmpList.append(stock)

    return tmpList

def unpaused(context,stock_list, positions_list):
    current_data = get_current_data()
    tmpList = []
    for stock in stock_list:
        if not current_data[stock].paused or stock in positions_list:
            tmpList.append(stock)
    return tmpList

def remove_st(context, stock_list, statsDate):
    current_data = get_current_data()
    return [s for s in stock_list if not current_data[s].is_st]

# 剔除涨停板的股票（如果没有持有的话）
def remove_limit_up(context, stock_list, positions_list):
    h = history(1, '1m', 'close', stock_list, df=False, skip_paused=False, fq='pre')
    h2 = history(1, '1m', 'high_limit', stock_list, df=False, skip_paused=False, fq='pre')
    tmpList = []
    for stock in stock_list:
        if h[stock][0] < h2[stock][0] or stock in positions_list:
#                if buyOrSellCheck(context, stock,3,1.4):
                tmpList.append(stock)
    return tmpList