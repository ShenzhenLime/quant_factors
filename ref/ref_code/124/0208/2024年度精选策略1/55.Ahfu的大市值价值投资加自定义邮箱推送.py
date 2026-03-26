# 克隆自聚宽文章：https://www.joinquant.com/post/43159
# 标题：Ahfu的大市值价值投资加自定义邮箱推送
# 作者：不如定投纳指

# 克隆自聚宽文章：https://www.joinquant.com/post/41921
# 标题：大市值价值投资，从2005年至今超额稳定
# 作者：Ahfu

# 导入函数库
from jqdata import *
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import base64

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
	# 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
	#防止未来函数
    set_option("avoid_future_data", True)
	# 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
	# 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')
	
	# 股票超参数
    g.buy_stock_count = 5
    g.check_out_lists = []
    g.high_limit_list = []
    g.hold_list = []
    # 邮箱相关参数 ###
    g.open_gmail = False #True/False 打开/关闭邮件功能
    g.gmail_host="smtp.163.com"  #你的邮箱服务器地址
    g.gmail_port= 25  #smtp服务端口默认25
    g.gmail_sender = 'xxxx@163.com'  #你的邮箱
    g.gmail_authcode = 'xxxx'  #授权码为开启第三方邮件服务后从弹窗提示中获取
    g.gmail_receivers = ['xxxx@163.com','xxxx@qq.com']   #要发送的邮箱列表
	### 股票相关设定 ###
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.00012, close_commission=0.00012, min_commission=5),
				   type='stock')
	
	# 开盘前运行
    run_daily(prepare_stock_list, time='9:05', reference_security='000300.XSHG')
    run_monthly(before_market_open, 1, time='9:05', reference_security='000300.XSHG')
	# 盘中运行
    run_monthly(my_trade, 1, time='9:30', reference_security='000300.XSHG')
    run_daily(check_limit_up, time='14:30', reference_security='000300.XSHG')
    # 每周邮件总结
    run_monthly(send_to_mail, -1, time='15:10', reference_security='000300.XSHG')
	# 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')
	

def prepare_stock_list(context):
    #获取已持有列表
    g.hold_list= []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)
    #获取昨日涨停列表
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close','high_limit'], count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.high_limit_list = list(df.code)
    else:
        g.high_limit_list = [] 

## 开盘时运行函数
def my_trade(context):
	# 买卖
	adjust_position(context, g.check_out_lists)

## 收盘后运行函数
def after_market_close(context):
	positions_dict = context.portfolio.positions
	for position in list(positions_dict.values()):
	    log.info("当前持仓:{0}, 数量:{1}, 市值:{2}, 盈利：{3}%, 建仓时间:{4}".format(get_name(position.security), position.total_amount, round(position.value,0), round((position.value-(position.avg_cost*position.total_amount))/(position.avg_cost*position.total_amount)*100,1), position.init_time))
	log.info('#########################################################################################\n\n')


# 自定义下单
# 根据Joinquant文档，当前报单函数都是阻塞执行，报单函数（如order_target_value）返回即表示报单完成
# 报单成功返回报单（不代表一定会成交），否则返回None
def order_target_value_(security, value):
	if value == 0:
		log.debug("卖出 %s" % (get_name(security)))
	else:
		log.debug("买入 %s ，市值： %f" % (get_name(security), value))

	# 如果股票停牌，创建报单会失败，order_target_value 返回None
	# 如果股票涨跌停，创建报单会成功，order_target_value 返回Order，但是报单会取消
	# 部成部撤的报单，聚宽状态是已撤，此时成交量>0，可通过成交量判断是否有成交
	return order_target_value(security, value)


# 开仓，买入指定价值的证券
# 报单成功并成交（包括全部成交或部分成交，此时成交量大于0），返回True
# 报单失败或者报单成功但被取消（此时成交量等于0），返回False
def open_position(security, value):
	order = order_target_value_(security, value)
	if order != None and order.filled > 0:
		return True
	return False


# 平仓，卖出指定持仓
# 平仓成功并全部成交，返回True
# 报单失败或者报单成功但被取消（此时成交量等于0），或者报单非全部成交，返回False
def close_position(position):
	security = position.security
	order = order_target_value_(security, 0)  # 可能会因停牌失败
	if order != None:
		if order.status == OrderStatus.held and order.filled == order.amount:
			return True
	
	return False


# 交易
def adjust_position(context, buy_stocks):
    sellbuystocklist=[]
    advlist_s=[]
    for stock in g.hold_list:
        if (stock not in buy_stocks) and (stock not in g.high_limit_list):
            log.info("卖出[%s]" % (stock))
            position = context.portfolio.positions[stock]
            sellbuystocklist.append(stock)
            advlist_s.append("卖出"+str(position.total_amount)+"股")
            close_position(position)
        else:
            log.info("已持有[%s]" % (stock))

	# 根据股票数量分仓
	# 此处只根据可用金额平均分配购买，不能保证每个仓位平均分配
    position_count = len(context.portfolio.positions)
    if g.buy_stock_count > position_count:
        value = context.portfolio.cash / (g.buy_stock_count - position_count)
        for stock in buy_stocks:
            if stock not in context.portfolio.positions:
                if open_position(stock, value):
                    sellbuystocklist.append(stock)
                    advlist_s.append("买入"+str(context.portfolio.positions[stock].total_amount)+"股")
                    if len(context.portfolio.positions) == g.buy_stock_count:
                        break
    send_mail(context,sellbuystocklist,advlist_s,'大市值价值投资-调仓通知','调仓通知')

# # 通过代码返回股票名称
def get_name(stk):
    return get_security_info(stk).display_name+':'+stk[:6]
    
def check_limit_up(context):
    now_time = context.current_dt
    nothighlist=[]
    advlist=[]
    if g.high_limit_list != []:
        #对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
        for stock in g.high_limit_list:
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close','high_limit'], skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            if current_data.iloc[0,0] < current_data.iloc[0,1]:
                log.info("[%s]涨停打开，卖出" % (stock))
                position = context.portfolio.positions[stock]
                nothighlist.append(stock)
                advlist.append('卖出'+str(position.total_amount)+"股")
                close_position(position)
            else:
                log.info("[%s]涨停，继续持有" % (stock))
        send_mail(context,nothighlist,advlist,'大市值价值投资-调仓通知','涨停打开，卖出')


## 开盘前运行函数
def before_market_open(context):
    g.check_out_lists = []
    current_data = get_current_data()
    check_date = context.previous_date - datetime.timedelta(days=200)
    all_stocks = list(get_all_securities(date=check_date).index)
    # 过滤创业板、ST、停牌、当日涨停
    all_stocks = [stock for stock in all_stocks if not (
            (current_data[stock].day_open == current_data[stock].high_limit) or  # 涨停开盘
            (current_data[stock].day_open == current_data[stock].low_limit) or  # 跌停开盘
            current_data[stock].paused or  # 停牌
            current_data[stock].is_st or  # ST
            ('ST' in current_data[stock].name) or
            ('*' in current_data[stock].name) or
            ('退' in current_data[stock].name) or
            (stock.startswith('30')) or  # 创业
            (stock.startswith('68')) or  # 科创
            (stock.startswith('8')) or  # 北交
            (stock.startswith('4'))   # 北交
    )]
    q = query(
        valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
        ).filter(
        valuation.pb_ratio < 1,
        cash_flow.subtotal_operate_cash_inflow > 1e6,
        indicator.adjusted_profit > 1e6,
        indicator.roa > 0.15,
        indicator.inc_net_profit_year_on_year > 0,
    	valuation.code.in_(all_stocks)
    	).order_by(
    	indicator.roa.desc()
    ).limit(
    	g.buy_stock_count * 3
    )
    
    check_out_lists = list(get_fundamentals(q).code)
    # 取需要的只数
    check_out_lists = check_out_lists[:g.buy_stock_count]
    
    g.check_out_lists = check_out_lists
    log.info("今日股票池：%s" % g.check_out_lists)
    advlist_t = []
    for sss in check_out_lists:
        advlist_t.append("可买")
    send_mail(context,check_out_lists,advlist_t,'大市值价值投资-本月筛选','本月筛选股票')
    
def send_mail(context,stocklist,advlist,title,info,sumup=""):
    if g.open_gmail==False:
        return False
    # 邮箱配置 ######################
    host=g.gmail_host  #你的邮箱服务器地址
    port=g.gmail_port  #smtp服务端口默认25
    sender =g.gmail_sender  #你的邮箱
    authcode =g.gmail_authcode  #授权码为开启第三方邮件服务后从邮箱服务商获取
    #要发送的邮箱列表
    receivers =g.gmail_receivers
    # 邮箱配置 ######################
    
    stockstr = ''
    cont = 1
    for stock in stocklist:
        stockcode = stock.split(".")[0]
        price_now = attribute_history(stock, 1, '1m', 'close')['close'][0]
        stockstr = stockstr+'<p>'+str(cont)+'. ' +get_security_info(stock).display_name+': '+stock+\
                '  最新价：'+str(price_now)+'</p>'+\
                '<p>'+advlist[cont-1]+'  '+\
                '<a href="https://gushitong.baidu.com/stock/ab-'+stockcode+'">查看📖'+\
                '</a></p>'
        
        cont = cont+1
    

    mail_msg = '<p><strong>'+info+'</strong></p>'\
    +'<p>'+str(context.current_dt)+'</p>'\
    +'<p>股票列表: </p>'\
    +stockstr+sumup
     
    
    try:
        #登录
        smtp = smtplib.SMTP(host, port)  
        smtp.login(sender, authcode)  

        for tomail in receivers:
            message = MIMEText(mail_msg, 'html', 'utf-8')
            fromnick = sender[:sender.find('@')]
            fromnicknamebase64 = base64.b64encode(bytes(fromnick, 'utf-8'))
            fromnickname64str = str(fromnicknamebase64, 'utf-8')
            message['From'] = Header('"=?utf-8?B?'+fromnickname64str+'=?=" <'+sender+'>')
            message['Subject'] = Header(title, 'utf-8')

            nickname = tomail[:tomail.find('@')]
            nicknamebase64 = base64.b64encode(bytes(nickname, 'utf-8'))
            nickname64str = str(nicknamebase64, 'utf-8')
            message['To'] =  Header('"=?utf-8?B?'+nickname64str+'=?=" <'+tomail+'>')

            smtp.sendmail(sender,[tomail], message.as_string())
        print ("邮件发送成功")
    except smtplib.SMTPException:
        print ("Error: 无法发送邮件")
    finally:
        # 退出服务器
        smtp.quit()

# 每星期发送报告邮件
def send_to_mail(context):
    stocklist=[]
    advlist=[]
    for position in list(context.portfolio.positions.values()):
        securities=position.security
        cost=position.avg_cost
        price=position.price
        ret=100*(price/cost-1)
        value=position.value
        amount=position.total_amount 
        
        stocklist.append(securities)
        advlist.append( '成本价:{}'.format(format(cost,'.2f'))+',现价:{}'.format(price)+',收益率:{}%'.format(format(ret,'.2f'))+',持仓(股):{}'.format(amount)+',市值:{}'.format(format(value,'.2f')) )
    sumup = '<p>'+'总资产:{}'.format(format(context.portfolio.total_value,'.2f'))+',可用资金:{}'.format(format(context.portfolio.available_cash,'.2f'))+\
        ',累计收益:{}'.format(format((context.portfolio.total_value/context.portfolio.inout_cash-1)*100,'.2f'))+'%</p>'
    send_mail(context,stocklist,advlist,'大市值价值投资-每周总结报告','每周总结',sumup)