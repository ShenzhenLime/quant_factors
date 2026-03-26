# 克隆自聚宽文章：https://www.joinquant.com/post/20497
# 标题：指数估值自动报表系统——源代码
# 作者：东南有大树

from jqdata import *
import datetime
# 导入批数估值模块
from index_valuation import *

def initialize(context):
    run_daily(option, time='7:00', reference_security='000300.XSHG')

def option(context):
    # 加时间限制，以免重复发送
    if context.current_dt.year != datetime.datetime.now().year\
    or context.current_dt.month != datetime.datetime.now().month\
    or context.current_dt.day != datetime.datetime.now().day:
        return
    else:
        send_message()
    
    
