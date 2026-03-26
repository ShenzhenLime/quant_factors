#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tushare as ts
df = ts.get_tick_data('000001',date='2019-12-12',src='tt')
df[(df.time<'09:30:30') & (df.time>='09:30:00')].head(10)


# In[7]:


import pandas as pd
from jqdata import * 

def funtrade(p_t,current):
    if p_t<current:
        return '买'
    elif p_t>current:
        return '卖'
    else:
        return ' '

cols = ['time', 'current', 'volume', 'money']
stk = '000001.XSHE'
curdate = '2019-12-12'
stadt = datetime.datetime.strptime(curdate + ' 09:25:00',"%Y-%m-%d %H:%M:%S")
enddt = datetime.datetime.strptime(curdate + ' 09:31:00',"%Y-%m-%d %H:%M:%S")
d = get_ticks(stk, start_dt=stadt, end_dt=enddt, fields=cols) 
hist = pd.DataFrame(d,columns=cols)
hist['volt'] = (hist['volume']-hist['volume'].shift(1))/100
hist['mont'] = hist['money'] - hist['money'].shift(1)
hist['p_t'] = hist['mont']/hist['volt']/100
hist['dir'] = hist.apply(lambda x: funtrade(x.p_t, x.current), axis = 1)
hist.head(10)


# In[ ]:




