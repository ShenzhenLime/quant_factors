# 策略总结：5年15倍小市值+基本面月度调仓

> 原策略来源：[聚宽文章 #45510](https://www.joinquant.com/post/45510)，作者 langcheng999

---

## 核心逻辑

这是一个**月度调仓的小市值+基本面筛选**策略，月末选股、月末换仓，持股约10只。

---

## 选股流程（每月执行）

```
全A股
  → 过滤科创板(68开头)/北交所(4/8开头)
  → 过滤ST、停牌、次新股(上市<250天)
  → 过滤涨停/跌停股
  → 过滤高价股(收盘价 > 10元)
  → 基本面过滤：ROE > 15% 且 ROA > 10%
  → 按市值从小到大排序
  → 过滤"最近30天内曾涨停过，且在最近30天内持有过"的股票(黑名单)
  → 取前10只
```

---

## 调仓规则

- **卖出**：不在新选股池里的股票，月末 14:55 卖出
- **买入**：按可用资金平均分配买入新选股池
- **特殊处理**：昨日涨停且今日 14:00 涨停打开 → 立刻卖出

---

## 黑名单机制（防追高）

过去 30 个交易日内**持有过**的股票，如果**在此期间曾经涨停**，则列入黑名单，下月不再买入。目的是避免追涨买回已拉升过的票。

---

## 用 Tushare 需要哪些数据

| 需求 | Tushare 接口 |
|---|---|
| 全A股股票列表（含上市日期、是否ST）| `pro.stock_basic()` |
| 日线行情（收盘价、涨跌停价、是否停牌）| `pro.daily()` + `pro.daily_basic()` |
| 财务指标（ROE、ROA）| `pro.fina_indicator()` |
| 市值数据 | `pro.daily_basic()`（`total_mv` 字段）|
| 交易日历 | `pro.trade_cal()` |

---

## 各步骤计算方式

### 1. 过滤列表

## 打包成函数，在回测中选择
```python
df = pro.stock_basic()
# 过滤创业板: ts_code.startswith('30')
# 过滤科创板: ts_code.startswith('68')
# 过滤北交所: ts_code.startswith('4') or ts_code.startswith('8')
# 过滤ST: 'ST' in name or '*' in name or '退' in name
# 过滤次新: (today - list_date) < 250天
```

### 2. 停牌 / 涨跌停过滤

```python
daily = pro.daily(trade_date=date, fields='ts_code,close,up_limit,down_limit')
daily_b = pro.daily_basic(trade_date=date, fields='ts_code,trade_status')
# 停牌: trade_status == '停牌'
# 涨停: close >= up_limit * 0.97
# 跌停: close <= down_limit * 1.04
```

### 3. 高价股过滤

```python
# close > 10 则排除（用当日收盘价）
```

### 4. 基本面筛选（ROE / ROA）

akshare 净资产收益率(%) roe 总资产净利润率(%) roa
```python
fin = pro.fina_indicator(period='20231231', fields='ts_code,roe,roa')
# roe > 15 且 roa > 10（tushare 已是百分比值）
# 注意：取最新已公布的季报/年报，避免未来数据
```

### 5. 小市值排序

```python
basic = pro.daily_basic(trade_date=date, fields='ts_code,total_mv')
# total_mv 单位是万元，升序排列
```

### 6. 涨停历史（黑名单）

```python
# 取过去30个交易日的日线数据
hist = pro.daily(ts_code=stock, start_date=..., end_date=..., fields='trade_date,close,up_limit')
# 判断是否存在 close == up_limit 的行
```

---

## 关键注意点

- `fina_indicator` 是**按报告期**返回的，需要取最新已公布的季报/年报，**严禁使用未来数据**
- `up_limit` / `down_limit` 在 `pro.daily()` 里有，部分历史数据可能缺失，可用 `close * 1.1` 和 `close * 0.9` 估算（涨跌停 10%）
- 批量拉财务数据建议按 `period` 而非按股票循环，性能差异大
- 财务指标更新频率低（季度），配合 `get_financial()` 中的 3 个月缓存机制使用即可
