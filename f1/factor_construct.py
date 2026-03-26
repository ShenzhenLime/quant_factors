# -*- coding: utf-8 -*-
"""
f1 因子构建：小市值 + 基本面质量（ROE/ROA）筛选
因子逻辑：
  1. 过滤非主板、ST、次新股、停牌、高价股
  2. 取截止该日期已公告的最新财报：ROE > 15% 且 ROA > 10%
  3. 因子值 = -total_mv（越小市值，因子值越高）
  4. 月末最后一个交易日作为 trade_date 写入 f1 表
"""
import pandas as pd
import numpy as np
from quant_infra import db_utils, get_data


def filter_stocks(date: str, 
                  filter_gem=True,
                  filter_star=True,
                  filter_bj=True,
                  filter_st=True,
                  filter_new=True,
                  filter_suspended=True,
                  filter_high_price=True,
                  price_threshold=10.0,
                  new_stock_days=250) -> list:
    """
    股票池过滤函数，返回过滤后的 ts_code 列表。

    Args:
        date(str): 交易日，格式 'YYYYMMDD'
        filter_gem(bool): 是否过滤创业板（30开头），默认True
        filter_star(bool): 是否过滤科创板（68开头），默认True
        filter_bj(bool): 是否过滤北交所（4/8开头），默认True
        filter_st(bool): 是否过滤ST/退市股，默认True
        filter_new(bool): 是否过滤次新股（上市不足 new_stock_days 天），默认True
        filter_suspended(bool): 是否过滤停牌股，默认True
        filter_high_price(bool): 是否过滤高价股（收盘价 > price_threshold），默认True
        price_threshold(float): 高价股阈值（元），默认10.0
        new_stock_days(int): 次新股判断天数，默认250天

    Returns:
        list: 过滤后的 ts_code 列表
    """
    basic = db_utils.read_sql("SELECT ts_code, name, list_date FROM stock_basic")

    # 过滤创业板
    if filter_gem:
        basic = basic[~basic['ts_code'].str.startswith('300')]
    # 过滤科创板
    if filter_star:
        basic = basic[~basic['ts_code'].str.startswith('68')]
    # 过滤北交所
    if filter_bj:
        basic = basic[~basic['ts_code'].str.startswith(('4', '8'))]
    # 过滤ST
    if filter_st:
        basic = basic[
            ~basic['name'].str.contains('ST|\\*|退', na=False)
        ]
    # 过滤次新股
    if filter_new:
        today = pd.to_datetime(date, format='%Y%m%d')
        basic['list_date'] = pd.to_datetime(basic['list_date'].astype(str), format='%Y%m%d', errors='coerce')
        basic = basic[(today - basic['list_date']).dt.days >= new_stock_days]

    stock_list = basic['ts_code'].tolist()

    # 过滤停牌 / 高价股（需要当日行情数据）
    if filter_suspended or filter_high_price:
        try:
            daily = db_utils.read_sql(
                f"SELECT ts_code, close FROM stock_bar WHERE trade_date = {date}"
            )
            if filter_suspended:
                # stock_bar 中无当日记录即视为停牌
                stock_list = [s for s in stock_list if s in daily['ts_code'].values]
            if filter_high_price:
                high_price_set = set(daily[daily['close'] > price_threshold]['ts_code'].tolist())
                stock_list = [s for s in stock_list if s not in high_price_set]
        except Exception as e:
            print(f"过滤停牌/高价股时读取行情失败（{date}）: {e}")

    return stock_list


def compute_f1_factor():
    """
    计算月度小市值质量因子，增量写入 f1 表。
    每月最后一个交易日为 trade_date，ROE>15% 且 ROA>10% 的主板小盘股，
    因子值 = -(rank_mv + rank_pb) / 2（小市值+低PB双排名均值越小越优）。
    """
    dates_to_download = get_data.get_dates_todo('f1')
    if not dates_to_download:
        print("f1 因子已是最新")
        return

    # 从待下载日期中取每月最后一个交易日
    dates_df = pd.DataFrame({'trade_date': [str(d) for d in dates_to_download]})
    dates_df['month'] = dates_df['trade_date'].str[:6]
    month_end_dates = dates_df.groupby('month')['trade_date'].max().tolist()
    print(f"需计算 {len(month_end_dates)} 个月末：{month_end_dates[0]} ~ {month_end_dates[-1]}")

    # 一次性读取财务数据（公告日 + 报告期 + ROE + ROA），只保留年报（end_date 以 1231 结尾）
    fina = db_utils.read_sql("SELECT ts_code, ann_date, end_date, roe, roa FROM fina_indicator")
    fina = fina.dropna(subset=['roe', 'roa', 'ann_date', 'end_date'])
    fina['ann_date'] = fina['ann_date'].astype(str)
    fina['end_date'] = fina['end_date'].astype(str)
    fina = fina[fina['end_date'].str.endswith('1231')]  # 只保留年报

    # 一次性读取市值和 PB 数据
    mv_df = db_utils.read_sql("SELECT ts_code, trade_date, total_mv, pb FROM daily_basic")
    mv_df['trade_date'] = mv_df['trade_date'].astype(str)

    results = []
    for date in month_end_dates:
        # 1. 股票池过滤（非主板/ST/次新/停牌/高价）
        stock_list = filter_stocks(date)
        if not stock_list:
            continue

        # 2. 取截止 date 已公告的最新财报，避免未来数据
        fina_at_date = fina[fina['ann_date'] <= date]
        latest_fina = (
            fina_at_date
            .sort_values('ann_date')
            .groupby('ts_code', as_index=False)
            .last()
        )

        # 3. 基本面筛选：ROE > 15 且 ROA > 10（已是百分比值）
        qualified = set(
            latest_fina[(latest_fina['roe'] > 15) & (latest_fina['roa'] > 10)]['ts_code']
        )

        # 4. 与股票池取交集
        final_stocks = [s for s in stock_list if s in qualified]
        if not final_stocks:
            continue

        # 5. 取当日市值和 PB，因子值 = -(rank_mv + rank_pb) / 2（小市值且低 PB 越优）
        mv_at_date = mv_df[
            (mv_df['trade_date'] == date) & (mv_df['ts_code'].isin(final_stocks))
        ][['ts_code', 'total_mv', 'pb']].copy()
        mv_at_date = mv_at_date.dropna(subset=['total_mv', 'pb'])

        if mv_at_date.empty:
            continue

        # 两个指标均按升序排名（小市值、低 PB 均得到较小排名值），取负值使小市值且低 PB 得到高因子值
        mv_at_date['rank_mv'] = mv_at_date['total_mv'].rank(ascending=True)
        mv_at_date['rank_pb'] = mv_at_date['pb'].rank(ascending=True)
        mv_at_date['factor'] = -((mv_at_date['rank_mv'] + mv_at_date['rank_pb']) / 2)
        mv_at_date['trade_date'] = date
        results.append(mv_at_date[['ts_code', 'trade_date', 'factor']])

    if results:
        factor_df = pd.concat(results, ignore_index=True)
        db_utils.write_to_db(factor_df, 'f1', save_mode='append')
        print(f"f1 因子计算完成，共 {len(factor_df)} 行，覆盖 {factor_df['trade_date'].nunique()} 个月末")
    else:
        print("f1 因子：无新数据可计算")
