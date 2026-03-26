# -*- coding: utf-8 -*-
"""
因子计算
"""
import pandas as pd
import numpy as np
from quant_infra import db_utils, get_data, factor_calc

def compute_week_effect():
    """
    计算周内效应因子：
        Factor_{i, week_t} = epsilon_{i, first_day_t} - mean(epsilon_{i, other_days_t})
    first_day 为该周内第一个实际交易日（兼容节假日无周一的情况），
    每周取最后一个交易日作为因子的 trade_date 写入 week_factor 表。
    """
    # ========== 检查 week_factor 表中已有哪些数据，返回需要计算的日期 list ==========
    dates_to_download = get_data.get_dates_todo('week_factor')
    if not dates_to_download:
        print("周内效应因子已是最新")
        return

    # 需要向前扩展到第一个需要计算周的周一，确保跨界周数据完整
    first_date = pd.to_datetime(str(dates_to_download[0]), format='%Y%m%d')
    # 如果 first_date 不是周一，则回退到本周周一
    week_start = first_date - pd.tseries.offsets.Week(n=0, weekday=0)
    week_start_str = week_start.strftime('%Y%m%d')
    end_date = dates_to_download[-1]

    df = db_utils.read_sql(
        f"SELECT ts_code, trade_date, resid FROM stock_resids "
        f"WHERE resid IS NOT NULL AND trade_date >= '{week_start_str}' AND trade_date <= '{end_date}'"
    )

    df['date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    df['week'] = df['date'].dt.to_period('W')       # 以 Sun 结尾的自然周
    # 每只股票在该周内第一个实际交易日，兼容节假日无周一的情况
    df['is_first_day'] = df.groupby(['ts_code', 'week'])['date'].transform('min') == df['date']

    # ---- 分别聚合每周第一个交易日 / 其余交易日的残差 ----
    mon_resid = (
        df[df['is_first_day']]
        .groupby(['ts_code', 'week'])['resid']
        .mean()
        .reset_index(name='mon_resid')
    )

    other_resid = (
        df[~df['is_first_day']]
        .groupby(['ts_code', 'week'])['resid']
        .mean()
        .reset_index(name='other_resid')
    )

    # inner join：只保留同一周内既有第一个交易日也有其余交易日数据的股票-周组合
    weekly = mon_resid.merge(other_resid, on=['ts_code', 'week'], how='inner')
    weekly['factor'] = weekly['mon_resid'] - weekly['other_resid']

    # 取每周最后一个交易日作为该周的 trade_date（与收益率对齐）
    week_lastday = (
        df.groupby(['ts_code', 'week'])['trade_date']
        .max()
        .reset_index(name='trade_date')
    )

    result = weekly.merge(week_lastday, on=['ts_code', 'week'])

    # 只保留 trade_date 落在需要计算的日期范围内的记录，避免重复写入旧数据
    result = result[
        (result['trade_date'].astype(str) >= dates_to_download[0]) &
        (result['trade_date'].astype(str) <= dates_to_download[-1])
    ]
    result = (result[['ts_code', 'trade_date', 'factor']]
              .sort_values(['trade_date', 'ts_code'])
              .reset_index(drop=True))
    result['factor'] = result.groupby('trade_date')['factor'].transform(factor_calc.winsorize)
    if result.empty:
        print("未生成有效因子，请检查残差数据是否覆盖完整的交易周")
        return

    db_utils.write_to_db(result, 'week_factor', save_mode='append')


