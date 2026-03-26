# -*- coding: utf-8 -*-
"""
主程序 - f1 策略（小市值 + 基本面质量因子）
流程：更新数据 → 计算因子 → 因子评测
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from factor_construct import compute_f1_factor
from quant_infra.get_data import get_stock_data_by_date, get_daily_basic, get_financial, get_basic
from quant_infra.factor_analyze import evaluate_factor, group_plot
from quant_infra import db_utils
from quant_infra.trade import simulate_trade


def main():
    """主流程 - 一键更新数据并评测因子"""

    # # 1. 更新股票基本信息（含上市日期，过滤次新股需用到）
    # print("\n更新股票基本信息...")
    # get_basic()

    # 2. 增量更新日线行情（用于过滤停牌/高价股，以及回测收益计算）
    print("\n获取股票日线数据...")
    get_stock_data_by_date()

    # 3. 增量更新每日指标（total_mv 市值，因子值来源）
    print("\n获取每日指标（市值等）...")
    get_daily_basic()

    # get_financial()

    # 5. 计算 f1 因子
    print("\n计算 f1 因子...")
    compute_f1_factor()

    # 6. 因子评测
    print("\n因子评测...")
    simulate_trade(factor_table='f1', fac_freq='月度', n_top=5, factor_direction='大', slippage_rate=0.005)


if __name__ == '__main__':
    main()
