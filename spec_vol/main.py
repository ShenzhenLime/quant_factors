# -*- coding: utf-8 -*-
"""
主程序 - 流程控制(完成一个因子后，存档)
"""

from quant_infra.get_data import get_stock_data_by_date, get_daily_basic
from quant_infra.factor_analyze import evaluate_factor, group_plot, ic_plot, ic_half_life
from quant_infra.factor_calc import calc_resid, calc_spec_vol
from quant_infra.trade import simulate_trade
from datetime import datetime

factor_table = 'spec_vol'
# ====================  主流程  ====================
def main():
    """主流程 - 一键运行"""
    ## 对于个股数据，按日增量获取。
    # 获取全量股票日线（Tushare更推荐按日循环，所以此处不对股票进行筛选）
    # 不用指定日期，会在代码内部自动填补最新数据

    print("\n获取股票日线数据...")
    get_stock_data_by_date()

    print("\n获取补充数据（如PB、MV）...")
    get_daily_basic()

    print("\n计算残差因子...")
    calc_resid()

    print("\n计算特质波动率因子（近20日残差波动率）...")
    calc_spec_vol()

    print("\n因子评测...")
    evaluate_factor(factor_table=factor_table, fac_freq='日度', bench_index='000002.SH')

    # print("\n模拟交易...")
    # simulate_trade(factor_table=factor_table, fac_freq='月度', n_top=5, factor_direction='小',slippage_rate = 0.005)

if __name__ == '__main__':
    # main()
    # group_plot('全市场', '月度', 'short', factor_table)
    # ic_plot('全市场', '月度', factor_table)
    ic_half_life(factor_table=factor_table, max_lag=200)
    
