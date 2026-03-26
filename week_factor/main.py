# -*- coding: utf-8 -*-
"""
主程序 - 流程控制(完成一个因子后，存档)
"""

from factor_construct import compute_week_effect
from quant_infra.get_data import get_stock_data_by_date, get_daily_basic
from quant_infra.factor_analyze import evaluate_factor, group_plot
from quant_infra.factor_calc import calc_resid
from datetime import datetime
# ====================  主流程  ====================
def main():
    """主流程 - 一键运行"""
    ## 对于个股数据，按日增量获取。
    # 获取全量股票日线（Tushare更推荐按日循环，所以此处不对股票进行筛选）
    # 不用指定日期，会在代码内部自动填补最新数据


    print("\n获取股票日线数据...")
    get_stock_data_by_date()
    ## 获取补充数据并计算因子
    print("\n获取补充数据（如PB、MV）...")
    get_daily_basic()

    print("\n计算残差因子...")
    calc_resid()

    # # 计算周末效应因子
    print("\n计算周末效应因子...")
    compute_week_effect()
    
    # #因子评测
    print("\n因子评测...")
    evaluate_factor(factor_table='week_factor', fac_freq='周度', bench_index='000002.SH')

if __name__ == '__main__':
    main()
    # group_plot('全市场', '周度', 'long', 'week_factor')
    
















