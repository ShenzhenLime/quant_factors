# -*- coding: utf-8 -*-
"""
主程序 - 流程控制(完成一个因子后，存档)
"""

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

    print("\n获取补充数据（如PB、MV）...")
    get_daily_basic()

    print("\n计算残差因子...")
    calc_resid()
    
    # print("\n因子评测...")
    # evaluate_factor(factor_table='stock_resids', fac_freq='日度', bench_index='000002.SH', other_name='resid')

if __name__ == '__main__':
    # main()
    group_plot('中证800', '周度', 'long', 'stock_resids')
    
















