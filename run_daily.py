# -*- coding: utf-8 -*-
"""
每日运行脚本 - 一键更新数据并生成最新信号
"""

import pandas as pd
import sqlalchemy
from datetime import datetime
import os


# ========== 因子配置 ==========
# 使用精简后的120个因子（40个精简原始 + 60个逻辑差异化 + 20个回撤控制）
GOOD_FACTORS = [f'alpha_{i:03d}' for i in range(1, 121)]

# ========== 数据库配置 ==========
DB_CONFIG = {
    'user': 'intern',
    'password': 'Hz123456',
    'host': 'rm-uf6c8yi6zdq6ea7p1qo.mysql.rds.aliyuncs.com',
    'port': '3306',
    'database': 'stock'
}

# 股票池（和 extract_portfolio.py 保持一致）
STOCK_POOL = [
    # 高估值成长股
    '300476.SZ', '300251.SZ', '002595.SZ', '002074.SZ', '301498.SZ', '300458.SZ',
    '300888.SZ', '300623.SZ', '001696.SZ', '300100.SZ', '300432.SZ', '300570.SZ',
    '300660.SZ', '301200.SZ', '002245.SZ', '300083.SZ', '301291.SZ', '301510.SZ',
    '002901.SZ', '300870.SZ', '301061.SZ', '300913.SZ', '300153.SZ', '300602.SZ',
    '300620.SZ', '301377.SZ', '301127.SZ', '301109.SZ', '000880.SZ', '300975.SZ',
    '300708.SZ', '300611.SZ', '001267.SZ', '301187.SZ', '003010.SZ', '301080.SZ',
    '301389.SZ', '001380.SZ', '001288.SZ', '300231.SZ', '002869.SZ', '001283.SZ',
    '301141.SZ', '002637.SZ', '300484.SZ', '301379.SZ', '301183.SZ', '301289.SZ',
    '301361.SZ', '300787.SZ', '001238.SZ', '003029.SZ', '001223.SZ', '300731.SZ',
    '300824.SZ', '301509.SZ', '300980.SZ',
    # 低估值价值股
    '002244.SZ', '000791.SZ', '002198.SZ', '000651.SZ', '002142.SZ', '000776.SZ',
    '002736.SZ', '000002.SZ', '001872.SZ', '002128.SZ', '002966.SZ', '000783.SZ',
    '000728.SZ', '000027.SZ', '002608.SZ', '000959.SZ', '002948.SZ', '000703.SZ',
    '300724.SZ', '000050.SZ', '002120.SZ', '000623.SZ', '000069.SZ', '002936.SZ',
    '000591.SZ', '002958.SZ', '300677.SZ', '001227.SZ', '000685.SZ', '002091.SZ',
    '002092.SZ', '002807.SZ', '002839.SZ', '002061.SZ', '002233.SZ', '000498.SZ',
    '000690.SZ', '000950.SZ', '000885.SZ', '002250.SZ', '002382.SZ', '002375.SZ',
    '000507.SZ', '300057.SZ', '300055.SZ', '300664.SZ', '300867.SZ'
]


def get_db_engine():
    """创建数据库连接"""
    return sqlalchemy.create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}",
        poolclass=sqlalchemy.pool.NullPool
    )


def download_latest_data():
    """从数据库下载最新数据"""
    print("=" * 50)
    print("步骤1: 下载最新数据")
    print("=" * 50)
    
    # 读取现有数据，获取最新日期
    portfolio_file = 'portfolio 2020-2025.12.csv'
    if os.path.exists(portfolio_file):
        existing = pd.read_csv(portfolio_file, encoding='utf-8')
        existing['DT'] = pd.to_datetime(existing['DT'])
        last_date = existing['DT'].max()
        print(f"现有数据最新日期: {last_date.strftime('%Y-%m-%d')}")
    else:
        print(f"未找到 {portfolio_file}，请先运行 extract_portfolio.py")
        return False
    
    # 连接数据库
    print("连接数据库...")
    engine = get_db_engine()
    
    # 下载新数据
    start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    stock_list_str = "','".join(STOCK_POOL)
    query = f"""
        SELECT * FROM marketinfo 
        WHERE DT >= '{start_date}' 
        AND TRADE_CODE IN ('{stock_list_str}')
        ORDER BY DT, TRADE_CODE
    """
    
    print(f"查询 {start_date} 之后的数据...")
    new_data = pd.read_sql(query, engine)
    
    if len(new_data) == 0:
        print("没有新数据")
        return False
    
    new_data['DT'] = pd.to_datetime(new_data['DT'])
    print(f"下载到 {len(new_data)} 条新数据")
    print(f"新数据日期范围: {new_data['DT'].min().strftime('%Y-%m-%d')} ~ {new_data['DT'].max().strftime('%Y-%m-%d')}")
    
    # 合并数据
    updated = pd.concat([existing, new_data], ignore_index=True)
    updated = updated.sort_values(['TRADE_CODE', 'DT'])
    updated = updated.drop_duplicates(subset=['DT', 'TRADE_CODE'], keep='last')
    
    # 保存
    updated.to_csv(portfolio_file, index=False, encoding='utf-8')
    print(f"已更新 {portfolio_file}")
    print(f"总记录数: {len(updated)}, 日期范围: {updated['DT'].min().strftime('%Y-%m-%d')} ~ {updated['DT'].max().strftime('%Y-%m-%d')}")
    
    return True


def run_model():
    """运行模型生成信号"""
    print("\n" + "=" * 50)
    print("步骤2: 运行模型")
    print("=" * 50)
    
    # 导入并运行模型
    import multi_factor_model as mfm
    ret_df, trade_df, final_holdings, latest_signal = mfm.main()
    
    return latest_signal


def main():
    print(f"\n{'='*60}")
    print(f"  多因子选股 - 每日运行  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")
    
    # 1. 下载最新数据
    has_new_data = download_latest_data()
    
    # 2. 运行模型（无论是否有新数据都运行，以便查看当前持仓）
    latest_signal = run_model()
    
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)
    print("\n查看结果:")
    print("  - latest_signal.csv: 当前持仓信号（用于QMT交易）")
    print("  - multi_factor_result.png: 回测图表")
    print("  - latest_holdings.png: 持仓可视化 (运行 visualize_trades.py)")


if __name__ == '__main__':
    main()
