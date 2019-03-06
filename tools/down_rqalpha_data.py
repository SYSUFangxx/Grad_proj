import os
from rqalpha.api import *

key_dict = {
    "abbrev_symbol": "名称缩写",
    "board_type": "板块类别",
    "concept_names": "概念股分类",
    "de_listed_date": "退市日期",
    "exchange": "交易所",
    "industry_code": "行业分类代码",
    "industry_name": "行业分类名称",
    "listed_date": "上市日期",
    "order_book_id": "证券代码",
    "round_lot": "一手股数",
    "sector_code": "板块缩写代码",
    "sector_code_name": "板块代码名",
    "special_type": "特别处理状态",
    "status": "合约状态",
    "symbol": "证券简称",
    "type": "合约类型"
}

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    logger.info("init")

def before_trading(context):
    pass

# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    get_all_stocks(context)
    # test_get_a_stock()

def get_all_stocks(context):
    stocks = all_instruments('CS')
    now = context.now.strftime('%Y-%m-%d')
    new_stocks = stocks[['order_book_id', 'symbol']]
    new_stocks['order_book_id'] = new_stocks['order_book_id'].apply(lambda x: x.replace('XSHE', 'SZ')).apply(lambda x: x.replace('XSHG', 'SH'))
    new_stocks['date'] = str(context.now)
    new_stocks.columns = ['wind_code', 'sec_name', 'date']
    new_stocks.index = [i for i in range(len(new_stocks.index))]
    new_stocks[['date', 'wind_code', 'sec_name']].to_csv('../data/all_stocks/%s.csv' % now)
    # new_stocks[['date', 'wind_code', 'sec_name']].to_csv('D:/python/Grad_proj/data/all_stocks/%s.csv' % now)
    print(now, 'Finish!')

def test_get_a_stock():
    s_info = instruments('000006.XSHE')