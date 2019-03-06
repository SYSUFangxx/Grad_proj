# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
from rqalpha.api import *
import pandas as pd
import numpy as np
import math
import os
# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):   
#选取板块
    context.stks=[]
    context.stks.append(sector("consumer discretionary"))
    context.stks.append(sector("consumer staples"))
    context.stks.append(sector("health care"))
    context.stks.append(sector("telecommunication services"))
    context.stks.append(sector("utilities")) 
    context.stks.append(sector("materials")) 

    context.flag=True
    # 确定运行频率    
    scheduler.run_daily(rebalance)

    # 手动添加银行板块   
    context.stocks = ['000001.XSHE','002142.XSHE','600000.XSHG','600015.XSHG','600016.XSHG','600036.XSHG','601009.XSHG','601166.XSHG','601169.XSHG','601288.XSHG','601328.XSHG','601398.XSHG','601818.XSHG','601939.XSHG','601988.XSHG','601998.XSHG'] 
    # 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新

def get_stocks(context, bar_dict):
    now_date = context.now.strftime('%Y-%m-%d')
    try:
        stocks_df = pd.read_excel("D:\python\Grad_proj\\test\stock_weight\%s.xls"%now_date, index_col=0)
        stocks = set(stocks_df.index)
    except:
        stocks = set([])
    return stocks

def rebalance(context, bar_dict):
    print("-" * 8, "rebalance", "-" * 8)

    stocks =  get_stocks(context, bar_dict) 
    holdings = set(get_holdings(context))

    to_buy = stocks - holdings
    to_sell = holdings - stocks
    to_buy2= stocks - holdings

    for stock in to_sell:
        if bar_dict[stock].suspended == False:
            order_target_percent(stock , 0)

    if len(to_buy) == 0:
        return

    to_buy = get_trading_stocks(to_buy, context, bar_dict)
    cash = context.portfolio.cash
    total_value=context.portfolio.total_value
    if len(to_buy) >0:
        average_value = total_value *0.025
        if average_value > total_value/len(to_buy):
            average_value = total_value/len(to_buy)

    for stock in to_buy:
        if (bar_dict[stock].suspended == False)and(context.portfolio.cash>average_value):
            order_target_value(stock, average_value)
    if context.flag==True :
        sell_open('IF88', 1)
        context.flag=False

# 得到交易的股票
def get_trading_stocks(to_buy, context, bar_dict):
    trading_stocks = []
    for stock in to_buy:
        if bar_dict[stock].suspended == False:
            trading_stocks.append(stock)

    return trading_stocks

# 持仓的股票
def get_holdings(context):
    positions = context.portfolio.stock_account.positions

    holdings = []
    for position in positions:
        if positions[position].quantity > 0:
            holdings.append(position)

    return holdings
def handle_bar(context, bar_dict):
    print("*" * 8, "hadle_bar", "*" * 8)

    # TODO: 开始编写你的算法吧！
    # IF88_future = instruments("IF88")
    now_date = context.now.strftime('%Y-%m-%d')
    try:
        stocks_df = pd.read_excel("D:\python\Grad_proj\\test\stock_weight\%s.xls"%now_date, index_col=0)
    except:
        stocks_df = pd.DataFrame()

    if not stocks_df.empty:
        ss = list(stocks_df.index)
        ws = list(stocks_df.weights)
        for i in range(len(ss)):
            order_target_percent(ss[i], ws[i])
    pass