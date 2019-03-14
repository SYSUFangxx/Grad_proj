from rqalpha.api import *
import pandas as pd
import numpy as np

from src.my_calculator import ReferPO

data_root = "../data/"


# @assistant function
#  return the format of rqalpha
def change_code_format_to_long(stocks):
    stocks = [s.replace('SH', 'XSHG') for s in stocks]
    stocks = [s.replace('SZ', 'XSHE') for s in stocks]
    return stocks


def change_code_format_to_short(stocks):
    stocks = [s.replace('XSHG', 'SH') for s in stocks]
    stocks = [s.replace('XSHE', 'SZ') for s in stocks]
    return stocks


def init(context):
    context.counter = 0
    context.close_df = pd.read_csv('../data/allA_data/allAclose.csv', index_col=0)
    scheduler.run_weekly(trade, tradingday=1)


def trade(context, bar_dict):
    date_data = get_previous_trading_date(context.now).strftime('%Y-%m-%d')
    print("Get data date:", date_data)

    ''' 获取仓位股票，用于生成总股池和前持仓权重，即all_stocks和list_pos_w '''
    position_stocks = list(context.portfolio.positions.keys())

    ''' 获取基准成分股，用于生成总股池和基准权重，即all_stocks和list_ben_w '''
    file = data_root + 'index_weight/HS300/%s.csv' % date_data
    # file = data_root + 'index_weight/ZZ500/%s.csv' % date_data
    df_ben = pd.read_csv(file, encoding='gbk', usecols=['code', 'i_weight'])
    ben_stocks = df_ben['code'].tolist()

    all_stocks = ben_stocks + [s for s in position_stocks if s not in ben_stocks]
    p = context.close_df.loc[:date_data, all_stocks].iloc[-2:]
    x_pred = p.iloc[-1] / p.iloc[-2]
    x_pred = x_pred.dropna()

    w_o = []
    po_stocks = change_code_format_to_long(x_pred.index.tolist())
    for s in po_stocks:
        if s in position_stocks:
            w_o.append(context.portfolio.positions[s])
        else:
            w_o.append(0)
    w_o = np.array(w_o)

    print("wosum", np.sum(w_o))

    olu_inputs = {'w_o': w_o, 'x_pred': x_pred}

    weights = ReferPO.olu(olu_inputs)

    # list_df = pd.DataFrame({'code': all_stocks, 'pos_w': list_pos_w, 'ben_w': list_ben_w, 'pr': list_pr})
    # list_df.sort_values(by='code', inplace=True)
    #
    # test_df = pd.DataFrame(X_all, index=all_stocks, columns=fields)
    # test_df['code'] = all_stocks
    # test_df['weight'] = weights
    # test_df.sort_index(inplace=True)
    # test_df = test_df[test_df['weight'] != 0]
    # test_df = test_df[['code', 'weight'] + fields]

    print("Nonzero weights", [w for w in weights if w])

    ''' 权重之和检查 '''
    check_sum = 0
    for iw in weights:
        check_sum += iw
    if check_sum != 1:
        print('weights sum: %f' % check_sum)

    ''' 股票仓位调整 '''
    # buy_list 用于保存买卖股票的列表
    buy_list = []

    for i in range(len(po_stocks)):
        if weights[i] > 0:
            buy_list.append(po_stocks[i])

    # 货币基金
    moneyFund = '510880.XSHG'

    for s in position_stocks:
        if s not in buy_list and s != moneyFund:
            order_target_percent(s, 0)

    remain_weight = 1
    for i in range(len(po_stocks)):
        if weights[i] > 0:
            ord = order_target_percent(po_stocks[i], weights[i])
            remain_weight -= weights[i]
            if ord:
                if ord._status == ORDER_STATUS.FILLED:
                    remain_weight -= weights[i]
                else:
                    print(ord, "=" * 200)
            else:
                pass
                # print(po_stocks[i], weights[i], "=" * 200)

    # 多余的钱，全部买货币基金
    if remain_weight > 0:
        order_target_percent(moneyFund, remain_weight)

    record_s_list = []
    record_w_list = []
    for s, w in zip(po_stocks+[moneyFund], weights.tolist()+[remain_weight]):
        if w > 0:
            record_s_list.append(s)
            record_w_list.append(w * 100)

    df_rt = pd.DataFrame({'stock': record_s_list, 'weight': record_w_list})
    df_rt = df_rt.sort_values(by='stock')
    file = data_root + 'cal_stock_weight/olu/%s.csv' % date_data
    df_rt.to_csv(file, index=False)


def before_trading(context):
    # before_trading
    pass


def after_trading(context):
    stocks = context.portfolio.positions.keys()

    # for i in range(0, len(stocks)):
    #     print('%s:%f' % (stocks[i], context.portfolio.positions[stocks[i]].value_percent))
