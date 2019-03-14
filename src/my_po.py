from rqalpha.api import *
import pandas as pd
import numpy as np

from src.my_calculator import calc_weight
from src.my_calculator import calc_weight_with_exprosure

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


def remove_suspended_and_st(stocks):
    return [s for s in stocks if not is_st_stock(s) and not is_suspended(s)]


def remove_small_lncap(stocks, lncaps):
    return [s for s, lc in zip(stocks, lncaps) if lc > 23.719]


def remove_low_ep(stocks, eps, lncaps):
    return [s for s, ep, lc in zip(stocks, eps, lncaps) if ep > 0.02 and lc > 23.719]


def init(context):
    context.counter = 0
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
    df_ben.dropna(inplace=True)
    df_ben = df_ben.reset_index(drop=True)
    df_ben['code'] = change_code_format_to_long(df_ben['code'])
    df_ben['i_weight'] = df_ben['i_weight'] / 100
    ben_stocks = df_ben['code'].tolist()
    s_ben_weights = df_ben['i_weight'].copy()
    s_ben_weights.index = ben_stocks

    good_codes = ['600519.SH', '000858.SZ', '002304.SZ', '002415.SZ', '002236.SZ',
                  '600597.SH', '603288.SH', '000895.SZ', '601318.SH', '601336.SH',
                  '000651.SZ', '000333.SZ', '600036.SH', '600104.SH', '600816.SH',
                  '600674.SH', '600690.SH', '002081.SZ', '600886.SH', '000625.SZ',
                  '600887.SH', '000423.SZ', '600276.SH', '000963.SZ', '600535.SH',
                  '600271.SH', '002294.SZ', '600660.SH', '300072.SZ', '000538.SZ',
                  '600066.SH', '000002.SZ', '002271.SZ', '600406.SH', '600900.SH',
                  '002027.SZ', '000063.SZ', '002594.SZ', '601238.SH', '600518.SH',
                  '600703.SH', '600309.SH', '002310.SZ', '002466.SZ', '002142.SZ',
                  '601111.SH', '601211.SH', '600487.SH', '002008.SZ', '600383.SH',
                  '601288.SH', '601988.SH', '601398.SH']
    good_codes = change_code_format_to_long(good_codes)
    ret_good = np.ones(len(good_codes)) * 0.01

    # good_codes = []

    # 使用周收益率
    # ret_df = pd.read_csv(data_root + 'ret/ret_weekly.csv', index_col=0)
    # ret_df = ret_df.shift(-1)
    # ret_s = ret_df.loc[date_data].dropna()
    # ret_s = ret_s[ben_stocks].dropna()
    # ret_s = ret_s.sort_values()[-50:]
    # good_codes = ret_s.index.tolist()
    # ret_good = ret_s.values
    # ret_good = np.ones(len(good_codes)) * 0.01

    # print("len(good_codes)", len(good_codes))

    dict_pr = {}
    for i in range(0, len(good_codes)):
        dict_pr[good_codes[i]] = ret_good[i]

    all_stocks = set(good_codes)
    all_stocks = all_stocks.union(ben_stocks)
    all_stocks = all_stocks.union(position_stocks)

    file = data_root + 'lncap_and_industry/%s.csv' % date_data
    df_X = pd.DataFrame()
    try:
        df_X = pd.read_csv(file, encoding='gbk')
    except IOError as e:
        print("Error: ", e)
    df_X.index = change_code_format_to_long(df_X.code)
    # df_X.dropna(inplace=True)

    all_stocks = all_stocks.intersection(set(df_X.index))
    all_stocks = list(all_stocks)

    for iss in ben_stocks:
        if iss not in df_X.index:
            print('missing some data of stock in benchmark:%s, no x_matrix data' % iss)

    # print(all_stocks[0:10])

    n_stocks = len(all_stocks)
    '''print("n_stocks", n_stocks)'''
    list_pos_w = np.zeros(n_stocks)
    list_ben_w = np.zeros(n_stocks)
    list_pr = np.ones(n_stocks) * (-0.1)

    """
    wbsum: 基准权重之和
    wbmax：基准权重最大值
    wosum：原来的基准之和
    """
    wbsum = 0
    wbmax = 0
    wosum = 0
    for i in range(0, n_stocks):
        if all_stocks[i] in position_stocks:
            pos = context.portfolio.positions[all_stocks[i]]
            list_pos_w[i] = pos.value_percent
            wosum = wosum + list_pos_w[i]
        if all_stocks[i] in ben_stocks:
            list_ben_w[i] = s_ben_weights[all_stocks[i]]
            if list_ben_w[i] > wbmax:
                wbmax = list_ben_w[i]
            wbsum = wbsum + list_ben_w[i]
        if all_stocks[i] in good_codes:
            list_pr[i] = dict_pr[all_stocks[i]]
        else:
            pass
            # print('no pr %s' % all_stocks[i])

    print("wosum", wosum)
    print("wbsum", wbsum)
    print("wbmax", wbmax)

    for i in range(0, n_stocks):
        if wosum != 0:
            list_pos_w[i] = list_pos_w[i] / wosum
        if wbsum != 0:
            list_ben_w[i] = list_ben_w[i] / wbsum

    fields = ['lncap']
    for fd in fields:
        mean_fd = df_X[fd].mean()
        std_fd = df_X[fd].std()
        if std_fd == 0:
            df_X[fd] = 0
            continue
        df_X[fd] = df_X[fd].apply(lambda x: (x - mean_fd) / std_fd)
        # 异常值处理
        df_X[fd] = df_X[fd].apply(lambda x: 3 if x > 3 else (-3 if x < -3 else x))

    industries = ['医药生物', '综合', '化工', '建筑材料', '有色金属', '采掘', '钢铁', '电气设备', '轻工制造', '食品饮料',
                  '公用事业', '银行', '交通运输', '休闲服务', '家用电器', '纺织服装', '建筑装饰', '计算机', '国防军工',
                  '房地产', '传媒', '汽车', '机械设备', '非银金融', '通信', '商业贸易', '电子', '农林牧渔']
    fields += industries
    X_all = df_X.loc[all_stocks, fields].values

    n_pos = [lp for lp in list_pr if lp >= 0.01]
    n_neg = [lp for lp in list_pr if lp == -0.1]
    print("list_pr", list_pr.tolist())
    print(len(n_pos), len(n_neg))

    weights = calc_weight(list_pos_w, list_ben_w, list_pr, X_all, upper=0.04, c=0.0006, l=1)
    # weights=calc_weight_with_exprosure(list_pos_w,list_ben_w,list_pr,X_all,upper=0.04,c=0.0006,exprosure=0.1)

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
    # trade_flags 用于保存是否需要修改仓位的标识
    buy_list = []
    trade_flags = []

    for i in range(len(all_stocks)):
        if weights[i] > 0:
            buy_list.append(all_stocks[i])
            trade_flags.append(True)

    # 货币基金
    moneyFund = '510880.XSHG'

    for s in position_stocks:
        if s not in buy_list and s != moneyFund:
            order_target_percent(s, 0)

    remain_weight = 1
    for i in range(len(all_stocks)):
        if weights[i] > 0:
            ord = order_target_percent(all_stocks[i], weights[i])
            remain_weight -= weights[i]
            if ord:
                if ord._status == ORDER_STATUS.FILLED:
                    remain_weight -= weights[i]
                else:
                    print(ord, "=" * 200)
            else:
                pass
                # print(all_stocks[i], weights[i], "=" * 200)

    # 多余的钱，全部买货币基金
    if remain_weight > 0:
        order_target_percent(moneyFund, remain_weight)

    record_s_list = []
    record_w_list = []
    for s, w in zip(all_stocks+[moneyFund], weights.tolist()+[remain_weight]):
        if w > 0:
            record_s_list.append(s)
            record_w_list.append(w * 100)

    df_rt = pd.DataFrame({'stock': record_s_list, 'weight': record_w_list})
    df_rt = df_rt.sort_values(by='stock')
    file = data_root + 'cal_stock_weight/%s.csv' % date_data
    df_rt.to_csv(file, index=False)


def before_trading(context):
    # before_trading
    pass


def after_trading(context):
    stocks = context.portfolio.positions.keys()

    # for i in range(0, len(stocks)):
    #     print('%s:%f' % (stocks[i], context.portfolio.positions[stocks[i]].value_percent))
