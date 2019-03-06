from rqalpha.api import *
import pandas as pd
import numpy as np

from src.my_calculator import calc_weight
from src.my_calculator import calc_weight_with_exprosure

data_root = "..\\data\\"


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
    print("date: ", date_data)

    file = data_root + 'index_weight\\HS300\\%s.csv' % date_data
    # file = data_root + 'index_weight\\ZZ500\\%s.csv' % date_data
    df_hs300 = pd.DataFrame()
    try:
        df_hs300 = pd.read_csv(file, encoding='gbk')
    except IOError as e:
        print("Error: ", e)

    df_hs300.dropna(how='any', inplace=True)
    df_hs300 = df_hs300.reset_index(drop=True)

    hs300_stocks = df_hs300['code'].values
    hs300_weights = df_hs300['i_weight'].values

    pr_codes = ['600519.SH', '000858.SZ', '002304.SZ', '002415.SZ', '002236.SZ',
                '600597.SH', '603288.SH', '000895.SZ', '601318.SH', '601336.SH',
                '000651.SZ', '000333.SZ', '600036.SH', '600104.SH', '600816.SH',
                '600674.SH', '600690.SH', '002081.SZ', '600886.SH', '000625.SZ',
                '600887.SH', '000423.SZ', '600276.SH', '000963.SZ', '600535.SH',
                '600271.SH', '002294.SZ', '600660.SH', '300072.SZ', '000538.SZ',
                '600066.SH', '000002.SZ', '002271.SZ', '600406.SH', '600900.SH',
                '002027.SZ', '000063.SZ', '002594.SZ', '601238.SH', '600518.SH',
                '600703.SH', '600309.SH', '002310.SZ', '002466.SZ', '002142.SZ',
                '601111.SH', '601211.SH', '600487.SH', '002008.SZ', '600383.SH',
                '601288.SH', '601988.SH', '601398.SH'
                ]
    pr_codes = []
    pr_pr = np.ones(len(pr_codes)) * 0.01

    pr_codes = change_code_format_to_long(pr_codes)
    # tmpprcodes=remove_suspended_and_st(pr_codes)
    pr_codes_top = pr_codes
    # print("len(pr_codes_top)", len(pr_codes_top))

    dict_pr = {}
    for i in range(0, len(pr_codes)):
        dict_pr[pr_codes[i]] = pr_pr[i]

    hs300_stocks = change_code_format_to_long(hs300_stocks)

    # print(c500_codes[0:10])

    dict_hs300 = {}
    for i in range(0, len(hs300_stocks)):
        dict_hs300[hs300_stocks[i]] = hs300_weights[i] / 100

    position_stocks = list(context.portfolio.positions.keys())
    all_stocks = set(pr_codes_top)
    all_stocks = all_stocks.union(hs300_stocks)
    all_stocks = all_stocks.union(position_stocks)

    file = data_root + 'lncap_and_industry\\%s.csv' % date_data
    df_X = pd.DataFrame()
    try:
        df_X = pd.read_csv(file, encoding='gbk')
    except IOError as e:
        print("Error: ", e)
    df_X.index = df_X.code

    tmp_list = change_code_format_to_long(list(df_X.index.values))

    all_stocks = all_stocks.intersection(set(tmp_list))

    all_stocks = list(all_stocks)

    # for iss in hs300_stocks:
    #     if iss not in tmp_list:
    #         print('missing hs300:%s' % iss)

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
        if all_stocks[i] in hs300_stocks:
            list_ben_w[i] = dict_hs300[all_stocks[i]]
            if list_ben_w[i] > wbmax:
                wbmax = list_ben_w[i]
            wbsum = wbsum + list_ben_w[i]
        if all_stocks[i] in pr_codes:
            list_pr[i] = dict_pr[all_stocks[i]]
        else:
            pass
            # print('no pr %s' % all_stocks[i])
    '''
    print("wosum", wosum)
    print("wbsum", wbsum)
    print("wbmax", wbmax)
    '''
    for i in range(0, n_stocks):
        if wosum != 0:
            list_pos_w[i] = list_pos_w[i] / wosum
        if wbsum != 0:
            list_ben_w[i] = list_ben_w[i] / wbsum

            #       for i in range(0,20):
            #           print('%s: %f %f %f' %(all_stocks[i],list_pos_w[i],list_ben_w[i],list_pr[i]))

    industries = ['医药生物', '综合', '化工', '建筑材料', '有色金属', '采掘', '钢铁', '电气设备', '轻工制造', '食品饮料', '公用事业', '银行', '交通运输',
                  '休闲服务', '家用电器', '纺织服装', '建筑装饰', '计算机', '国防军工', '房地产', '传媒', '汽车', '机械设备', '非银金融', '通信', '商业贸易',
                  '电子', '农林牧渔']
    # fields=['LNCAP','ETOP', 'BTOP', 'CTOP','BETA','CMRA', 'RSTR', 'DASTD', 'HSIGMA','STOM', 'STOA', 'STOQ']

    fields = ['lncap']
    # fields=[]
    for ins in range(0, len(industries)):
        fields.append(industries[ins])
    all_ss = change_code_format_to_short(all_stocks)
    tmpX = df_X.loc[all_ss, fields].values

    # tmp_cap对应着log市值
    tmp_cap = tmpX[:, 0].T
    n_cap = len(tmp_cap)
    for kkk in range(0, n_cap):
        tmpX[kkk, 0] = tmpX[kkk, 0] * tmpX[kkk, 0] * tmpX[kkk, 0]

    for p in range(0, len(fields)):
        temp = tmpX[:, p].T
        mean_ = temp.mean()
        std_ = temp.std()
        nn = len(temp)
        for q in range(0, nn):
            if std_ == 0:
                tmpX[q, p] = 0
            else:
                temp2 = (tmpX[q, p] - mean_) / std_
                if temp2 > 3:
                    temp2 = 3
                if temp2 < -3:
                    temp2 = -3
                tmpX[q, p] = temp2

    # print("tmpX.shape", tmpX.shape)
    penalty = np.linalg.norm(np.dot(list_ben_w, tmpX)) ** 2
    # print("penalty", penalty)
    penalty_r = np.linalg.norm(list_pr)
    # print("penalty_r", penalty_r)
    # weights = calc_weight(list_pos_w, list_ben_w, list_pr, tmpX, upper=0.04, c=0.006, l=0)
    weights=calc_weight_with_exprosure(list_pos_w,list_ben_w,list_pr,tmpX,upper=0.04,c=0.006,exprosure=0.1)

    # print("weights", weights)
    # print("nonzero", weights.nonzero())
    # w = 1.0 / len(pr_codes)
    # for i in range(len(all_stocks)):
    #     if all_stocks[i] in pr_codes:
    #         weights[i] = w
    #     else:
    #         weights[i] = 0

    tmp_sum = 0
    for i in range(0, len(weights)):
        tmp_sum = tmp_sum + weights[i]
    print('weights sum: %f' % tmp_sum)

    # buy_list 用于保存买卖股票的列表
    # trade_flags 用于保存是否需要修改仓位的标识
    buy_list = []
    trade_flags = []

    for i in range(len(all_stocks)):
        if weights[i] > 0:
            buy_list.append(all_stocks[i])
            trade_flags.append(True)

    moneyFund = '510880.XSHG'

    for s in position_stocks:
        if s not in buy_list and s != moneyFund:
            # print('sell %s' % s)
            order_target_percent(s, 0)

    remain_weight = 1
    for i in range(len(all_stocks)):
        if weights[i] > 0:
            order_target_percent(all_stocks[i], weights[i])
            remain_weight -= weights[i]

    # 多余的钱，全部买货币基金
    if remain_weight > 0:
        order_target_percent(moneyFund, remain_weight)

    my_list = {}
    my_list_weights = []
    record_stocks = all_stocks
    for i in range(0, len(record_stocks)):
        if weights[i] > 0:
            my_list[record_stocks[i]] = weights[i]
            my_list_weights.append(weights[i] * 100)

    if remain_weight > 0:
        my_list[moneyFund] = remain_weight
        my_list_weights.append(remain_weight * 100)

    df_rt = pd.DataFrame(my_list_weights, index=my_list.keys(), columns=['weight'])
    df_rt.index.name = 'stock'

    file = data_root + 'cal_stock_weight\\%s.csv' % date_data

    # writer=pd.ExcelWriter(file)
    df_rt.to_csv(file)


def before_trading(context):
    # before_trading
    pass


def after_traing(context):
    stocks = context.portfolio.positions.keys()

    for i in range(0, len(stocks)):
        print('%s:%f' % (stocks[i], context.portfolio.positions[stocks[i]].value_percent))

