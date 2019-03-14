from rqalpha.api import *
import datetime
import numpy as np
import pandas as pd

def init(context):
    scheduler.run_weekly(trade, tradingday=-1)

def trade(context, bar_dict):
    print(context.now)

def handle_bar(context, bar_dict):
    st_date = '2016-01-01'
    end_date = '2019-01-01'

    # # 更新交易日序列，st_date为起始日期，end_date为结束日期
    # update_trading_dates(st_date, end_date)

    # 更新test_date
    update_test_dates()

    # # 更新周收益率
    # mdt = MyRQDataTool()
    # codes = mdt.get_all_a_stock_one_date(end_date)['order_book_id'].tolist()
    # mdt.get_ret_weekly(st_date, end_date, codes)

    # # # 测试MyTimeTool类
    # mtt = MyRQTimeTool()
    # # print('\n'.join(mtt.get_trading_dates_str(st_date, end_date)))
    # # # 测试MyDataTool类
    # mdt = MyRQDataTool()
    # d = st_date
    # while d <= end_date:
    #     res = mdt.get_all_a_stock_one_date(d)
    #     res.to_csv('../data/all_stocks_rqalpha/%s.csv'%d)
    #     print(d, 'Finish!')
    #     d = mtt.get_next_trading_date(d)
    # print("End...")


def update_test_dates(e_date='2019-01-01'):
    file = "../data/dates/test_date.csv"
    date_df = pd.read_csv(file, index_col=0)
    test_dates = date_df['date'].tolist()
    mtt = MyRQTimeTool()
    while test_dates[-1] <= e_date:
        test_dates.append(mtt.get_next_last_day(test_dates[-1]))
    res_df = pd.DataFrame({'date': test_dates})
    res_df.to_csv(file)


def update_trading_dates(s_date, e_date):
    res_path = "../data/dates/trading_days.txt"
    mtt = MyRQTimeTool()
    tds = mtt.get_trading_dates_str(s_date, e_date)
    with open(res_path, 'w') as file:
        file.writelines(str(tds))


class MyRQTimeTool:
    def __init__(self):
        pass

    def get_previous_trading_date(self, date):
        return str(get_previous_trading_date(date))[:10]

    def get_next_trading_date(self, date):
        return str(get_next_trading_date(date))[:10]

    def get_next_first_day(self, date):
        start = datetime.datetime.strptime(date, '%Y-%m-%d')
        next = datetime.datetime.strptime(date, '%Y-%m-%d')
        while True:
            next = get_next_trading_date(str(next)[:10])

            if (next - start).days <= 4 and start.weekday() < next.weekday():
                continue
            else:
                return str(next)[:10]

    def get_next_last_day(self, date):
        nfd = self.get_next_first_day(date)
        pre_date = get_previous_trading_date(nfd)
        nld = str(pre_date)[:10]
        if nld == date:
            # 若条件成立，说明date是这周的最后一天
            nfd = self.get_next_first_day(nfd)
            pre_date = get_previous_trading_date(nfd)
            nld = str(pre_date)[:10]
        return nld

    def get_trading_dates_str(self, st_date, end_date):
        tds = get_trading_dates(st_date, end_date)
        return [str(d)[:10] for d in tds]


class MyRQDataTool:
    def __init__(self):
        pass

    def get_all_a_stock_one_date(self, date, keys=None):
        return all_instruments('CS', date)[['order_book_id', 'symbol']] if not keys else all_instruments('CS', date)[keys]

    def get_all_a_stocks(self, st_date, end_date, keys=None):
        """
        获取所有A股市场的股票信息，默认包括代码和名称
        :param st_date: 开始查询日期
        :param end_date: 结束查询日期，包含
        :return: 返回一个字典，{date1: df1, date2: df2,...}
        """
        res_dict = {}
        d = st_date
        while d <= end_date:
            res_dict[d] = self.get_all_a_stock_one_date(d)
        return res_dict

    def get_price(self, order_book_ids, start_date, end_date, frequency='1d', fields='close', adjust_type='pre',
                  skip_suspended=False):
        if adjust_type != 'pre' or skip_suspended:
            print("Sorry, there are something not think about.\n"
                  " Please think about the parameter 'adjust_type' or 'skip_suspended' or 'end_date'.")
            return

        if isinstance(order_book_ids, str):
            order_book_ids = [order_book_ids]
        if isinstance(fields, str):
            fields = [fields]

        trading_dates = get_trading_dates(start_date, end_date)
        trading_dates = [trading_date.strftime('%Y-%m-%d') for trading_date in trading_dates]

        bar_count = len(trading_dates)
        nan_arr = np.array([np.nan for i in range(bar_count)])

        dict_pn = {}
        for field in fields:
            dict_df = {}
            for order_book_id in order_book_ids:
                hist_order = history_bars(order_book_id, bar_count, frequency, field)

                # drop the data which is not complete
                if len(hist_order) != bar_count:
                    # print(order_book_id, field, str(len(hist_order)))
                    hist_order = nan_arr
                dict_df[order_book_id] = hist_order
                print(field, fields.index(field)+1, len(fields),
                      order_book_id, order_book_ids.index(order_book_id)+1, len(order_book_ids))
            dict_pn[field] = pd.DataFrame(dict_df, index=trading_dates)
        return dict_pn

    def get_ret_weekly(self, st_date, end_date, codes):
        """
        直接写入文件“ret_weekly.csv”
        文件index为date，每一周的最后一个交易日，columns对应stock，每一只股票
        矩阵中每个值代表对应date这一周的收益率
        :param st_date: 
        :param end_date: 
        :param codes: 
        :return: 
        """
        if isinstance(codes, str):
            codes = [codes]

        mtt = MyRQTimeTool()
        ret_d = []
        d = mtt.get_next_last_day(st_date)
        while d <= end_date:
            ret_d.append(d)
            d = mtt.get_next_last_day(d)

        dict_pn = self.get_price(codes, st_date, end_date)
        close_df = dict_pn['close']
        # print(close_df.index)
        close_last_df = close_df.loc[ret_d, :]
        ret_df = close_last_df.pct_change()
        # ret_df = ret_df.dropna()
        ret_df.to_csv('../data/ret/ret_weekly.csv')
