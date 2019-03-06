import pandas as pd
import numpy as np
import os

import func_timeout
from func_timeout import func_set_timeout

from WindPy import w
w.start()
# wind数据下载超时设置
WIND_TIMEOUT = 100

class MyDataDownloader:
    def __init__(self):
        self.RQ_stocks_root = '../data/all_stocks_rqalpha/'
        self.mdt = MyDataTool()

    def get_trading_dates(self, st_date, end_date):
        """
        获取一段时间的交易日
        :param st_date: 开始日期
        :param end_date: 结束日期
        :return: 返回一段交易日（若st_date和end_date是交易日，则包含两者），类型为list
        """
        dates_path = "../data/dates/trading_days.txt"
        with open(dates_path, 'r') as file:
            trading_days = eval(file.readline())
        return [d for d in trading_days if st_date <= d <= end_date]

    def get_all_a_stocks(self, date):
        """
        获取rqalhpa股票信息数据
        :param date: 指定日期
        :return: rqalpha股票代码列表
        """
        path = os.path.join(self.RQ_stocks_root, '%s.csv'%date)
        if not os.path.exists(path):
            print(date, '不是交易日，或者该日期没有数据')
            return
        df = pd.read_csv(path, index_col=0, encoding='gbk')
        return df['order_book_id'].tolist()

    @func_set_timeout(WIND_TIMEOUT)
    def down_wind_wss_data_all_a_stocks(self, date, keys):
        """
        下载wind数据
        :param date: 日期
        :param keys: 股票信息关键字
        :return: 返回格式为wind的数据格式
        """
        all_stocks = self.get_all_a_stocks(date)
        all_stocks = self.mdt.transform_code_format(all_stocks)
        print("获取wss_data...")
        wss_data = w.wss(all_stocks, keys, "unit=1;tradeDate=%s;industryType=1" % date)
        if wss_data.ErrorCode != 0:
            print(u"下载数据出错")
            print(wss_data)
            return None
        print("成功获取wss_data。")
        return wss_data

    def down_wind_datas(self, st_date, end_date, keys, res_root):
        # 默认keys参数：下载申万一级行业分类和对应的总市值2
        # keys = ["mkt_cap_ard", "industry_sw"]
        all_dates = self.get_trading_dates(st_date, end_date)
        have_dates = [d.split('.')[0] for d in os.listdir(res_root)]
        update_dates = [d for d in all_dates if d not in have_dates]
        print(len(update_dates), update_dates)
        for cnt, d in enumerate(update_dates):
            # while True 为了不断尝试访问wind数据端，因为在家使用wifi，访问wind数据不稳定
            while True:
                try:
                    wss_data = self.down_wind_wss_data_all_a_stocks(d, keys)
                    if wss_data:
                        df = self.mdt.wind2df(wss_data)
                        break
                    else:
                        raise Exception("下载wind数据出错")
                except func_timeout.exceptions.FunctionTimedOut:
                    print("WIND下载超时（超过%d秒），正在重新尝试下载..."%WIND_TIMEOUT)
                    continue
            df.to_csv(os.path.join(res_root, d + '.csv'))
            print('Date:', d, 'Finish:', cnt+1, 'Total:', len(update_dates))


class MyDataTool:
    def __init__(self):
        pass

    def transform_code_format(self, stocks, type='rq2wind'):
        if type == 'rq2wind':
            stocks = [s.replace('XSHE', 'SZ') for s in stocks]
            stocks = [s.replace('XSHG', 'SH') for s in stocks]
        elif type == 'wind2rq':
            stocks = [s.replace('SH', 'XSHG') for s in stocks]
            stocks = [s.replace('SZ', 'XSHE') for s in stocks]
        else:
            print('转换类型错误')
            return
        return stocks

    def wind2df(self, data):
        """
        将wind格式数据转换为DataFrame格式
        :param data: wind数据，wsd或者wss数据均可
        :return: DataFrame
        """
        data_df = pd.DataFrame(data.Data)
        data_df = data_df.T
        data_df.index = data.Codes
        data_df.columns = data.Fields
        return data_df

    def gen_dummy_variable_matrix(self):
        """
        生成哑变量矩阵，将cap_industry文件夹中的文件转换为lncap_and_industry文件夹中的文件
        :return: 无，直接在函数中实现转换
        """
        src_root = '../data/cap_industry'
        res_root = '../data/lncap_and_industry'
        industry_str = "医药生物,综合,化工,建筑材料,有色金属,采掘,钢铁,电气设备,轻工制造,食品饮料," \
                       "公用事业,银行,交通运输,休闲服务,家用电器,纺织服装,建筑装饰,计算机,国防军工,房地产," \
                       "传媒,汽车,机械设备,非银金融,通信,商业贸易,电子,农林牧渔"
        industry_list = industry_str.split(',')
        cols = ['code', 'lncap'] + industry_list

        res_dict = {}

        paths = os.listdir(src_root)
        exist_paths = os.listdir(res_root)
        for p in paths:
            if p in exist_paths:
                continue

            try:
                df = pd.read_csv(os.path.join(src_root, p), index_col=0)
            except:
                df = pd.read_csv(os.path.join(src_root, p), index_col=0, encoding='gbk')
            df = df.dropna()
            res_dict['code'] = df.index.tolist()
            res_dict['lncap'] = df['MKT_CAP_ARD'].apply(np.log)
            for col in cols[2:]:
                res_dict[col] = df['INDUSTRY_SW'].apply(lambda x: 1 if x == col else 0)
            res_df = pd.DataFrame(res_dict)
            res_df.to_csv(os.path.join(res_root, p))

            print(p, 'Finish:', paths.index(p)+1, 'Total:', len(paths))

    def check_dummy_variable_matrix(self):
        """
        运行该函数，检测是否存在异常样例
        :return: 无
        """
        root = '../data/lncap_and_industry'
        paths = os.listdir(root)
        for p in paths:
            df = pd.read_csv(os.path.join(root, p), index_col=0, encoding='gbk')
            cols = df.columns.tolist()
            for ind, row in df.iterrows():
                cnt = 0
                for c in cols:
                    if c == 'code' or c == 'lncap':
                        continue
                    cnt += row[c]

                if cnt != 1:
                    print(p, ind, cnt)
            print(p, 'Finish!')

    def modify_lncap_and_industry(self):
        """
        修改test文件夹下的lncap_and_industry中的数据格式，及文件名格式，并将新的文件添加至data/lncap_and_industry文件夹下
        :return: 无
        """
        src_root = '../test/lncap_and_industry'
        res_root = '../data/lncap_and_industry'
        paths = os.listdir(src_root)
        for p in paths:
            res_p = p[:4] + '-' + p[4:6] + '-' + p[6:8] + '.csv'
            df = pd.read_csv(os.path.join(src_root, p), index_col=0, encoding='gbk')
            cols = [c.lower() if c == 'LNCAP' else c for c in df.columns]
            df.columns = cols
            df.to_csv(os.path.join(res_root, res_p))
            print(p)

    def unify_columns_in_index_weight(self):
        """
        统一index_weight文件夹下的每个文件columns
        :return: 无
        """
        data_root = '../data/old_index_weight'
        new_root = '../data/index_weight'
        index_roots = os.listdir(data_root)

        for ir in index_roots:
            paths = os.listdir(os.path.join(data_root, ir))
            for p in paths:
                df = pd.read_csv(os.path.join(data_root, ir, p), index_col=0)
                cols = df.columns.tolist()
                new_cols = []
                for c in cols:
                    if c == 'stock' or c == 'wind_code':
                        new_cols.append('code')
                    elif c == 'weight':
                        new_cols.append('i_weight')
                    else:
                        new_cols.append(c)
                df.columns = new_cols
                df.to_csv(os.path.join(new_root, ir, p), index=False)
                print(ir, p)
