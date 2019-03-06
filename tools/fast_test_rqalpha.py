from rqalpha.api import *


def change_code_format_to_long(stocks):
    stocks = [s.replace('SH', 'XSHG') for s in stocks]
    stocks = [s.replace('SZ', 'XSHE') for s in stocks]
    return stocks

def init(context):
    pass

def handle_bar(context, bar_dict):
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
    rq_codes = change_code_format_to_long(pr_codes)
    for c in rq_codes:
        print(instruments(c))