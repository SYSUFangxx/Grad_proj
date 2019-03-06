from tools.data_tool import MyDataDownloader, MyDataTool

if __name__ == '__main__':
    s_date = '2017-11-18'
    e_date = '2019-01-01'
    keys = ["mkt_cap_ard", "industry_sw"]
    res_root = '../data/cap_industry'
    mdd = MyDataDownloader()
    mdt = MyDataTool()

    # # 下载rqalpha全市场A股的申万一级行业分类和对应的总市值2
    # mdd.down_wind_datas(s_date, e_date, keys, res_root)

    # # 生成哑变量矩阵，并检验是否生成异常值
    # mdt.gen_dummy_variable_matrix()
    # mdt.check_dummy_variable_matrix()

    # # 修改test文件夹下的lncap_and_industry中的数据格式，及文件名格式，并将新的文件添加至data/lncap_and_industry文件夹下
    # mdt.modify_lncap_and_industry()

    # 统一index_weight下的各个指数权重文件的列名
    mdt.unify_columns_in_index_weight()