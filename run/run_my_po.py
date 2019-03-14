from rqalpha import run_file

# s_date = "2016-02-08"
s_date = "2017-01-01"
e_date = "2018-01-01"
# e_date = "2019-01-01"
benchmark = "000300.XSHG"

strategy_path = "../src/my_po.py"
config = {
    "base":
    {
        "start_date": s_date,
        "end_date": e_date,
        "benchmark": benchmark,
        "accounts": {
            "stock": 50000000
        }
    },
    "mod":
    {
        "sys_analyser": {
            "enabled": True,
            "plot": True
        },
        "sys_simulation": {
            "slippage": 0.002
        }
    }
}

run_file(strategy_path, config)
