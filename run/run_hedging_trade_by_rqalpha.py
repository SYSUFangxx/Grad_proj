import subprocess

# bt_st_date = "2016-01-08"
bt_st_date = "2017-01-08"
bt_end_date = "2018-01-01"

cmd = "rqalpha run -f ../src/hedging_trade_by_rqalpha.py -s {} -e {} --plot --benchmark 000300.XSHG" \
      " --account stock 50000000 --account future 50000000 --slippage 0.002".format(bt_st_date, bt_end_date)
# cmd = "rqalpha run -f ../src/hedging_trade_by_rqalpha.py -s {} -e {} --plot --benchmark 000300.XSHG" \
#       " --account stock 50000000 --slippage 0.002".format(bt_st_date, bt_end_date)
subprocess.call(cmd.split())
