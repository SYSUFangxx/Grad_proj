import subprocess

# s_date = "2016-01-08"
s_date = "2017-01-01"
e_date = "2017-11-19"
# e_date = "2019-01-01"
benchmark = "000300.XSHG"

cmd = f"rqalpha run -f ../src/my_po.py -s {s_date} -e {e_date} --plot" \
      f" --benchmark {benchmark} --account stock 50000000 --slippage 0.002"
subprocess.call(cmd.split())
