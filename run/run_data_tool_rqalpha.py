import subprocess

cmd = "rqalpha run -f ../tools/data_tool_rqalpha.py -s 2018-12-28 -e 2018-12-28 --plot --benchmark 000300.XSHG --account stock 50000000 --slippage 0.002"
subprocess.call(cmd.split())
