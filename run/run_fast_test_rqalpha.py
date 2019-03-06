import subprocess

cmd = "rqalpha run -f ../tools/fast_test_rqalpha.py -s 2017-01-01 -e 2017-01-05 --plot --benchmark 000300.XSHG --account stock 50000000 --slippage 0.002"
subprocess.call(cmd.split())