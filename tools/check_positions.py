import pandas as pd
import os

root = '../test/stock'
paths = os.listdir(root)

buy_stocks = {}
for p in paths:
    df = pd.read_excel(os.path.join(root, p), index_col=0)
    buy_stocks[p] = df.index
    weight = df.weight

all_stocks = []
for p in buy_stocks:
    for s in buy_stocks[p]:
        if s not in all_stocks:
            all_stocks.append(s)

print(len(all_stocks))
