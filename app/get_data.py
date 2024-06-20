from vnstock3 import Vnstock
import os
from datetime import datetime
import pandas as pd

if "ACCEPT_TC" not in os.environ:
    os.environ["ACCEPT_TC"] = "tôi đồng ý"


def get_list():
    stock = Vnstock().stock(symbol='VN30F1M', source='VCI')
    return stock.listing.all_symbols()

def get_data_his(ticker, num_day=26, day_start=None):
    if day_start is None:
        day_start = datetime.now().strftime("%Y-%m-%d")
    stock = Vnstock().stock(symbol=ticker, source='VCI')
    df = stock.quote.history(start="2011-01-01", end=day_start)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df[-num_day:]

def get_data_realtime(ticker):
    stock = Vnstock().stock(symbol=ticker, source='VCI')
    df = stock.quote.intraday(symbol=ticker, show_log=False)
    return df['price'][0]