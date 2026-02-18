
from pykrx import stock
import pandas as pd

date = "20251208"
ticker = "005930"

print("Testing pykrx single date fetch for " + date)

print("--- Market Cap by Ticker (stock.get_market_cap_by_ticker) ---")
try:
    df_cap = stock.get_market_cap_by_ticker(date)
    if df_cap is not None:
        print("Shape:", df_cap.shape)
        # Filter for Samsung Electronics
        if ticker in df_cap.index:
            print(df_cap.loc[ticker])
        else:
            print("Ticker " + ticker + " not found in result")
    else:
        print("df_cap is None")
except Exception as e:
    print("Error:", e)

print("--- Short Selling by Ticker (stock.get_shorting_status_by_ticker) ---")
try:
    # Note: get_shorting_status_by_ticker might not exist or be named differently?
    # Checking docs or common usage. usually it is get_shorting_status_by_date for range.
    # But there is likely a function for single date.
    # Let's try get_shorting_status_by_date with same start/end
    df_short = stock.get_shorting_status_by_date(date, date, ticker)
    if df_short is not None:
        print("Shape (by_date single day):", df_short.shape)
        print(df_short)
    else:
        print("df_short (by_date single day) is None")
except Exception as e:
    print("Error:", e)
