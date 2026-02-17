from pykrx import stock
import pandas as pd

start_date = "20251208"
end_date = "20260206"
ticker = "005930"

print("Testing pykrx for " + ticker + " from " + start_date + " to " + end_date)

print("--- Market Cap (stock.get_market_cap) ---")
try:
    df_cap = stock.get_market_cap(start_date, end_date, ticker)
    if df_cap is not None:
        print("Shape:", df_cap.shape)
        print(df_cap.head())
    else:
        print("df_cap is None")
except Exception as e:
    print("Error:", e)

print("--- Short Selling (stock.get_shorting_status_by_date) ---")
try:
    df_short = stock.get_shorting_status_by_date(start_date, end_date, ticker)
    if df_short is not None:
        print("Shape:", df_short.shape)
        print(df_short.head())
    else:
        print("df_short is None")
except Exception as e:
    print("Error:", e)