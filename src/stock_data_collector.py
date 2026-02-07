# collect_stock_data.py
# 한번 실행한 경우 몇 거래일 뒤에 다시 실행해야 오류가 발생하지 않음
import datetime
import time
import pandas as pd
import numpy as np
from pykrx import stock
from db_setup import get_db_connection

def get_market_ohlcv_with_fallback(start_date, end_date, ticker):
    """
    KRX 원본가 기준(adjusted=False)으로 OHLCV를 조회.
    """
    return stock.get_market_ohlcv(start_date, end_date, ticker, adjusted=False)

def collect_stock_data(per_threshold=10, pbr_threshold=1, div_threshold=3.5):
    conn = get_db_connection()  # Establish a connection to the database
    cur = conn.cursor()  # Create a cursor object to interact with the database
    
    current_date = datetime.datetime.now()  # Get the current date
    cur.execute("SELECT ticker FROM ticker_list")  # Fetch all tickers from the ticker_list table
    all_tickers = [row[0] for row in cur.fetchall()]

    for ticker in all_tickers:
        cur.execute(f"SELECT status FROM ticker_status WHERE ticker = '{ticker}'")  # Check the status of each ticker
        status = cur.fetchone()
        if status and status[0] == 'completed':
            continue  # Skip if the ticker status is 'completed'

        if ticker[-1] != '0':
            print(f'{ticker} skipped because it does not end with 0')
            continue  # Skip tickers that do not end with '0'

        cur.execute(f"SELECT MAX(date) FROM stock_data WHERE ticker = '{ticker}'")  # Check the last recorded date for the ticker
        last_date = cur.fetchone()[0]
        if last_date:
            last_recorded_date = pd.to_datetime(last_date)
            if (current_date - last_recorded_date).days <= 28:
                print(f'{ticker} already done')
                continue  # Skip if data was updated within the last 28 days

        start_date = (last_date + datetime.timedelta(days=1)).strftime("%Y%m%d") if last_date else "19800102"
        end_date = current_date.strftime("%Y%m%d")

        time.sleep(3)
        df2 = stock.get_market_fundamental(start_date, end_date, ticker)
        if df2.empty or 'PER' not in df2.columns:
            cur.execute(f"UPDATE ticker_status SET status = 'completed' WHERE ticker = '{ticker}'")
            conn.commit()
            print(f'{ticker} no.2 pass condition')
            time.sleep(3)
            continue  # Mark as 'completed' if no fundamental data is found

        if ('PER' in df2.columns and 'PBR' in df2.columns and 'DIV' in df2.columns):
            condition = (df2['PER'] > 0) & (df2['PER'] <= per_threshold) & \
                        (df2['PBR'] > 0) & (df2['PBR'] <= pbr_threshold) & \
                        (df2['DIV'] >= div_threshold)
            if not condition.any() and last_date is None:
                cur.execute(f"UPDATE ticker_status SET status = 'completed' WHERE ticker = '{ticker}'")
                conn.commit()
                print(f'{ticker} no.3 pass condition')
                time.sleep(5.1)
                continue  # Skip if the ticker does not meet the fundamental criteria
        else:
            cur.execute(f"UPDATE ticker_status SET status = 'completed' WHERE ticker = '{ticker}'")
            conn.commit()
            print(f'{ticker} no.4 pass condition')
            continue  # Skip if necessary columns are missing

        time.sleep(5.2)
        df1 = get_market_ohlcv_with_fallback(start_date, end_date, ticker)
        if df1.empty or '고가' not in df1.columns:
            cur.execute(f"UPDATE ticker_status SET status = 'completed' WHERE ticker = '{ticker}'")
            conn.commit()
            print(f'{ticker} no.5 pass condition')
            continue  # Skip if no OHLCV data is found

        time.sleep(5)
        df3 = stock.get_market_cap(start_date, end_date, ticker)
        df1.reset_index(inplace=True)
        df2.reset_index(inplace=True)
        df3.reset_index(inplace=True)
        df1.rename(columns={'날짜': 'date', '시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '거래량': 'volume'}, inplace=True)
        df2.rename(columns={'날짜': 'date', 'PER': 'PER', 'PBR': 'PBR', 'DIV': 'dividend', 'BPS': 'BPS', 'EPS': 'EPS', 'DPS': 'DPS'}, inplace=True)
        df3.rename(columns={'날짜': 'date', '시가총액': 'market_cap', '상장주식수': 'shares_outstanding', '거래대금': 'value', '거래량': 'volume'}, inplace=True)

        merged_df = pd.merge(df1, df2, on='date', how='outer', suffixes=('', '_duplicate'))
        merged_df = pd.merge(merged_df, df3, on='date', how='outer', suffixes=('', '_duplicate'))
        for column in merged_df.columns:
            if 'duplicate' in column:
                base_column = column.replace('_duplicate', '')
                if base_column in merged_df.columns:
                    merged_df.drop(column, axis=1, inplace=True)

        merged_df = merged_df[['date', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'shares_outstanding', 'PER', 'PBR', 'dividend', 'BPS', 'EPS', 'DPS']]
        merged_df = merged_df.replace({np.nan: None})
        name = stock.get_market_ticker_name(ticker)
        merged_df['ticker'] = ticker
        merged_df['name'] = name
        merged_df.set_index('date', inplace=True)

        if last_date:
            cur.execute(f"SELECT * FROM stock_data WHERE ticker = '{ticker}' AND date <= '{last_date}'")
            existing_data = pd.DataFrame(cur.fetchall(), columns=['ticker', 'name', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'shares_outstanding', 'PER', 'PBR', 'dividend', 'BPS', 'EPS', 'DPS','normalized_value'])
            existing_data['date'] = pd.to_datetime(existing_data['date'])
            existing_data.set_index('date', inplace=True)
            combined_df = pd.concat([existing_data, merged_df], axis=0, join='outer')
        else:
            combined_df = merged_df

        combined_df['normalized_value'] = (combined_df['close'] - combined_df['close'].rolling(window=252*5, min_periods=252).min()) / \
                                          (combined_df['close'].rolling(window=252*5, min_periods=252).max() - combined_df['close'].rolling(window=252*5, min_periods=252).min()) * 100
        combined_df.reset_index(inplace=True)
        combined_df = combined_df.replace({np.nan: None})

        time.sleep(0.4)
        for _, row in combined_df.iterrows():
            sql = '''
            INSERT INTO stock_data (ticker, name, date, open, high, low, close, volume, value, market_cap, shares_outstanding, PER, PBR, dividend, BPS, EPS, DPS, normalized_value)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open=VALUES(open),
                high=VALUES(high),
                low=VALUES(low),
                close=VALUES(close),
                volume=VALUES(volume),
                value=VALUES(value),
                market_cap=VALUES(market_cap),
                shares_outstanding=VALUES(shares_outstanding),
                PER=VALUES(PER),
                PBR=VALUES(PBR),
                dividend=VALUES(dividend),
                BPS=VALUES(BPS),
                EPS=VALUES(EPS),
                DPS=VALUES(DPS),
                normalized_value=VALUES(normalized_value)
            '''
            cur.execute(sql, (
                row['ticker'],
                row['name'],
                row['date'].strftime('%Y-%m-%d') if row['date'] else None,
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                int(row['volume']) if row['volume'] is not None else None,
                int(row['value']) if row['value'] is not None else None,
                int(row['market_cap']) if row['market_cap'] is not None else None,
                int(row['shares_outstanding']) if row['shares_outstanding'] is not None else None,
                row['PER'],
                row['PBR'],
                row['dividend'],
                row['BPS'],
                row['EPS'],
                row['DPS'],
                row['normalized_value']
            ))

        conn.commit()
        last_recorded_date = pd.to_datetime(combined_df['date']).max()
        if (current_date - last_recorded_date).days > 60:
            cur.execute(f"UPDATE ticker_status SET status = 'completed' WHERE ticker = '{ticker}'")
            conn.commit()
        print(f"{ticker} data successfully saved to the database")
        time.sleep(5)

    cur.close()
    conn.close()
    print("All data has been successfully saved.")
