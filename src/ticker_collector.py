# collect_tickers.py
import datetime
import time
from pykrx import stock
from db_setup import get_db_connection

def collect_tickers():
    conn = get_db_connection() # Establish a connection to the database
    cur = conn.cursor() # Create a cursor object to interact with the database
    
    all_tickers = set()
    start_date = '19950102'
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    current_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    
    # Collect ticker lists for all dates from start_date to end_date
    while current_date.strftime('%Y%m%d') <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        try:
            tickers_kosdaq = stock.get_market_ticker_list(date_str, market='KOSDAQ')
            tickers_kospi = stock.get_market_ticker_list(date_str, market='KOSPI')
            all_tickers.update(tickers_kosdaq)
            all_tickers.update(tickers_kospi)
            print(len(all_tickers), f'{date_str}')
            time.sleep(0.8) # To avoid too many requests in a short period
        except Exception as e:
            print(f"Error on date {date_str}: {e}")
        current_date += datetime.timedelta(days=180)
    
    # Save the ticker list to MySQL database
    for ticker in all_tickers:
        market = 'KOSDAQ' if ticker in tickers_kosdaq else 'KOSPI'
        name = stock.get_market_ticker_name(ticker)
        cur.execute('''
        INSERT INTO ticker_list (ticker, market, name) VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE market=VALUES(market), name=VALUES(name)
        ''', (ticker, market, name))
        conn.commit()

    cur.close()
    conn.close()
    print("모든 티커 목록이 성공적으로 저장되었습니다.")
