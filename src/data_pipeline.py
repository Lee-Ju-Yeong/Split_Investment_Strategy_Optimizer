# main.py
from db_setup import get_db_connection, create_tables
from collect_tickers import collect_tickers
from stock_data_collector import collect_stock_data
from collect_etf_data import collect_etf_data

if __name__ == "__main__":
    conn = get_db_connection()
    create_tables(conn)
    conn.close()
    
    collect_tickers()
    collect_stock_data()
    collect_etf_data()
