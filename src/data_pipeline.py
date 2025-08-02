"""
data_pipeline.py

This module contains the functions for the data pipeline for the Magic Split Strategy.
"""

from db_setup import get_db_connection, create_tables
from ticker_collector import collect_tickers
from stock_data_collector import collect_stock_data
from etf_data_collector import collect_etf_data

if __name__ == "__main__":
    conn = get_db_connection()
    create_tables(conn)
    conn.close()
    
    #collect_tickers()
    collect_stock_data()
    #collect_etf_data()
