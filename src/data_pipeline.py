"""
data_pipeline.py

This module contains the functions for the data pipeline for the Magic Split Strategy.
"""

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/data_pipeline.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from .db_setup import get_db_connection, create_tables
from .ticker_collector import collect_tickers
from .stock_data_collector import collect_stock_data
from .etf_data_collector import collect_etf_data

if __name__ == "__main__":
    conn = get_db_connection()
    create_tables(conn)
    conn.close()
    
    #collect_tickers()
    collect_stock_data()
    #collect_etf_data()
