# db_setup.py
import configparser
import pymysql

def get_db_connection():
    config = configparser.ConfigParser()
    config.read('config.ini')

    user= config['mysql']['user']
    password = config['mysql']['password']
    host = config['mysql']['host']
    database = config['mysql']['database']

    conn = pymysql.connect(host=host, user=user, password=password, db=database, charset='utf8')
    return conn

def create_tables(conn):
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        ticker VARCHAR(10),
        name VARCHAR(100),
        date DATE,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume BIGINT,
        value BIGINT,
        market_cap BIGINT,
        shares_outstanding BIGINT,
        PER FLOAT,
        PBR FLOAT,
        dividend FLOAT,
        BPS FLOAT,
        EPS FLOAT,
        DPS FLOAT,
        normalized_value FLOAT,
        PRIMARY KEY (ticker, date)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS ticker_list (
        ticker VARCHAR(10) PRIMARY KEY,
        market VARCHAR(10),
        name VARCHAR(100),
        last_updated DATE,
        is_delisted BOOLEAN DEFAULT FALSE
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS ticker_status (
        ticker VARCHAR(10) PRIMARY KEY,
        status VARCHAR(20)
    )
    ''')
    conn.commit()
    cur.close()
