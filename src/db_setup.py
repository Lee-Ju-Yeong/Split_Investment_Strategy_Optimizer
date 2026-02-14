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
    def ensure_index(table_name, index_name, column_expr):
        cur.execute(
            '''
            SELECT 1
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND INDEX_NAME = %s
            LIMIT 1
            ''',
            (table_name, index_name)
        )
        if cur.fetchone() is None:
            cur.execute(f'CREATE INDEX {index_name} ON {table_name} ({column_expr})')
    def ensure_column(table_name, column_name, column_definition):
        cur.execute(
            '''
            SELECT 1
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND COLUMN_NAME = %s
            LIMIT 1
            ''',
            (table_name, column_name)
        )
        if cur.fetchone() is None:
            cur.execute(
                f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}'
            )
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
        last_updated DATETIME,
        is_delisted BOOLEAN DEFAULT FALSE
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS ticker_status (
        ticker VARCHAR(10) PRIMARY KEY,
        status VARCHAR(20)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS CompanyInfo (
        stock_code VARCHAR(20) PRIMARY KEY,
        company_name VARCHAR(255),
        market_type VARCHAR(50),
        last_updated DATETIME
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS WeeklyFilteredStocks (
        filter_date DATE,
        stock_code VARCHAR(20),
        company_name VARCHAR(255),
        PRIMARY KEY (filter_date, stock_code)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS DailyStockPrice (
        stock_code VARCHAR(20),
        date DATE,
        open_price DECIMAL(20, 5),
        high_price DECIMAL(20, 5),
        low_price DECIMAL(20, 5),
        close_price DECIMAL(20, 5),
        adj_close DECIMAL(20, 5) NULL,
        adj_ratio DECIMAL(20, 10) NULL,
        volume BIGINT,
        PRIMARY KEY (stock_code, date)
    )
    ''')
    ensure_column('DailyStockPrice', 'adj_close', 'DECIMAL(20, 5) NULL')
    ensure_column('DailyStockPrice', 'adj_ratio', 'DECIMAL(20, 10) NULL')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS CalculatedIndicators (
        stock_code VARCHAR(6),
        date DATE,
        ma_5 FLOAT NULL,
        ma_20 FLOAT NULL,
        atr_14_ratio FLOAT NULL,
        price_vs_5y_low_pct FLOAT NULL,
        price_vs_10y_low_pct FLOAT NULL,
        PRIMARY KEY (stock_code, date)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS FinancialData (
        stock_code VARCHAR(20),
        date DATE,
        per FLOAT NULL,
        pbr FLOAT NULL,
        eps FLOAT NULL,
        bps FLOAT NULL,
        dps FLOAT NULL,
        div_yield FLOAT NULL,
        roe FLOAT NULL,
        source VARCHAR(50) DEFAULT 'pykrx',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (stock_code, date)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS InvestorTradingTrend (
        stock_code VARCHAR(20),
        date DATE,
        individual_net_buy BIGINT DEFAULT 0,
        foreigner_net_buy BIGINT DEFAULT 0,
        institution_net_buy BIGINT DEFAULT 0,
        total_net_buy BIGINT DEFAULT 0,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (stock_code, date)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS MarketCapDaily (
        stock_code VARCHAR(20),
        date DATE,
        market_cap BIGINT NULL,
        shares_outstanding BIGINT NULL,
        source VARCHAR(50) DEFAULT 'pykrx',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (stock_code, date)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS ShortSellingDaily (
        stock_code VARCHAR(20),
        date DATE,
        short_volume BIGINT NULL,
        short_value BIGINT NULL,
        short_balance BIGINT NULL,
        short_balance_value BIGINT NULL,
        source VARCHAR(50) DEFAULT 'pykrx',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (stock_code, date)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS DailyStockTier (
        date DATE,
        stock_code VARCHAR(20),
        tier TINYINT NOT NULL,
        reason VARCHAR(255),
        liquidity_20d_avg_value BIGINT NULL,
        computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (date, stock_code)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS TickerUniverseSnapshot (
        snapshot_date DATE,
        stock_code VARCHAR(20),
        market_type VARCHAR(20),
        company_name VARCHAR(255) NULL,
        source VARCHAR(50) DEFAULT 'pykrx',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (snapshot_date, stock_code)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS TickerUniverseHistory (
        stock_code VARCHAR(20) PRIMARY KEY,
        listed_date DATE NOT NULL,
        last_seen_date DATE NOT NULL,
        delisted_date DATE NULL,
        latest_market_type VARCHAR(20) NULL,
        latest_company_name VARCHAR(255) NULL,
        source VARCHAR(50) DEFAULT 'snapshot_agg',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS CorporateMajorChanges (
        stock_code VARCHAR(20),
        change_date DATE,
        prev_company_name VARCHAR(255),
        new_company_name VARCHAR(255),
        prev_sector VARCHAR(255),
        new_sector VARCHAR(255),
        prev_par_value DECIMAL(20, 5),
        new_par_value DECIMAL(20, 5),
        prev_ceo TEXT,
        new_ceo TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (stock_code, change_date)
    )
    ''')
    ensure_index('FinancialData', 'idx_financial_date_stock', 'date, stock_code')
    ensure_index('InvestorTradingTrend', 'idx_investor_date_stock', 'date, stock_code')
    ensure_index('InvestorTradingTrend', 'idx_investor_date_flow', 'date, foreigner_net_buy, institution_net_buy')
    ensure_index('MarketCapDaily', 'idx_mcap_date_stock', 'date, stock_code')
    ensure_index('ShortSellingDaily', 'idx_short_date_stock', 'date, stock_code')
    ensure_index('DailyStockTier', 'idx_tier_stock_date', 'stock_code, date')
    ensure_index('DailyStockTier', 'idx_tier_date_tier_stock', 'date, tier, stock_code')
    ensure_index('TickerUniverseSnapshot', 'idx_tus_stock_date', 'stock_code, snapshot_date')
    ensure_index('TickerUniverseSnapshot', 'idx_tus_date_market_stock', 'snapshot_date, market_type, stock_code')
    ensure_index('TickerUniverseHistory', 'idx_tuh_listed_date', 'listed_date')
    ensure_index('TickerUniverseHistory', 'idx_tuh_last_seen_date', 'last_seen_date')
    ensure_index('TickerUniverseHistory', 'idx_tuh_delisted_date', 'delisted_date')
    conn.commit()
    cur.close()
