# collect_etf_data.py

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from db_setup import get_db_connection

def collect_etf_data():
    conn = get_db_connection()  # Establish a connection to the database
    cur = conn.cursor()  # Create a cursor object to interact with the database

    # Set up Chrome options for headless browsing
    options = webdriver.ChromeOptions()
    options.add_argument('headless')  # Run browser in headless mode
    options.add_argument('disable-gpu')  # Disable GPU usage
    options.add_argument('lang=ko_KR')  # Set language to Korean
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    url = 'https://finance.naver.com/sise/etf.naver'
    driver.get(url)  # Navigate to the ETF page on Naver Finance

    etf_data = []
    rows = driver.find_elements(By.CSS_SELECTOR, '#etfItemTable > tr')  # Locate ETF table rows
    for row in rows:
        name_col = row.find_elements(By.CSS_SELECTOR, 'td.ctg a')  # Locate name column
        if name_col:
            name = name_col[0].text.strip()  # Extract ETF name
            ticker = name_col[0].get_attribute('href').split('=')[-1]  # Extract ETF ticker
            etf_data.append([name, ticker])  # Add name and ticker to data list

    driver.quit()  # Close the browser

    df = pd.DataFrame(etf_data, columns=['종목명', '티커'])  # Create DataFrame from ETF data
    excel_filename = 'etf_list.xlsx'
    df.to_excel(excel_filename, index=False)  # Save DataFrame to Excel file
    print(f"Excel 파일로 저장되었습니다: {excel_filename}")

    # Create the etf_list table if it doesn't exist
    cur.execute('''
    CREATE TABLE IF NOT EXISTS etf_list (
        name VARCHAR(100),
        ticker VARCHAR(20),
        PRIMARY KEY (ticker)
    )
    ''')
    conn.commit()

    # Insert ETF data into the etf_list table
    for index, row in df.iterrows():
        cur.execute('''
        INSERT INTO etf_list (name, ticker) VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE name=VALUES(name)
        ''', (row['종목명'], row['티커']))
    
    conn.commit()
    cur.close()
    conn.close()
    print("MySQL 데이터베이스에 저장되었습니다.")
