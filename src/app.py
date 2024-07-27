from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import datetime
import random
import configparser
from concurrent.futures import ThreadPoolExecutor
from single_backtest import single_backtesting
from backtest_strategy import calculate_mdd

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# 설정 파일 읽기
config = configparser.ConfigParser()
config.read('config.ini')

# 데이터베이스 연결 정보 설정
db_params = {
    'host': config['mysql']['host'],
    'user': config['mysql']['user'],
    'password': config['mysql']['password'],
    'database': config['mysql']['database'],
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    data = request.json
    initial_capital = int(data['initial-capital'])
    num_splits = int(data['num-splits'])
    investment_ratio = float(data['investment-ratio'])
    pbr_threshold = float(data['pbr-threshold'])
    per_threshold = float(data['per-threshold'])
    div_threshold = float(data['dividend-threshold'])
    buy_threshold = int(data['buy-threshold'])
    start_date = data['start-date']
    end_date = data['end-date']
    consider_delisting = data['consider-delisting'] == 'on'
    max_stocks = int(data['portfolio-size'])
    seed = random.randint(0, 10000)
    results_folder = "results_of_single_test"

    # 백테스팅 실행
    positions_dict, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr,mdd= single_backtesting(
        seed, num_splits, buy_threshold, investment_ratio, start_date, end_date,
        per_threshold, pbr_threshold, div_threshold, 0.005, consider_delisting,
        max_stocks, save_files=True
    )

    # 결과 파일 경로
    current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    excel_file_name = f'trading_history_{num_splits}_{max_stocks}_{buy_threshold}_{current_time_str}.xlsx'
    plot_file_name = f'trading_history_{num_splits}_{max_stocks}_{buy_threshold}_{current_time_str}.png'
    excel_file_path = os.path.join(results_folder, excel_file_name)
    plot_file_path = os.path.join(results_folder, plot_file_name)

    # 결과 반환
    response = {
        'total_portfolio_value': total_portfolio_value,
        'cagr': cagr,
        'mdd': mdd,
        'all_trading_dates': all_trading_dates,
        'portfolio_values_over_time': portfolio_values_over_time.tolist(),
        'capital_over_time': capital_over_time.tolist(),
        'excel_file_path': excel_file_name,
        'plot_file_path': plot_file_name
    }
    return jsonify(response)


@app.route('/results_of_single_test/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('results_of_single_test', filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
