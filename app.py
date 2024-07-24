
pip install -r requirements.txt


from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import configparser
import random
import numpy as np
from MagicSplit_Backtesting_Optimizer import (
    get_stock_codes, load_stock_data_from_mysql, calculate_additional_buy_drop_rate,
    calculate_sell_profit_rate, initial_buy_sell, additional_buy, additional_sell,
    get_trading_dates_from_db, portfolio_backtesting, calculate_mdd, plot_backtesting_results,
)

app = Flask(__name__)
CORS(app)

config = configparser.ConfigParser()
config.read('config.ini')

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
    per_threshold = float(data['per-threshold'])
    pbr_threshold = float(data['pbr-threshold'])
    div_threshold = float(data['dividend-threshold'])
    buy_threshold = int(data['buy-threshold'])
    start_date = data['start-date']
    end_date = data['end-date']
    consider_delisting = bool(data.get('consider-delisting', False))
    max_stocks = int(data['portfolio-size'])

    seed = 101
    results_folder = "results_of_single_test"

    positions_dict, total_portfolio_value, portfolio_values_over_time, capital_over_time, buy_signals, sell_signals, all_trading_dates, cagr = portfolio_backtesting(seed,
        initial_capital, num_splits, investment_ratio, buy_threshold, start_date, end_date, db_params, per_threshold, pbr_threshold, div_threshold, 0.005, consider_delisting, max_stocks, results_folder, save_files=False
    )
    mdd = calculate_mdd(portfolio_values_over_time)

    result = {
        "total_portfolio_value": total_portfolio_value,
        "cagr": cagr,
        "mdd": mdd,
        "portfolio_values_over_time": portfolio_values_over_time.tolist(),
        "capital_over_time": capital_over_time.tolist(),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "all_trading_dates": [date.strftime('%Y-%m-%d') for date in all_trading_dates]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
