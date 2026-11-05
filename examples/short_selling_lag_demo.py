"""가상 데이터로 기준일과 반영일을 비교한다. 저장소 루트에서 -m으로 실행."""

import numpy as np
import pandas as pd

from src.pipeline.daily_stock_tier_batch import (
    DEFAULT_SHORT_SELLING_PUBLICATION_LAG_TRADING_DAYS,
    build_daily_stock_tier_frame,
)


# 설명용 거래일을 명시한다. 실제 거래소 달력을 조회하거나 추정하지 않는다.
TRADING_DATES = (
    "2024-01-02", "2024-01-03", "2024-01-04",
    "2024-01-05", "2024-01-08", "2024-01-09",
)


def make_price_frame():
    return pd.DataFrame({
        "stock_code": "SAMPLE",
        "date": pd.to_datetime(TRADING_DATES),
        "close_price": 100.0,
        "volume": 20_000_000,
        "high_price": 110.0,
        "low_price": 90.0,
        "adj_close": 100.0,
        "adj_high": 110.0,
        "adj_low": 90.0,
    })


def make_short_balance_frame():
    return pd.DataFrame([{
        "stock_code": "SAMPLE",
        "date": pd.Timestamp(TRADING_DATES[0]),
        "short_balance_value": 30_000_000,
        "market_cap": 1_000_000_000,
    }])


def calculate_ratio(lag_days):
    result = build_daily_stock_tier_frame(
        price_df=make_price_frame(),
        short_balance_df=make_short_balance_frame(),
        short_selling_publication_lag_trading_days=lag_days,
    )
    # 여섯 행으로 장기 종목 등급을 설명할 수 없으므로 날짜와 비율만 보여준다.
    return result.set_index("date")["sbv_ratio"]


def check_expected_values(actual, expected_values):
    expected = pd.Series(
        expected_values,
        index=pd.Index(TRADING_DATES, name="date"),
        name="sbv_ratio",
    )
    pd.testing.assert_series_equal(actual, expected)


def main():
    lag_days = DEFAULT_SHORT_SELLING_PUBLICATION_LAG_TRADING_DAYS
    if lag_days != 3:
        raise RuntimeError("기본 반영 정책이 바뀌었습니다. 예제의 설명과 기대값을 검토하세요.")
    same_day = calculate_ratio(0)
    delayed = calculate_ratio(lag_days)
    check_expected_values(same_day, [0.03, np.nan, np.nan, np.nan, np.nan, np.nan])
    check_expected_values(delayed, [np.nan, np.nan, np.nan, 0.03, np.nan, np.nan])
    table = pd.DataFrame({"당일 반영(비교용)": same_day, "3거래일 뒤 반영": delayed})
    print("가상 종목 SAMPLE / 공매도 잔고금액 ÷ 시가총액 = 0.03 (3%)")
    print(table.to_string(na_rep="미반영", float_format=lambda value: f"{value:.2f}"))
    print("확인 완료: 2024-01-02 기준 값은 세 거래일 뒤인 2024-01-05에 반영됩니다.")
    print("데이터 반영 시점만 보여주는 예제입니다. 거래·수익률은 계산하지 않습니다.")


if __name__ == "__main__":
    main()
