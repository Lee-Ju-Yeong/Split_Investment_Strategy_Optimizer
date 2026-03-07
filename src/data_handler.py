import pandas as pd
import warnings
import logging
import json
import hashlib
import os
import tempfile
from mysql.connector import pooling
from functools import lru_cache
from datetime import timedelta

try:
    from .price_policy import (
        is_adjusted_price_basis,
        normalize_iso_date,
        normalize_price_basis,
        resolve_price_policy,
        validate_backtest_window_for_price_policy,
    )
    from .universe_policy import (
        is_survivor_optimistic_mode,
        normalize_universe_mode,
        resolve_universe_mode,
    )
except ImportError:  # pragma: no cover
    from price_policy import (  # type: ignore
        is_adjusted_price_basis,
        normalize_iso_date,
        normalize_price_basis,
        resolve_price_policy,
        validate_backtest_window_for_price_policy,
    )
    from universe_policy import (  # type: ignore
        is_survivor_optimistic_mode,
        normalize_universe_mode,
        resolve_universe_mode,
    )

# CompanyInfo 캐시를 직접 관리
STOCK_CODE_TO_NAME_CACHE = {}
logger = logging.getLogger(__name__)


class PitRuntimeError(ValueError):
    """Structured PIT/runtime failure used by #67 artifacts and run manifests."""

    def __init__(self, code, message, *, stage, details=None):
        super().__init__(message)
        self.pit_code = str(code)
        self.pit_stage = str(stage)
        self.pit_details = dict(details or {})


class PointInTimeViolation(PitRuntimeError):
    """Raised when a data row later than the requested as-of date is returned."""

    def __init__(self, row_date, as_of_date):
        row_ts = pd.to_datetime(row_date)
        as_of_ts = pd.to_datetime(as_of_date)
        super().__init__(
            "pit_row_newer_than_as_of",
            f"PIT violation: row_date({row_ts.date()}) is newer than as_of_date({as_of_ts.date()}).",
            stage="pit_as_of_validation",
            details={
                "row_date": row_ts.strftime("%Y-%m-%d"),
                "as_of_date": as_of_ts.strftime("%Y-%m-%d"),
            },
        )


def _normalize_failure_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dict):
        return {
            str(key): _normalize_failure_value(inner)
            for key, inner in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalize_failure_value(item) for item in value]
    return str(value)


def build_pit_failure_record(
    error,
    *,
    trade_date=None,
    signal_date=None,
    fallback_code=None,
    fallback_stage=None,
):
    code = getattr(error, "pit_code", None) or fallback_code
    stage = getattr(error, "pit_stage", None) or fallback_stage
    if code is None and stage is None:
        return None

    details = _normalize_failure_value(getattr(error, "pit_details", None) or {})
    record = {
        "code": str(code) if code is not None else None,
        "stage": str(stage) if stage is not None else None,
        "exception_type": type(error).__name__,
        "message": str(error)[:240],
        "details": details,
    }
    if trade_date is not None:
        record["trade_date"] = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
    if signal_date is not None:
        record["signal_date"] = pd.to_datetime(signal_date).strftime("%Y-%m-%d")
    return record


class DataHandler:
    def __init__(
        self,
        db_config,
        *,
        price_basis=None,
        adjusted_price_gate_start_date=None,
        universe_mode=None,
        strategy_params=None,
    ):
        self.db_config = db_config
        resolved_price_basis, resolved_gate_start_date = resolve_price_policy(
            strategy_params=strategy_params
        )
        if price_basis is not None:
            resolved_price_basis = normalize_price_basis(price_basis)
        if adjusted_price_gate_start_date is not None:
            resolved_gate_start_date = normalize_iso_date(
                adjusted_price_gate_start_date,
                field_name="adjusted_price_gate_start_date",
            )
        resolved_universe_mode = resolve_universe_mode(
            strategy_params=strategy_params,
            universe_mode=universe_mode,
        )
        self.price_basis = resolved_price_basis
        self.adjusted_price_gate_start_date = resolved_gate_start_date
        self.use_adjusted_prices = is_adjusted_price_basis(self.price_basis)
        self.universe_mode = normalize_universe_mode(resolved_universe_mode)
        self._eventual_delisted_codes_cache = None
        self._frozen_tier_candidate_manifest = None
        self._frozen_tier_candidate_manifest_key = None
        self._lazy_tier_candidate_cache = None
        self._lazy_tier_candidate_cache_key = None
        self._strict_frozen_candidate_manifest_required = False
        self.last_frozen_candidate_manifest_summary = {}
        self.frozen_candidate_manifest_mode = self._resolve_frozen_candidate_manifest_mode(
            strategy_params
        )
        self.frozen_candidate_manifest_path = self._resolve_frozen_candidate_manifest_path(
            strategy_params
        )
        self.frozen_candidate_manifest_expected_sha256 = (
            self._resolve_frozen_candidate_manifest_expected_sha256(strategy_params)
        )
        try:
            self.connection_pool = pooling.MySQLConnectionPool(pool_name="data_pool",
                                                               pool_size=10,
                                                               use_pure=True,
                                                               **self.db_config)
            self.has_stored_adj_ohlc = self._detect_stored_adj_ohlc_columns()
            self.has_tier_flow5_mcap = self._detect_tier_flow5_mcap_column()
            self._load_company_info_cache()
        except Exception as e:
            print(f"DB 연결 풀 생성 또는 캐시 로딩 실패: {e}")
            raise

    def _detect_stored_adj_ohlc_columns(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'DailyStockPrice'
                      AND COLUMN_NAME IN ('adj_open', 'adj_high', 'adj_low')
                    """
                )
                row = cur.fetchone()
            count = int((row[0] if row else 0) or 0)
            return count == 3
        except Exception as exc:
            print(
                "[DataHandler] warning: failed to detect stored adjusted OHLC columns "
                f"({type(exc).__name__}). fallback=formula"
            )
            return False
        finally:
            conn.close()

    @staticmethod
    def _resolve_frozen_candidate_manifest_mode(strategy_params):
        raw_mode = (strategy_params or {}).get("frozen_candidate_manifest_mode", "off")
        mode = str(raw_mode).strip().lower()
        if mode not in {"off", "record_strict", "replay_strict"}:
            raise PitRuntimeError(
                "frozen_manifest_mode_invalid",
                "Unsupported frozen_candidate_manifest_mode="
                f"{raw_mode!r}. supported=[off, record_strict, replay_strict]",
                stage="strict_frozen_manifest_config",
                details={"frozen_candidate_manifest_mode": raw_mode},
            )
        return mode

    @staticmethod
    def _resolve_frozen_candidate_manifest_path(strategy_params):
        raw_path = (strategy_params or {}).get("frozen_candidate_manifest_path")
        if raw_path is None:
            return None
        path = str(raw_path).strip()
        return path or None

    @staticmethod
    def _resolve_frozen_candidate_manifest_expected_sha256(strategy_params):
        raw_sha = (strategy_params or {}).get("frozen_candidate_manifest_expected_sha256")
        if raw_sha is None:
            return None
        sha = str(raw_sha).strip().lower()
        if not sha:
            return None
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise PitRuntimeError(
                "frozen_manifest_expected_sha_invalid",
                "frozen_candidate_manifest_expected_sha256 must be a 64-char hex digest",
                stage="strict_frozen_manifest_config",
                details={"frozen_candidate_manifest_expected_sha256": str(raw_sha)},
            )
        return sha

    def _detect_tier_flow5_mcap_column(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'DailyStockTier'
                      AND COLUMN_NAME = 'flow5_mcap'
                    """
                )
                row = cur.fetchone()
            count = int((row[0] if row else 0) or 0)
            return count == 1
        except Exception as exc:
            print(
                "[DataHandler] warning: failed to detect DailyStockTier.flow5_mcap column "
                f"({type(exc).__name__}). fallback=NULL"
            )
            return False
        finally:
            conn.close()

    def _load_company_info_cache(self):
        """DB의 CompanyInfo 테이블에서 데이터를 읽어와 인메모리 캐시를 채웁니다."""
        global STOCK_CODE_TO_NAME_CACHE
        print("CompanyInfo 캐시 로딩 중...")
        conn = self.get_connection()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # pymysql은 read_sql_table을 지원하지 않으므로 read_sql_query 사용
                df = pd.read_sql_query('SELECT stock_code, company_name FROM CompanyInfo', conn)
            if not df.empty:
                STOCK_CODE_TO_NAME_CACHE = pd.Series(df.company_name.values, index=df.stock_code).to_dict()
                print(f"CompanyInfo 캐시 로드 완료: {len(STOCK_CODE_TO_NAME_CACHE)}개 종목.")
            else:
                print("경고: CompanyInfo 테이블이 비어있습니다. 종목명이 N/A로 표시될 수 있습니다.")
                STOCK_CODE_TO_NAME_CACHE = {}
        except Exception as e:
            print(f"CompanyInfo 캐시 로드 중 오류: {e}")
            STOCK_CODE_TO_NAME_CACHE = {}
        finally:
            conn.close()
    
    def get_name_from_ticker(self, ticker_code):
        """캐시에서 종목코드로 종목명을 조회합니다."""
        return STOCK_CODE_TO_NAME_CACHE.get(ticker_code)

    def get_connection(self):
        return self.connection_pool.get_connection()

    @staticmethod
    def assert_point_in_time(row_date, as_of_date):
        row_ts = pd.to_datetime(row_date)
        as_of_ts = pd.to_datetime(as_of_date)
        if row_ts > as_of_ts:
            raise PointInTimeViolation(row_date=row_ts, as_of_date=as_of_ts)

    @staticmethod
    def get_previous_trading_date(trading_dates, current_day_idx):
        if current_day_idx is None or current_day_idx <= 0:
            return None
        return pd.to_datetime(trading_dates[current_day_idx - 1])

    @staticmethod
    def _normalize_date_key(value):
        return pd.to_datetime(value).strftime('%Y-%m-%d')

    def _build_tier_manifest_key(
        self,
        min_liquidity_20d_avg_value,
        min_tier12_coverage_ratio,
    ):
        liquidity_gate = None
        if min_liquidity_20d_avg_value is not None:
            liquidity_gate = int(min_liquidity_20d_avg_value)

        coverage_gate = None
        if min_tier12_coverage_ratio is not None:
            coverage_gate = float(min_tier12_coverage_ratio)

        return self.universe_mode, liquidity_gate, coverage_gate

    def clear_frozen_tier_candidate_manifest(self):
        self._frozen_tier_candidate_manifest = None
        self._frozen_tier_candidate_manifest_key = None
        self._strict_frozen_candidate_manifest_required = False

    def clear_lazy_tier_candidate_cache(self):
        self._lazy_tier_candidate_cache = None
        self._lazy_tier_candidate_cache_key = None

    def clear_load_stock_data_cache(self):
        self._load_stock_data_cached.cache_clear()

    def _validate_window(self, start_date, end_date):
        validate_backtest_window_for_price_policy(
            start_date=start_date,
            end_date=end_date,
            price_basis=self.price_basis,
            adjusted_price_gate_start_date=self.adjusted_price_gate_start_date,
        )

    def get_trading_dates(self, start_date, end_date):
        self._validate_window(start_date, end_date)
        conn = self.get_connection()
        try:
            query = "SELECT DISTINCT date FROM DailyStockPrice WHERE date BETWEEN %s AND %s ORDER BY date"
            start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # pd.read_sql 사용 시 날짜 파싱이 더 안정적
                df = pd.read_sql(query, conn, params=(start_date_str, end_date_str))
            return pd.to_datetime(df['date']).tolist()
        finally:
            conn.close()

    def load_stock_data(self, ticker, start_date, end_date):
        self._validate_window(start_date, end_date)
        ticker_key = str(ticker)
        start_date_str = self._normalize_date_key(start_date)
        end_date_str = self._normalize_date_key(end_date)
        return self._load_stock_data_cached(
            ticker_key,
            start_date_str,
            end_date_str,
            self.universe_mode,
        )

    @lru_cache(maxsize=200)
    def _load_stock_data_cached(self, ticker, start_date_str, end_date_str, _universe_mode):
        conn = self.get_connection()
        # 지표 계산에 필요한 충분한 과거 데이터를 위해 시작 날짜 확장
        extended_start_date = pd.to_datetime(start_date_str) - timedelta(days=252*10 + 50)
        extended_start_date_str = extended_start_date.strftime('%Y-%m-%d')
        
        if self.use_adjusted_prices:
            if self.has_stored_adj_ohlc:
                price_select_sql = """
                    CASE
                        WHEN dsp.adj_ratio IS NULL THEN NULL
                        WHEN dsp.adj_open IS NULL THEN dsp.open_price * dsp.adj_ratio
                        WHEN ABS(dsp.adj_open - (dsp.open_price * dsp.adj_ratio)) > 1e-5
                            THEN dsp.open_price * dsp.adj_ratio
                        ELSE dsp.adj_open
                    END AS open_price,
                    CASE
                        WHEN dsp.adj_ratio IS NULL THEN NULL
                        WHEN dsp.adj_high IS NULL THEN dsp.high_price * dsp.adj_ratio
                        WHEN ABS(dsp.adj_high - (dsp.high_price * dsp.adj_ratio)) > 1e-5
                            THEN dsp.high_price * dsp.adj_ratio
                        ELSE dsp.adj_high
                    END AS high_price,
                    CASE
                        WHEN dsp.adj_ratio IS NULL THEN NULL
                        WHEN dsp.adj_low IS NULL THEN dsp.low_price * dsp.adj_ratio
                        WHEN ABS(dsp.adj_low - (dsp.low_price * dsp.adj_ratio)) > 1e-5
                            THEN dsp.low_price * dsp.adj_ratio
                        ELSE dsp.adj_low
                    END AS low_price,
                    dsp.adj_close AS close_price
                """
            else:
                price_select_sql = """
                    CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.open_price * dsp.adj_ratio ELSE NULL END AS open_price,
                    CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.high_price * dsp.adj_ratio ELSE NULL END AS high_price,
                    CASE WHEN dsp.adj_ratio IS NOT NULL THEN dsp.low_price * dsp.adj_ratio ELSE NULL END AS low_price,
                    dsp.adj_close AS close_price
                """
        else:
            price_select_sql = """
                dsp.open_price,
                dsp.high_price,
                dsp.low_price,
                dsp.close_price
            """

        query = f"""
            SELECT
                dsp.date,
                {price_select_sql},
                dsp.volume,
                ci.ma_5, ci.ma_20, ci.atr_14_ratio, ci.price_vs_5y_low_pct, ci.price_vs_10y_low_pct AS normalized_value,
                mcd.market_cap,
                dst.cheap_score,
                dst.cheap_score_confidence,
                {"dst.flow5_mcap" if self.has_tier_flow5_mcap else "NULL AS flow5_mcap"}
            FROM DailyStockPrice dsp
            LEFT JOIN CalculatedIndicators ci ON dsp.stock_code = ci.stock_code AND dsp.date = ci.date
            LEFT JOIN MarketCapDaily mcd ON dsp.stock_code = mcd.stock_code AND dsp.date = mcd.date
            LEFT JOIN DailyStockTier dst ON dsp.stock_code = dst.stock_code AND dsp.date = dst.date
            WHERE dsp.stock_code = %s AND dsp.date BETWEEN %s AND %s
            ORDER BY dsp.date ASC
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df = pd.read_sql(query, conn, params=(ticker, extended_start_date_str, end_date_str))
            if df.empty:
                return df

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            start_dt = pd.to_datetime(start_date_str)
            end_dt = pd.to_datetime(end_date_str)
            df_filtered = df.loc[start_dt:end_dt].copy()
            if df_filtered.empty:
                return df_filtered

            if self.use_adjusted_prices:
                missing_price_mask = df_filtered[
                    ["open_price", "high_price", "low_price", "close_price"]
                ].isna().any(axis=1)
                if missing_price_mask.any():
                    first_missing_date = pd.to_datetime(
                        df_filtered.loc[missing_price_mask, :].index[0]
                    ).date()
                    raise ValueError(
                        "Adjusted price mode found NULL adjusted OHLC values "
                        f"for ticker={ticker} at date={first_missing_date}. "
                        f"Backtest window must satisfy date >= {self.adjusted_price_gate_start_date}."
                    )

            return df_filtered
        finally:
            conn.close()


    def get_latest_price(self, date, ticker, start_date, end_date):
        data_row = self.get_stock_row_as_of(ticker, date, start_date, end_date)
        if data_row is None:
            return None
        try:
            return data_row['close_price']
        except KeyError:
            return None

    def get_stock_row_as_of(self, ticker, as_of_date, start_date, end_date):
        stock_data = self.load_stock_data(ticker, start_date, end_date)
        if stock_data is None or stock_data.empty:
            return None

        target_date = pd.to_datetime(as_of_date)
        row_index = stock_data.index.asof(target_date)
        if pd.isna(row_index):
            return None

        self.assert_point_in_time(row_index, target_date)
        data_row = stock_data.loc[row_index].copy()
        data_row.name = row_index
        return data_row

    def get_ohlc_data_on_date(self, date, ticker, start_date, end_date):
        stock_data = self.load_stock_data(ticker, start_date, end_date)
        if stock_data is None or stock_data.empty:
            return None

        target_date = pd.to_datetime(date)
        if target_date not in stock_data.index:
            return None

        data_row = stock_data.loc[target_date]
        if isinstance(data_row, pd.DataFrame):
            data_row = data_row.iloc[-1]
        data_row = data_row.copy()
        data_row.name = target_date
        return data_row

    def get_filtered_stock_codes(self, date):
        conn = self.get_connection()
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        query = """
            SELECT stock_code FROM WeeklyFilteredStocks
            WHERE filter_date = (SELECT MAX(filter_date) FROM WeeklyFilteredStocks WHERE filter_date < %s)
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df = pd.read_sql(query, conn, params=[date_str])
            return df['stock_code'].tolist()
        finally:
            conn.close()

    def get_stock_tier_as_of(self, ticker, as_of_date, *, conn=None):
        owns_conn = conn is None
        conn = conn or self.get_connection()
        as_of_date_str = pd.to_datetime(as_of_date).strftime('%Y-%m-%d')
        query = """
            SELECT date, stock_code, tier, reason, liquidity_20d_avg_value
            FROM DailyStockTier
            WHERE stock_code = %s AND date <= %s
            ORDER BY date DESC
            LIMIT 1
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df = pd.read_sql(query, conn, params=[ticker, as_of_date_str])
            if df.empty:
                return None
            row = df.iloc[0]
            self.assert_point_in_time(row["date"], as_of_date)
            return {
                "stock_code": row["stock_code"],
                "date": pd.to_datetime(row["date"]),
                "tier": int(row["tier"]),
                "reason": row.get("reason"),
                "liquidity_20d_avg_value": row.get("liquidity_20d_avg_value"),
            }
        finally:
            if owns_conn:
                conn.close()

    def get_tiers_as_of(self, as_of_date, tickers=None, allowed_tiers=None, *, conn=None):
        owns_conn = conn is None
        conn = conn or self.get_connection()
        as_of_date_str = pd.to_datetime(as_of_date).strftime('%Y-%m-%d')

        tickers = list(tickers) if tickers else []
        allowed_tiers = list(allowed_tiers) if allowed_tiers else []

        subquery_conditions = ["date <= %s"]
        subquery_params = [as_of_date_str]
        if tickers:
            ticker_placeholders = ", ".join(["%s"] * len(tickers))
            subquery_conditions.append(f"stock_code IN ({ticker_placeholders})")
            subquery_params.extend(tickers)

        subquery_where = " AND ".join(subquery_conditions)
        main_where = ""
        main_params = list(subquery_params)
        if allowed_tiers:
            tier_placeholders = ", ".join(["%s"] * len(allowed_tiers))
            main_where = f"WHERE t.tier IN ({tier_placeholders})"
            main_params.extend(allowed_tiers)

        query = f"""
            SELECT t.stock_code, t.date, t.tier, t.reason, t.liquidity_20d_avg_value
            FROM DailyStockTier t
            JOIN (
                SELECT stock_code, MAX(date) AS max_date
                FROM DailyStockTier
                WHERE {subquery_where}
                GROUP BY stock_code
            ) latest ON t.stock_code = latest.stock_code AND t.date = latest.max_date
            {main_where}
            ORDER BY t.stock_code
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df = pd.read_sql(query, conn, params=main_params)
            if df.empty:
                return {}

            for row_date in df["date"]:
                self.assert_point_in_time(row_date, as_of_date)

            result = {}
            for _, row in df.iterrows():
                result[row["stock_code"]] = {
                    "stock_code": row["stock_code"],
                    "date": pd.to_datetime(row["date"]),
                    "tier": int(row["tier"]),
                    "reason": row.get("reason"),
                    "liquidity_20d_avg_value": row.get("liquidity_20d_avg_value"),
                }
            return result
        finally:
            if owns_conn:
                conn.close()

    def _get_eventual_delisted_codes(self):
        if self._eventual_delisted_codes_cache is not None:
            return self._eventual_delisted_codes_cache

        conn = self.get_connection()
        try:
            query = """
                SELECT DISTINCT stock_code
                FROM TickerUniverseHistory
                WHERE delisted_date IS NOT NULL
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df = pd.read_sql(query, conn)
            if df.empty:
                self._eventual_delisted_codes_cache = set()
                return self._eventual_delisted_codes_cache

            codes = set(df["stock_code"].dropna().astype(str).tolist())
            self._eventual_delisted_codes_cache = codes
            return self._eventual_delisted_codes_cache
        finally:
            conn.close()

    def _apply_universe_mode_filter(self, codes):
        if not codes:
            return []
        if not is_survivor_optimistic_mode(self.universe_mode):
            return list(codes)

        eventual_delisted = self._get_eventual_delisted_codes()
        if not eventual_delisted:
            return list(codes)
        return [code for code in codes if str(code) not in eventual_delisted]

    def get_pit_universe_codes_as_of(self, as_of_date, *, conn=None):
        """
        Issue #67 Phase2: PIT 유니버스 기본 조회 경로.
        1) TickerUniverseSnapshot latest(as_of)
        2) empty면 TickerUniverseHistory active(as_of) fallback
        Returns:
            (codes, source)
        """
        as_of_ts = pd.to_datetime(as_of_date)
        as_of_date_str = as_of_ts.strftime("%Y-%m-%d")
        owns_conn = conn is None
        conn = conn or self.get_connection()
        try:
            snapshot_query = """
                SELECT stock_code
                FROM TickerUniverseSnapshot
                WHERE snapshot_date = (
                    SELECT MAX(snapshot_date)
                    FROM TickerUniverseSnapshot
                    WHERE snapshot_date <= %s
                )
                ORDER BY stock_code
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                snapshot_df = pd.read_sql(snapshot_query, conn, params=[as_of_date_str])
            if not snapshot_df.empty:
                codes = self._apply_universe_mode_filter(snapshot_df["stock_code"].tolist())
                source = "SNAPSHOT_ASOF"
                if is_survivor_optimistic_mode(self.universe_mode):
                    source = f"{source}_SURVIVOR_ONLY"
                return codes, source

            history_query = """
                SELECT stock_code
                FROM TickerUniverseHistory
                WHERE listed_date <= %s
                  AND (delisted_date IS NULL OR delisted_date > %s)
                ORDER BY stock_code
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                history_df = pd.read_sql(history_query, conn, params=[as_of_date_str, as_of_date_str])
            if not history_df.empty:
                codes = self._apply_universe_mode_filter(history_df["stock_code"].tolist())
                source = "HISTORY_ACTIVE_ASOF"
                if is_survivor_optimistic_mode(self.universe_mode):
                    source = f"{source}_SURVIVOR_ONLY"
                return codes, source
            return [], "NO_UNIVERSE"
        finally:
            if owns_conn:
                conn.close()

    def _query_latest_tier_codes(self, conn, as_of_date_str, max_tier):
        query = """
            SELECT t.stock_code
            FROM DailyStockTier t
            JOIN (
                SELECT stock_code, MAX(date) AS max_date
                FROM DailyStockTier
                WHERE date <= %s
                GROUP BY stock_code
            ) latest ON t.stock_code = latest.stock_code AND t.date = latest.max_date
            WHERE t.tier <= %s
            ORDER BY t.stock_code
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_sql(query, conn, params=[as_of_date_str, max_tier])
        if df.empty:
            return []
        return df["stock_code"].tolist()

    def _filter_by_min_liquidity(self, tier_map, min_liquidity_20d_avg_value):
        if min_liquidity_20d_avg_value is None:
            return tier_map

        min_liq = int(min_liquidity_20d_avg_value)
        filtered = {}
        for code, info in tier_map.items():
            liq_val = info.get("liquidity_20d_avg_value")
            if pd.isna(liq_val):
                continue
            if int(liq_val) >= min_liq:
                filtered[code] = info
        return filtered

    def _enforce_tier12_coverage_gate(self, date, pit_size, tier1_count, tier12_count, min_tier12_coverage_ratio):
        logger.debug(
            f"[TierCoverage] date={pd.to_datetime(date).date()} "
            f"tier1_count={tier1_count} tier12_count={tier12_count} universe_count={pit_size}"
        )

        if min_tier12_coverage_ratio is None or pit_size <= 0:
            return

        ratio = float(tier12_count) / float(pit_size)
        threshold = float(min_tier12_coverage_ratio)
        if ratio < threshold:
            raise PitRuntimeError(
                "tier12_coverage_gate_failed",
                f"Tier coverage gate failed on {pd.to_datetime(date).date()}: "
                f"tier12_ratio={ratio:.4f} < threshold={threshold:.4f} "
                f"(tier12_count={tier12_count}, universe_count={pit_size})",
                stage="tier12_coverage_gate",
                details={
                    "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                    "tier12_ratio": round(ratio, 6),
                    "threshold": round(threshold, 6),
                    "tier1_count": int(tier1_count),
                    "tier12_count": int(tier12_count),
                    "universe_count": int(pit_size),
                },
            )

    def _begin_snapshot_connection(self):
        conn = self.get_connection()
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                cur.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
                cur.execute("SELECT CONNECTION_ID(), CURRENT_TIMESTAMP()")
                row = cur.fetchone()
            return conn, {
                "connection_id": int((row[0] if row else 0) or 0),
                "snapshot_started_at": str(row[1]) if row and len(row) > 1 else None,
            }
        except Exception:
            conn.close()
            raise

    @staticmethod
    def _iter_signal_dates(trading_dates):
        dates = pd.to_datetime(list(trading_dates))
        if len(dates) <= 1:
            return []
        return sorted({pd.to_datetime(value).strftime("%Y-%m-%d") for value in dates[:-1]})

    @staticmethod
    def _hash_file(path):
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _load_frozen_candidate_manifest_file(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload

    def _write_frozen_candidate_manifest_file(self, manifest_path, payload):
        directory = os.path.dirname(manifest_path) or "."
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=directory,
            prefix=".tmp_frozen_manifest_",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
            tmp_path = f.name
        os.replace(tmp_path, manifest_path)

    def get_frozen_candidate_manifest_summary(self):
        return dict(self.last_frozen_candidate_manifest_summary)

    def _store_lazy_tier_candidate_payload(
        self,
        date,
        payload,
        min_liquidity_20d_avg_value=None,
        min_tier12_coverage_ratio=None,
    ):
        manifest_key = self._build_tier_manifest_key(
            min_liquidity_20d_avg_value,
            min_tier12_coverage_ratio,
        )
        if (
            self._lazy_tier_candidate_cache is None
            or manifest_key != self._lazy_tier_candidate_cache_key
        ):
            self._lazy_tier_candidate_cache = {}
            self._lazy_tier_candidate_cache_key = manifest_key

        self._lazy_tier_candidate_cache[self._normalize_date_key(date)] = payload

    def _resolve_tier_candidate_payload(
        self,
        date,
        min_liquidity_20d_avg_value=None,
        min_tier12_coverage_ratio=None,
        *,
        conn=None,
    ):
        pit_codes, pit_source = self.get_pit_universe_codes_as_of(date, conn=conn)
        if not pit_codes:
            return {
                "candidate_codes": [],
                "source": f"NO_CANDIDATES_{pit_source}",
                "pit_size": 0,
                "tier1_count": 0,
                "tier12_count": 0,
            }

        tier12_map = self.get_tiers_as_of(
            as_of_date=date,
            tickers=pit_codes,
            allowed_tiers=[1, 2],
            conn=conn,
        )
        tier12_map = self._filter_by_min_liquidity(tier12_map, min_liquidity_20d_avg_value)
        tier1_map = {
            code: info
            for code, info in tier12_map.items()
            if int(info.get("tier", 0) or 0) == 1
        }

        self._enforce_tier12_coverage_gate(
            date=date,
            pit_size=len(pit_codes),
            tier1_count=len(tier1_map),
            tier12_count=len(tier12_map),
            min_tier12_coverage_ratio=min_tier12_coverage_ratio,
        )

        if tier1_map:
            candidate_codes = [code for code in pit_codes if code in tier1_map]
            source = f"TIER_1_{pit_source}"
        elif tier12_map:
            candidate_codes = [code for code in pit_codes if code in tier12_map]
            source = f"TIER_2_FALLBACK_{pit_source}"
        else:
            candidate_codes = []
            source = f"NO_CANDIDATES_{pit_source}"

        return {
            "candidate_codes": candidate_codes,
            "source": source,
            "pit_size": len(pit_codes),
            "tier1_count": len(tier1_map),
            "tier12_count": len(tier12_map),
        }

    def _get_frozen_tier_candidate_payload(
        self,
        date,
        min_liquidity_20d_avg_value=None,
        min_tier12_coverage_ratio=None,
    ):
        if self._frozen_tier_candidate_manifest is None:
            return None

        key = self._build_tier_manifest_key(
            min_liquidity_20d_avg_value,
            min_tier12_coverage_ratio,
        )
        if key != self._frozen_tier_candidate_manifest_key:
            return None

        return self._frozen_tier_candidate_manifest.get(self._normalize_date_key(date))

    def _get_lazy_tier_candidate_payload(
        self,
        date,
        min_liquidity_20d_avg_value=None,
        min_tier12_coverage_ratio=None,
    ):
        if self._lazy_tier_candidate_cache is None:
            return None

        key = self._build_tier_manifest_key(
            min_liquidity_20d_avg_value,
            min_tier12_coverage_ratio,
        )
        if key != self._lazy_tier_candidate_cache_key:
            return None

        return self._lazy_tier_candidate_cache.get(self._normalize_date_key(date))

    def freeze_tier_candidate_manifest(
        self,
        trading_dates,
        *,
        min_liquidity_20d_avg_value=None,
        min_tier12_coverage_ratio=None,
    ):
        manifest_key = self._build_tier_manifest_key(
            min_liquidity_20d_avg_value,
            min_tier12_coverage_ratio,
        )
        self.clear_frozen_tier_candidate_manifest()
        manifest = {}
        normalized_dates = pd.to_datetime(list(trading_dates))
        for trading_date in normalized_dates:
            payload = self._resolve_tier_candidate_payload(
                trading_date,
                min_liquidity_20d_avg_value=min_liquidity_20d_avg_value,
                min_tier12_coverage_ratio=min_tier12_coverage_ratio,
            )
            manifest[self._normalize_date_key(trading_date)] = payload

        self._frozen_tier_candidate_manifest = manifest
        self._frozen_tier_candidate_manifest_key = manifest_key

        if not manifest:
            return {"days": 0}

        manifest_dates = sorted(manifest.keys())
        return {
            "days": len(manifest_dates),
            "first_date": manifest_dates[0],
            "last_date": manifest_dates[-1],
            "universe_mode": manifest_key[0],
            "min_liquidity_20d_avg_value": manifest_key[1],
            "min_tier12_coverage_ratio": manifest_key[2],
        }

    def prepare_strict_frozen_candidate_manifest(
        self,
        trading_dates,
        *,
        candidate_lookup_error_policy,
        min_liquidity_20d_avg_value=None,
        min_tier12_coverage_ratio=None,
    ):
        mode = self.frozen_candidate_manifest_mode
        if mode == "off":
            return {}
        if str(candidate_lookup_error_policy).strip().lower() != "raise":
            raise PitRuntimeError(
                "strict_frozen_requires_raise_policy",
                "strict frozen candidate manifest requires candidate_lookup_error_policy='raise'",
                stage="strict_frozen_manifest_prepare",
                details={
                    "candidate_lookup_error_policy": str(candidate_lookup_error_policy),
                    "mode": mode,
                },
            )

        manifest_key = self._build_tier_manifest_key(
            min_liquidity_20d_avg_value,
            min_tier12_coverage_ratio,
        )
        signal_dates = self._iter_signal_dates(trading_dates)
        if mode == "replay_strict":
            if not self.frozen_candidate_manifest_path:
                raise PitRuntimeError(
                    "frozen_manifest_path_missing",
                    "replay_strict requires frozen_candidate_manifest_path",
                    stage="strict_frozen_manifest_replay",
                    details={"mode": mode},
                )
            if not self.frozen_candidate_manifest_expected_sha256:
                raise PitRuntimeError(
                    "frozen_manifest_expected_sha_missing",
                    "replay_strict requires frozen_candidate_manifest_expected_sha256",
                    stage="strict_frozen_manifest_replay",
                    details={
                        "mode": mode,
                        "path": self.frozen_candidate_manifest_path,
                    },
                )
            actual_sha256 = self._hash_file(self.frozen_candidate_manifest_path)
            if actual_sha256 != self.frozen_candidate_manifest_expected_sha256:
                raise PitRuntimeError(
                    "frozen_manifest_sha_mismatch",
                    "Frozen candidate manifest sha256 mismatch.",
                    stage="strict_frozen_manifest_replay",
                    details={
                        "path": self.frozen_candidate_manifest_path,
                        "expected_sha256": self.frozen_candidate_manifest_expected_sha256,
                        "actual_sha256": actual_sha256,
                    },
                )
            payload = self._load_frozen_candidate_manifest_file(
                self.frozen_candidate_manifest_path
            )
            meta = payload.get("meta", {})
            if tuple(meta.get("manifest_key", ())) != manifest_key:
                raise PitRuntimeError(
                    "frozen_manifest_key_mismatch",
                    "Frozen candidate manifest key mismatch.",
                    stage="strict_frozen_manifest_replay",
                    details={
                        "path": self.frozen_candidate_manifest_path,
                        "expected_manifest_key": list(manifest_key),
                        "actual_manifest_key": list(meta.get("manifest_key", ())),
                    },
                )
            manifest = payload.get("manifest", {})
            manifest_dates = sorted(str(key) for key in manifest.keys())
            if manifest_dates != signal_dates:
                sample_size = 3
                raise PitRuntimeError(
                    "frozen_manifest_signal_date_set_mismatch",
                    "Frozen candidate manifest signal-date set mismatch.",
                    stage="strict_frozen_manifest_replay",
                    details={
                        "path": self.frozen_candidate_manifest_path,
                        "expected_signal_dates_count": len(signal_dates),
                        "actual_signal_dates_count": len(manifest_dates),
                        "expected_signal_dates_head": signal_dates[:sample_size],
                        "expected_signal_dates_tail": signal_dates[-sample_size:],
                        "actual_signal_dates_head": manifest_dates[:sample_size],
                        "actual_signal_dates_tail": manifest_dates[-sample_size:],
                    },
                )
            self._frozen_tier_candidate_manifest = manifest
            self._frozen_tier_candidate_manifest_key = manifest_key
            self._strict_frozen_candidate_manifest_required = True
            self.last_frozen_candidate_manifest_summary = {
                "mode": mode,
                "path": self.frozen_candidate_manifest_path,
                "sha256": actual_sha256,
                "expected_sha256": self.frozen_candidate_manifest_expected_sha256,
                "signal_dates": len(manifest),
            }
            return self.get_frozen_candidate_manifest_summary()

        snapshot_conn, snapshot_meta = self._begin_snapshot_connection()
        try:
            manifest = {}
            for signal_date in signal_dates:
                manifest[signal_date] = self._resolve_tier_candidate_payload(
                    signal_date,
                    min_liquidity_20d_avg_value=min_liquidity_20d_avg_value,
                    min_tier12_coverage_ratio=min_tier12_coverage_ratio,
                    conn=snapshot_conn,
                )
        finally:
            snapshot_conn.rollback()
            snapshot_conn.close()

        self._frozen_tier_candidate_manifest = manifest
        self._frozen_tier_candidate_manifest_key = manifest_key
        self._strict_frozen_candidate_manifest_required = True
        summary = {
            "mode": mode,
            "signal_dates": len(signal_dates),
            "signal_date_start": signal_dates[0] if signal_dates else None,
            "signal_date_end": signal_dates[-1] if signal_dates else None,
            "universe_mode": manifest_key[0],
            "min_liquidity_20d_avg_value": manifest_key[1],
            "min_tier12_coverage_ratio": manifest_key[2],
            "snapshot_connection_id": snapshot_meta.get("connection_id"),
            "snapshot_started_at": snapshot_meta.get("snapshot_started_at"),
        }
        if self.frozen_candidate_manifest_path:
            payload = {
                "meta": {
                    "manifest_key": list(manifest_key),
                    "signal_dates": signal_dates,
                    **summary,
                },
                "manifest": manifest,
            }
            self._write_frozen_candidate_manifest_file(
                self.frozen_candidate_manifest_path,
                payload,
            )
            summary["path"] = self.frozen_candidate_manifest_path
            summary["sha256"] = self._hash_file(self.frozen_candidate_manifest_path)
        self.last_frozen_candidate_manifest_summary = summary
        return self.get_frozen_candidate_manifest_summary()

    def get_candidates_with_tier_fallback_pit_gated(
        self,
        date,
        min_liquidity_20d_avg_value=None,
        min_tier12_coverage_ratio=None,
    ):
        """
        Issue #67 Phase2:
        PIT 유니버스(as-of) 안에서 Tier 1 우선, 없으면 Tier <= 2 fallback.
        Optional gates:
            - min_liquidity_20d_avg_value
            - min_tier12_coverage_ratio
        Returns:
            (candidates_list, source)
        """
        payload = self._get_frozen_tier_candidate_payload(
            date=date,
            min_liquidity_20d_avg_value=min_liquidity_20d_avg_value,
            min_tier12_coverage_ratio=min_tier12_coverage_ratio,
        )
        if payload is None and self._strict_frozen_candidate_manifest_required:
            normalized_date = self._normalize_date_key(date)
            raise PitRuntimeError(
                "frozen_manifest_miss",
                f"Frozen candidate manifest miss for {normalized_date}",
                stage="candidate_payload_lookup",
                details={
                    "date": normalized_date,
                    "manifest_mode": self.frozen_candidate_manifest_mode,
                    "manifest_key": list(self._frozen_tier_candidate_manifest_key or ()),
                },
            )
        if payload is None:
            payload = self._get_lazy_tier_candidate_payload(
                date=date,
                min_liquidity_20d_avg_value=min_liquidity_20d_avg_value,
                min_tier12_coverage_ratio=min_tier12_coverage_ratio,
            )
        if payload is None:
            payload = self._resolve_tier_candidate_payload(
                date=date,
                min_liquidity_20d_avg_value=min_liquidity_20d_avg_value,
                min_tier12_coverage_ratio=min_tier12_coverage_ratio,
            )
            self._store_lazy_tier_candidate_payload(
                date=date,
                payload=payload,
                min_liquidity_20d_avg_value=min_liquidity_20d_avg_value,
                min_tier12_coverage_ratio=min_tier12_coverage_ratio,
            )

        return list(payload["candidate_codes"]), str(payload["source"])
