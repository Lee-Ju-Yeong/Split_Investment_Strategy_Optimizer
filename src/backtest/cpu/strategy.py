"""
strategy.py

This module contains the functions for generating the signals for the Magic Split Strategy.
"""

from abc import ABC, abstractmethod
import math
import logging
import pandas as pd
import numpy as np
import uuid

try:
    from ...candidate_runtime_policy import normalize_runtime_candidate_policy
    from ...data_handler import build_pit_failure_record
    from ...tier_hysteresis_policy import normalize_tier_hysteresis_mode
except ImportError:  # pragma: no cover
    from candidate_runtime_policy import normalize_runtime_candidate_policy  # type: ignore
    from data_handler import build_pit_failure_record  # type: ignore
    from tier_hysteresis_policy import normalize_tier_hysteresis_mode  # type: ignore

logger = logging.getLogger(__name__)

class Position:
    """ 포지션 정보를 저장하는 데이터 클래스 """
    def __init__(self, buy_price, quantity, order, additional_buy_drop_rate, sell_profit_rate):
        self.position_id = str(uuid.uuid4())
        self.buy_price = buy_price
        self.quantity = quantity
        self.order = order
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate
        self.open_date = None 

class Strategy(ABC):
    @abstractmethod
    def generate_sell_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None): 
        """매도 관련 신호(수익 실현, 손절 등)를 생성합니다."""
        raise NotImplementedError("generate_sell_signals() 메소드를 구현해야 합니다.")

    # 신규 진입과 추가 매수를 위한 두 개의 구체적인 추상 메소드를 정의합니다.
    @abstractmethod
    def generate_new_entry_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        """신규 종목 진입 신호를 생성합니다."""
        raise NotImplementedError("generate_new_entry_signals() 메소드를 구현해야 합니다.")

    @abstractmethod
    def generate_additional_buy_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        """기존 보유 종목의 추가 매수 신호를 생성합니다."""
        raise NotImplementedError("generate_additional_buy_signals() 메소드를 구현해야 합니다.")

class MagicSplitStrategy(Strategy):
    VALID_ADDITIONAL_BUY_PRIORITIES = {"lowest_order", "highest_drop"}

    def __init__(
        self,
        max_stocks,
        order_investment_ratio,
        additional_buy_drop_rate,
        sell_profit_rate,
        backtest_start_date,
        backtest_end_date,
        additional_buy_priority="lowest_order",
        cooldown_period_days=5,
        # --- [New] Advanced Risk Management Parameters ---
        stop_loss_rate=-0.15,
        max_splits_limit=10,
        max_inactivity_period=90,
        # --- [Issue #67] Candidate Source Config ---
        candidate_source_mode="tier",
        use_weekly_alpha_gate=False,
        min_liquidity_20d_avg_value=0,
        min_tier12_coverage_ratio=0.0,
        tier_hysteresis_mode="strict_hysteresis_v1",
        candidate_lookup_error_policy="raise",
        buy_commission_rate=0.00015,
        enable_candidate_rank_trace=False,
    ):
        self.max_stocks = max_stocks
        self.order_investment_ratio = order_investment_ratio
        self.additional_buy_drop_rate = additional_buy_drop_rate
        self.sell_profit_rate = sell_profit_rate
        self.backtest_start_date = pd.to_datetime(backtest_start_date)
        self.backtest_end_date = pd.to_datetime(backtest_end_date)
        self.additional_buy_priority = self._normalize_additional_buy_priority(additional_buy_priority)
        self.cooldown_period_days = cooldown_period_days
        self.stop_loss_rate = stop_loss_rate
        self.max_splits_limit = max_splits_limit
        self.max_inactivity_period = max_inactivity_period
        
        # [Issue #67]
        normalized_mode, normalized_weekly_gate = normalize_runtime_candidate_policy(
            candidate_source_mode,
            use_weekly_alpha_gate,
        )
        self.candidate_source_mode = normalized_mode
        self.use_weekly_alpha_gate = normalized_weekly_gate
        self.min_liquidity_20d_avg_value = min_liquidity_20d_avg_value
        self.min_tier12_coverage_ratio = min_tier12_coverage_ratio
        self.tier_hysteresis_mode = normalize_tier_hysteresis_mode(tier_hysteresis_mode)
        self.candidate_lookup_error_policy = str(candidate_lookup_error_policy).strip().lower()
        if self.candidate_lookup_error_policy not in {"raise", "skip"}:
            raise ValueError(
                "Unsupported candidate_lookup_error_policy="
                f"{candidate_lookup_error_policy!r}. supported=[raise, skip]"
            )
        self.buy_commission_rate = float(buy_commission_rate)
        self.enable_candidate_rank_trace = bool(enable_candidate_rank_trace)
        self.strict_hysteresis_enabled = True
        
        self.investment_per_order = 0
        self.previous_month = -1
        self.cooldown_tracker = {}  # 매도된 종목 추적
        self.last_entry_context = {
            "signal_date": None,
            "tier_source": "UNINITIALIZED",
            "raw_candidate_count": 0,
            "active_candidate_count": 0,
            "ranked_candidate_count": 0,
            "selected_count": 0,
            "strategy_candidate_mode": self._resolve_candidate_mode(),
        }
        self.candidate_rank_history = []
        self.candidate_lookup_error_count = 0
        self.first_candidate_lookup_error = None
        self.last_candidate_lookup_error = None
        self.candidate_lookup_error_counts_by_code = {}

    @classmethod
    def _normalize_additional_buy_priority(cls, value):
        key = str(value).strip().lower()
        if key == "biggest_drop":
            key = "highest_drop"
        if key not in cls.VALID_ADDITIONAL_BUY_PRIORITIES:
            raise ValueError(
                "Unsupported additional_buy_priority="
                f"{value!r}. supported=[lowest_order, highest_drop]"
            )
        return key

    def _resolve_signal_date(self, current_date, trading_dates, current_day_idx, data_handler):
        if current_day_idx is None:
            return pd.to_datetime(current_date)
        return data_handler.get_previous_trading_date(trading_dates, current_day_idx)

    def _resolve_candidate_mode(self):
        return self.candidate_source_mode

    @staticmethod
    def _build_entry_context(
        signal_date,
        tier_source,
        *,
        strategy_candidate_mode,
        raw_candidate_count=0,
        active_candidate_count=0,
        ranked_candidate_count=0,
        selected_count=0,
        pit_failure_code=None,
        pit_failure_stage=None,
    ):
        return {
            "signal_date": signal_date,
            "tier_source": str(tier_source),
            "raw_candidate_count": int(raw_candidate_count),
            "active_candidate_count": int(active_candidate_count),
            "ranked_candidate_count": int(ranked_candidate_count),
            "selected_count": int(selected_count),
            "strategy_candidate_mode": strategy_candidate_mode,
            "pit_failure_code": pit_failure_code,
            "pit_failure_stage": pit_failure_stage,
        }

    def _record_candidate_lookup_error(self, failure_record):
        if not failure_record:
            return
        self.candidate_lookup_error_count += 1
        if self.first_candidate_lookup_error is None:
            self.first_candidate_lookup_error = dict(failure_record)
        self.last_candidate_lookup_error = dict(failure_record)
        code = str(failure_record.get("code") or "unknown_candidate_lookup_error")
        self.candidate_lookup_error_counts_by_code[code] = (
            int(self.candidate_lookup_error_counts_by_code.get(code, 0)) + 1
        )

    @staticmethod
    def _build_candidate_rank_rows(current_date, signal_date, sorted_candidates):
        trade_date_str = pd.to_datetime(current_date).strftime("%Y-%m-%d")
        signal_date_str = pd.to_datetime(signal_date).strftime("%Y-%m-%d")
        rows = []
        for rank, candidate in enumerate(sorted_candidates, start=1):
            rows.append(
                {
                    "trade_date": trade_date_str,
                    "signal_date": signal_date_str,
                    "rank": int(rank),
                    "ticker": str(candidate["ticker"]),
                    "entry_composite_score_q": int(candidate["entry_composite_score_q"]),
                    "flow_score_q": int(candidate["flow_score_q"]),
                    "atr_score_q": int(candidate["atr_score_q"]),
                    "market_cap_q": int(candidate["market_cap_q"]),
                    "atr_14_ratio": float(candidate["atr_14_ratio"]),
                }
            )
        return rows

    @staticmethod
    def _coerce_tier_value(tier_info):
        if tier_info is None:
            return 0
        raw_tier = tier_info.get("tier") if isinstance(tier_info, dict) else None
        if raw_tier is None:
            return 0
        try:
            if pd.isna(raw_tier):
                return 0
            return int(raw_tier)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _get_tick_size(price):
        if price < 2000:
            return 1
        if price < 5000:
            return 5
        if price < 20000:
            return 10
        if price < 50000:
            return 50
        if price < 200000:
            return 100
        if price < 500000:
            return 500
        return 1000

    @classmethod
    def _adjust_price_up(cls, price):
        tick_size = cls._get_tick_size(price)
        divided = price / tick_size
        rounded = round(divided, 5)
        return math.ceil(rounded) * tick_size

    @staticmethod
    def _mul_f32(a, b):
        return float(np.float32(a) * np.float32(b))

    def _calculate_monthly_investment(self, current_date, current_day_idx, trading_dates, portfolio, data_handler):
        # 전일 날짜를 기준으로 자산을 평가하도록 로직 변경
        current_month = current_date.month
        if current_month != self.previous_month:
            # 첫 거래일에는 전일이 없으므로, 초기 자본을 기준으로 투자금을 계산
            if current_day_idx == 0:
                total_portfolio_value = portfolio.initial_cash
            else:
                # 전일 날짜를 가져옴
                previous_day_date = trading_dates[current_day_idx - 1]
                total_portfolio_value = portfolio.get_total_value(previous_day_date, data_handler)
            
            self.investment_per_order = np.float32(total_portfolio_value) * np.float32(self.order_investment_ratio)
            self.previous_month = current_month
    # 신규 진입 신호만 생성하는 함수
    def generate_new_entry_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        # 각 신호 생성 함수가 독립적으로 호출되므로, 투자금 계산 로직은 각 함수에 포함되어야 합니다.
        self._calculate_monthly_investment(current_date, current_day_idx, trading_dates, portfolio, data_handler)
        buy_signals = []
        signal_date = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
        resolved_mode = self._resolve_candidate_mode()
        self.last_entry_context = self._build_entry_context(
            signal_date,
            "NO_SIGNAL_DATE" if signal_date is None else "UNSET",
            strategy_candidate_mode=resolved_mode,
        )
        if signal_date is None:
            return buy_signals

        # 신규 매수 신호 생성 로직 (기존 generate_buy_signals의 2번 로직)
        available_slots = self.max_stocks - len(portfolio.positions)
        if (current_date - self.backtest_start_date).days < 15:
            from tqdm import tqdm
            log_msg = (
                f"[CPU_SLOT_DEBUG] {current_date.strftime('%Y-%m-%d')} | "
                f"MaxStocks: {self.max_stocks}, "
                f"CurrentHoldings: {len(portfolio.positions)}, "
                f"AvailableSlots: {available_slots}"
            )
            tqdm.write(log_msg)
        
        if available_slots > 0:
            candidate_codes = []
            
            # --- [Issue #67] Candidate Source Logic Start ---
            get_tier_candidates_pit_gated = getattr(
                data_handler,
                "get_candidates_with_tier_fallback_pit_gated",
                None,
            )
            if callable(get_tier_candidates_pit_gated):
                try:
                    used_tier = ""
                    raw_candidate_count = 0
                    tier_codes, used_tier = get_tier_candidates_pit_gated(
                        signal_date,
                        min_liquidity_20d_avg_value=self.min_liquidity_20d_avg_value,
                        min_tier12_coverage_ratio=self.min_tier12_coverage_ratio,
                    )
                    raw_candidate_count = len(tier_codes)
                    candidate_codes = tier_codes
                    if self.strict_hysteresis_enabled and used_tier.startswith("TIER_2_FALLBACK"):
                        candidate_codes = []
                        used_tier = f"{used_tier}_BLOCKED_BY_HYSTERESIS"

                    from tqdm import tqdm
                    if used_tier.startswith("TIER_2_FALLBACK"):
                        blocked_suffix = (
                            " [blocked by strict_hysteresis_v1]"
                            if used_tier.endswith("BLOCKED_BY_HYSTERESIS")
                            else ""
                        )
                        tqdm.write(
                            f"[Strategy] {current_date.date()} | "
                            f"Fallback: Tier 1 (0) -> Tier 2 ({raw_candidate_count}){blocked_suffix}"
                        )
                except Exception as exc:
                    failure_record = build_pit_failure_record(
                        exc,
                        trade_date=current_date,
                        signal_date=signal_date,
                        fallback_code="candidate_lookup_runtime_error",
                        fallback_stage="candidate_lookup",
                    )
                    self._record_candidate_lookup_error(failure_record)
                    self.last_entry_context = self._build_entry_context(
                        signal_date,
                        "CANDIDATE_LOOKUP_ERROR",
                        strategy_candidate_mode=resolved_mode,
                        pit_failure_code=(failure_record or {}).get("code"),
                        pit_failure_stage=(failure_record or {}).get("stage"),
                    )
                    if self.candidate_lookup_error_policy == "raise":
                        logger.exception(
                            "[PITCandidateFailure] code=%s stage=%s policy=raise trade_date=%s signal_date=%s "
                            "message=%s",
                            (failure_record or {}).get("code"),
                            (failure_record or {}).get("stage"),
                            (failure_record or {}).get("trade_date"),
                            (failure_record or {}).get("signal_date"),
                            (failure_record or {}).get("message"),
                        )
                        raise
                    logger.warning(
                        "[PITCandidateFailure] code=%s stage=%s policy=skip trade_date=%s signal_date=%s "
                        "message=%s. Returning empty candidate set.",
                        (failure_record or {}).get("code"),
                        (failure_record or {}).get("stage"),
                        (failure_record or {}).get("trade_date"),
                        (failure_record or {}).get("signal_date"),
                        (failure_record or {}).get("message"),
                    )
                    candidate_codes = []
                    raw_candidate_count = 0
                    used_tier = "CANDIDATE_LOOKUP_ERROR"
                self.last_entry_context = self._build_entry_context(
                    signal_date,
                    used_tier,
                    strategy_candidate_mode=resolved_mode,
                    raw_candidate_count=raw_candidate_count,
                    pit_failure_code=(self.last_candidate_lookup_error or {}).get("code")
                    if used_tier == "CANDIDATE_LOOKUP_ERROR"
                    else None,
                    pit_failure_stage=(self.last_candidate_lookup_error or {}).get("stage")
                    if used_tier == "CANDIDATE_LOOKUP_ERROR"
                    else None,
                )
            else:
                logger.warning(
                    "get_candidates_with_tier_fallback_pit_gated missing. Returning empty candidate set."
                )
                candidate_codes = []
                self.last_entry_context = self._build_entry_context(
                    signal_date,
                    "CANDIDATE_SOURCE_MISSING",
                    strategy_candidate_mode=resolved_mode,
                )
            # --- [Issue #67] Logic End ---
            
            active_candidates = []
            for code in candidate_codes:
                # 쿨다운 체크 로직을 GPU와 동일하게 거래일 인덱스 차이로 
                is_in_cooldown = False
                if self.cooldown_tracker.get(code) is not None:
                    if (current_day_idx - self.cooldown_tracker.get(code)) < self.cooldown_period_days:
                        is_in_cooldown = True

                if code in portfolio.positions or is_in_cooldown:
                    continue
                active_candidates.append(code)
            self.last_entry_context["active_candidate_count"] = len(active_candidates)

            candidate_rows_by_ticker = {}
            get_stock_rows_as_of = getattr(data_handler, "get_stock_rows_as_of", None)
            if active_candidates and callable(get_stock_rows_as_of):
                batch_rows = get_stock_rows_as_of(
                    active_candidates,
                    signal_date,
                    self.backtest_start_date,
                    self.backtest_end_date,
                )
                if isinstance(batch_rows, dict):
                    candidate_rows_by_ticker = batch_rows

            ranked_candidates = []
            for ticker in active_candidates:
                if ticker in candidate_rows_by_ticker:
                    signal_row = candidate_rows_by_ticker[ticker]
                else:
                    signal_row = data_handler.get_stock_row_as_of(
                        ticker,
                        signal_date,
                        self.backtest_start_date,
                        self.backtest_end_date,
                    )
                if signal_row is None or "atr_14_ratio" not in signal_row.index:
                    continue
                latest_atr = signal_row["atr_14_ratio"]
                signal_close = signal_row.get("close_price")
                if not (pd.notna(latest_atr) and pd.notna(signal_close)):
                    continue

                atr_float = float(latest_atr)
                if atr_float <= 0.0:
                    continue

                market_cap_raw = signal_row.get("market_cap")
                if pd.notna(market_cap_raw):
                    market_cap_float = float(market_cap_raw)
                    market_cap_q = int(market_cap_float // 1_000_000) if market_cap_float > 0.0 else 0
                else:
                    market_cap_q = 0

                cheap_score_raw = signal_row.get("cheap_score")
                cheap_conf_raw = signal_row.get("cheap_score_confidence")
                cheap_score = float(cheap_score_raw) if pd.notna(cheap_score_raw) else 0.0
                cheap_conf = float(cheap_conf_raw) if pd.notna(cheap_conf_raw) else 0.0
                cheap_effective = max(min(cheap_score, 1.0), 0.0) * max(min(cheap_conf, 1.0), 0.0)
                cheap_score_q = int(round(cheap_effective * 10000.0))
                flow5_mcap_raw = signal_row.get("flow5_mcap")
                flow5_mcap = float(flow5_mcap_raw) if pd.notna(flow5_mcap_raw) else np.nan

                atr_q = int(round(atr_float * 10000))
                ranked_candidates.append(
                    {
                        "ticker": ticker,
                        "signal_close_price": float(signal_close),
                        "cheap_effective": cheap_effective,
                        "cheap_score_q": cheap_score_q,
                        "flow5_mcap": flow5_mcap,
                        "market_cap_q": market_cap_q,
                        "atr_14_ratio": atr_float,
                        "atr_q": atr_q,
                    }
                )
            self.last_entry_context["ranked_candidate_count"] = len(ranked_candidates)

            sorted_candidates = []
            if ranked_candidates:
                ranked_df = pd.DataFrame(ranked_candidates)
                ranked_df["flow_score"] = (
                    pd.to_numeric(ranked_df["flow5_mcap"], errors="coerce")
                    .rank(method="average", pct=True, ascending=True)
                    .fillna(0.0)
                )
                ranked_df["flow_score_q"] = (ranked_df["flow_score"] * 10000.0).round().astype(int)
                ranked_df["atr_score"] = (
                    pd.to_numeric(ranked_df["atr_14_ratio"], errors="coerce")
                    .rank(method="average", pct=True, ascending=True)
                    .fillna(0.0)
                )
                ranked_df["atr_score_q"] = (ranked_df["atr_score"] * 10000.0).round().astype(int)
                ranked_df["entry_composite_score"] = (
                    (0.50 * ranked_df["cheap_effective"])
                    + (0.30 * ranked_df["flow_score"])
                    + (0.20 * ranked_df["atr_score"])
                )
                ranked_df["entry_composite_score_q"] = (
                    ranked_df["entry_composite_score"].fillna(0.0) * 10000.0
                ).round().astype(int)
                ranked_df.sort_values(
                    by=["entry_composite_score_q", "market_cap_q", "ticker"],
                    ascending=[False, False, True],
                    inplace=True,
                )
                sorted_candidates = ranked_df.to_dict("records")
                if self.enable_candidate_rank_trace:
                    self.candidate_rank_history.extend(
                        self._build_candidate_rank_rows(current_date, signal_date, sorted_candidates)
                    )
            # 슬롯이 찰 때까지만 신호 생성
            num_new_entries = 0
            temp_cash = float(portfolio.cash)
            for candidate in sorted_candidates:
                if num_new_entries >= available_slots: break
                ticker = candidate["ticker"]
                signal_close_price = candidate["signal_close_price"]
                if signal_close_price > 0 and self.investment_per_order > 0:
                    if temp_cash < float(self.investment_per_order):
                        break
                    ohlc_row = data_handler.get_ohlc_data_on_date(
                        current_date,
                        ticker,
                        self.backtest_start_date,
                        self.backtest_end_date,
                    )
                    if ohlc_row is None:
                        continue
                    open_price = ohlc_row.get("open_price")
                    if not pd.notna(open_price):
                        continue
                    execution_price = self._adjust_price_up(float(open_price))
                    if execution_price <= 0:
                        continue
                    expected_quantity = int(math.floor(float(self.investment_per_order) / float(execution_price)))
                    if expected_quantity <= 0:
                        continue
                    gross_cost = float(execution_price) * float(expected_quantity)
                    commission = math.floor(gross_cost * float(self.buy_commission_rate))
                    total_cost = gross_cost + commission
                    if temp_cash < total_cost:
                        continue
                    new_pos = Position(signal_close_price, 0, 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                    buy_signals.append(
                        self._create_buy_signal(
                            current_date,
                            ticker,
                            self.investment_per_order,
                            new_pos,
                            1,
                            (
                                -candidate["entry_composite_score_q"],
                                -candidate["market_cap_q"],
                            ),
                            "신규 진입",
                            signal_close_price,
                        )
                    )
                    temp_cash -= total_cost
                    num_new_entries += 1
            self.last_entry_context["selected_count"] = len(buy_signals)
        else:
            self.last_entry_context = self._build_entry_context(
                signal_date,
                "NO_AVAILABLE_SLOTS",
                strategy_candidate_mode=resolved_mode,
            )

        # 신규 매수 신호 내에서의 정렬
        buy_signals.sort(key=lambda s: (s["priority_group"], s["sort_metric"], s["ticker"]))
        
        for signal in buy_signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date
            
        return buy_signals

    # 추가 매수 신호만 생성하는 함수
    def generate_additional_buy_signals(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        # 투자금 재계산은 신규 매수에서 이미 했을 수 있지만, 이 함수가 단독으로 사용될 수도 있으므로 여기서도 호출합니다.
        self._calculate_monthly_investment(current_date, current_day_idx, trading_dates, portfolio, data_handler)
        buy_signals = []
        signal_date = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
        if signal_date is None:
            return buy_signals

        holdings_tickers = list(portfolio.positions.keys())
        tier_map = None
        get_tiers_as_of = getattr(data_handler, "get_tiers_as_of", None)
        if holdings_tickers and callable(get_tiers_as_of):
            batch_tiers = get_tiers_as_of(
                signal_date,
                tickers=holdings_tickers,
                allowed_tiers=[1, 2],
            )
            if isinstance(batch_tiers, dict):
                tier_map = batch_tiers

        eligible_tickers = []
        eligible_positions = {}
        for ticker in holdings_tickers:
            if self.cooldown_tracker.get(ticker) == current_day_idx:
                continue
            if isinstance(tier_map, dict):
                tier_info = tier_map.get(ticker)
            else:
                tier_info = data_handler.get_stock_tier_as_of(ticker, signal_date)
            tier_value = self._coerce_tier_value(tier_info)
            if tier_value <= 0 or tier_value > 2:
                continue

            positions = portfolio.positions[ticker]
            if len(positions) >= self.max_splits_limit:
                continue

            first_position = next((p for p in positions if p.order == 1), None)
            if first_position is None or first_position.open_date is None or first_position.open_date >= current_date:
                continue

            eligible_tickers.append(ticker)
            eligible_positions[ticker] = positions

        signal_rows_by_ticker = {}
        get_stock_rows_as_of = getattr(data_handler, "get_stock_rows_as_of", None)
        if eligible_tickers and callable(get_stock_rows_as_of):
            batch_rows = get_stock_rows_as_of(
                eligible_tickers,
                signal_date,
                self.backtest_start_date,
                self.backtest_end_date,
            )
            if isinstance(batch_rows, dict):
                signal_rows_by_ticker = batch_rows

        # 추가 매수 신호 생성 로직
        for ticker in eligible_tickers:
            positions = eligible_positions[ticker]
            if ticker in signal_rows_by_ticker:
                signal_row = signal_rows_by_ticker[ticker]
            else:
                signal_row = data_handler.get_stock_row_as_of(
                    ticker,
                    signal_date,
                    self.backtest_start_date,
                    self.backtest_end_date,
                )
            if signal_row is None:
                continue
            
            signal_close = signal_row["close_price"]
            signal_low = signal_row["low_price"]
            if signal_close <= 0 or signal_low <= 0:
                continue

            last_pos = portfolio.positions[ticker][-1]
            buy_trigger_price = self._mul_f32(
                last_pos.buy_price,
                np.float32(1.0) - np.float32(self.additional_buy_drop_rate),
            )
            
            if signal_low <= buy_trigger_price:
                if self.investment_per_order > 0:
                    new_pos = Position(signal_close, 0, len(portfolio.positions[ticker]) + 1, self.additional_buy_drop_rate, self.sell_profit_rate)
                    if self.additional_buy_priority == "lowest_order":
                        sort_metric = len(portfolio.positions[ticker])
                    elif self.additional_buy_priority == "highest_drop":
                        sort_metric = -((last_pos.buy_price - signal_close) / last_pos.buy_price)
                    else:
                        raise ValueError(
                            "Unsupported additional_buy_priority="
                            f"{self.additional_buy_priority!r}. supported=[lowest_order, highest_drop]"
                        )
                    buy_signals.append(self._create_buy_signal(current_date, ticker, self.investment_per_order, new_pos, 2, sort_metric, "추가 매수(하락)", buy_trigger_price))

        # 추가 매수 신호 내에서의 정렬
        buy_signals.sort(key=lambda s: (s["priority_group"], s["sort_metric"], s["ticker"]))

        for signal in buy_signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date

        return buy_signals

    


    def _create_sell_signal(self, date, ticker, position, reason, trigger_price):
        return {"date": date, "ticker": ticker, "type": "SELL", "quantity": position.quantity, "position": position, "reason_for_trade": reason, "trigger_price": trigger_price}

    def get_candidate_lookup_error_summary(self):
        return {
            "error_count": int(self.candidate_lookup_error_count),
            "first_error": self.first_candidate_lookup_error,
            "last_error": self.last_candidate_lookup_error,
            "failure_counts": dict(sorted(self.candidate_lookup_error_counts_by_code.items())),
            "policy": self.candidate_lookup_error_policy,
        }

    def _create_buy_signal(self, date, ticker, investment_amount, position, priority, sort_metric, reason, trigger_price):
        return {"date": date, "ticker": ticker, "type": "BUY", "investment_amount": investment_amount, "position": position, "priority_group": priority, "sort_metric": sort_metric, "reason_for_trade": reason, "trigger_price": trigger_price}
    def generate_sell_signals(self, current_date, portfolio, data_handler,trading_dates,current_day_idx=None):
        self._calculate_monthly_investment(current_date, current_day_idx, trading_dates, portfolio, data_handler)
        signals = []
        signal_date = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
        if signal_date is None:
            return signals

        for ticker in list(portfolio.positions.keys()):
            row = data_handler.get_stock_row_as_of(
                ticker, signal_date, self.backtest_start_date, self.backtest_end_date
            )
            if row is None:
                continue

            current_price = row["close_price"]
            current_high = row["high_price"]
            if current_price <= 0: continue

            positions = portfolio.positions[ticker]
            avg_buy_price = sum(p.quantity * p.buy_price for p in positions) / sum(p.quantity for p in positions)

            # --- 1. 리스크 관리 조건 확인 (종목 전체 청산) ---
            liquidate = False
            reason = ""
            trigger_price = current_price

            stop_loss_trigger_price = self._mul_f32(
                avg_buy_price,
                np.float32(1.0) + np.float32(self.stop_loss_rate),
            )
            if current_price <= stop_loss_trigger_price:
                liquidate = True
                reason = "손절매 (평균가 기준)"
                trigger_price = stop_loss_trigger_price

            if not liquidate:
                last_trade_idx = portfolio.last_trade_day_indices.get(ticker)
                if last_trade_idx is not None:
                    # GPU와 동일한 '실제 거래일' 기준으로 비활성 기간 계산
                    days_inactive = current_day_idx - last_trade_idx
                    
                    if days_inactive >= self.max_inactivity_period -1: # GPU 로직과 동일하게 >= 및 -1 조건 사용
                        liquidate = True
                        reason = "매매 미발생 기간 초과"

            if liquidate:
                # 임시 객체 생성 대신, 실제 보유 중인 모든 포지션에 대해 매도 신호를 생성합니다.
                # 이는 execution 단계에서 올바른 position_id를 참조하여 포지션을 제거할 수 있도록 보장합니다.
                for p in positions:
                    signals.append(self._create_sell_signal(current_date, ticker, p, reason, trigger_price))
                
                # 신호가 하나라도 생성되었다면 쿨다운을 설정합니다.
                if positions:
                    self.cooldown_tracker[ticker] = current_day_idx
                
                continue # 이 종목에 대한 다른 매도(수익실현) 검사는 건너뜁니다.
            # reversed()를 사용하여 가장 최근 매수 포지션부터 순회
            for p in reversed(positions):
                # [규칙] 당일 매수한 포지션은 익절 대상에서 제외
                if p.open_date is not None and p.open_date >= current_date:
                    continue
                
                sell_trigger_price = self._mul_f32(
                    p.buy_price,
                    np.float32(1.0) + np.float32(self.sell_profit_rate),
                )
                if current_high >= sell_trigger_price:
                    signals.append(self._create_sell_signal(current_date, ticker, p, "수익 실현", sell_trigger_price))
                    # 1차 매도 여부와 상관없이 개별 익절이므로 cooldown만 설정
                    self.cooldown_tracker[ticker] = current_day_idx
        
        for signal in signals:
            signal["start_date"] = self.backtest_start_date
            signal["end_date"] = self.backtest_end_date
        return signals
