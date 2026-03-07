# backtester.py (수정 필수!)

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    # Optional dependency: allow CPU backtester to run in minimal/laptop envs.
    def tqdm(iterable, **_kwargs):
        return iterable
import pandas as pd
import warnings
import logging
import os
from typing import Optional

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class BacktestEngine:
    def __init__(
        self,
        start_date,
        end_date,
        portfolio,
        strategy,
        data_handler,
        execution_handler,
        *,
        logger: Optional[logging.Logger] = None,
        debug_ticker: Optional[str] = None,
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.portfolio = portfolio
        self.strategy = strategy
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.logger = logger or logging.getLogger(__name__)
        self.debug_ticker = debug_ticker

    def _clear_tier_candidate_cache_if_supported(self):
        clear_manifest = getattr(
            self.data_handler,
            "clear_lazy_tier_candidate_cache",
            None,
        )
        if not callable(clear_manifest):
            return

        clear_manifest()

    def run(self):
        self.logger.info("백테스팅 엔진을 시작합니다...")
        
        trading_dates = self.data_handler.get_trading_dates(self.start_date, self.end_date)
        self._clear_tier_candidate_cache_if_supported()
        entry_stats = {
            "entry_opportunity_days": 0,
            "candidate_eval_days": 0,
            "empty_entry_days": 0,
            "tier1_source_days": 0,
            "tier2_fallback_days": 0,
            "tier2_blocked_days": 0,
            "no_candidates_days": 0,
            "no_signal_date_days": 0,
            "lookup_error_days": 0,
            "source_missing_days": 0,
            "unknown_source_days": 0,
            "metrics_cast_error_count": 0,
            "raw_candidate_count_sum": 0,
            "active_candidate_count_sum": 0,
            "ranked_candidate_count_sum": 0,
            "selected_signal_count_sum": 0,
        }

        debug_ticker = self.debug_ticker
        if not debug_ticker:
            env_debug_ticker = os.getenv("BACKTEST_DEBUG_TICKER", "").strip()
            debug_ticker = env_debug_ticker or None
        
        # tqdm의 mininterval을 늘려 로그 출력이 밀리지 않게 함
        for i, current_date in enumerate(tqdm(trading_dates, desc="Backtesting Progress", mininterval=1.0)):
            if debug_ticker and self.logger.isEnabledFor(logging.DEBUG):
                try:
                    stock_data = self.data_handler.load_stock_data(debug_ticker, self.start_date, self.end_date)
                    # 데이터프레임이 유효하고, 현재 날짜의 데이터가 존재하는지 확인
                    if stock_data is not None and not stock_data.empty and current_date in stock_data.index:
                        ohlc = stock_data.loc[current_date]
                        self.logger.debug(
                            "[CPU_DATA_DEBUG] %s | %s | Open=%s, High=%s, Low=%s, Close=%s",
                            current_date.strftime("%Y-%m-%d"),
                            debug_ticker,
                            ohlc.get("open_price"),
                            ohlc.get("high_price"),
                            ohlc.get("low_price"),
                            ohlc.get("close_price"),
                        )
                except Exception:
                    # 디버그 경로는 백테스트를 깨지 않되, 조사 가능하도록 예외를 남깁니다.
                    self.logger.debug("debug ticker fetch skipped", exc_info=True)
            # --- 1. [변경] 신호 생성 및 실행 로직 분리 ---
            
            # [추가] 오늘 거래가 실행되기 전의 거래 내역 개수를 기록
            num_trades_before = len(self.portfolio.trade_history)

            # 단계 1-1: 매도 신호 생성 및 즉시 실행
            sell_signals = self.strategy.generate_sell_signals(current_date, self.portfolio, self.data_handler, trading_dates, i)
            if sell_signals:
                for signal in sell_signals:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler, i)

            # 단계 1-2: 매수 신호를 '신규 진입' -> '추가 매수' 순으로 분리하여 실행
            # (1) 신규 진입 신호 생성 및 실행
            strategy_max_stocks = int(getattr(self.strategy, "max_stocks", 0) or 0)
            available_slots_before_entry = max(0, strategy_max_stocks - len(self.portfolio.positions))
            new_entry_signals = self.strategy.generate_new_entry_signals(current_date, self.portfolio, self.data_handler, trading_dates, i)
            entry_context = getattr(self.strategy, "last_entry_context", {}) or {}
            if available_slots_before_entry > 0:
                def _safe_int(value):
                    if value is None:
                        return 0, False
                    try:
                        if pd.isna(value):
                            return 0, False
                    except Exception:
                        pass
                    try:
                        return int(value), False
                    except (TypeError, ValueError):
                        return 0, True

                entry_stats["entry_opportunity_days"] += 1
                if not new_entry_signals:
                    entry_stats["empty_entry_days"] += 1
                tier_source = str(entry_context.get("tier_source", ""))
                if tier_source.startswith("TIER_1"):
                    entry_stats["tier1_source_days"] += 1
                    entry_stats["candidate_eval_days"] += 1
                elif tier_source.startswith("TIER_2_FALLBACK"):
                    entry_stats["tier2_fallback_days"] += 1
                    entry_stats["candidate_eval_days"] += 1
                    if tier_source.endswith("BLOCKED_BY_HYSTERESIS"):
                        entry_stats["tier2_blocked_days"] += 1
                elif tier_source.startswith("NO_CANDIDATES"):
                    entry_stats["no_candidates_days"] += 1
                    entry_stats["candidate_eval_days"] += 1
                elif tier_source.startswith("NO_SIGNAL_DATE"):
                    entry_stats["no_signal_date_days"] += 1
                elif tier_source.startswith("CANDIDATE_LOOKUP_ERROR"):
                    entry_stats["lookup_error_days"] += 1
                elif tier_source.startswith("CANDIDATE_SOURCE_MISSING"):
                    entry_stats["source_missing_days"] += 1
                elif tier_source.startswith("NO_AVAILABLE_SLOTS"):
                    # available_slots_before_entry > 0 에서는 발생하지 않아야 하는 방어용 카운트
                    entry_stats["unknown_source_days"] += 1
                elif not tier_source:
                    entry_stats["unknown_source_days"] += 1
                else:
                    entry_stats["unknown_source_days"] += 1
                raw_count, raw_cast_error = _safe_int(entry_context.get("raw_candidate_count", 0))
                active_count, active_cast_error = _safe_int(entry_context.get("active_candidate_count", 0))
                ranked_count, ranked_cast_error = _safe_int(entry_context.get("ranked_candidate_count", 0))
                selected_count, selected_cast_error = _safe_int(entry_context.get("selected_count", 0))
                entry_stats["raw_candidate_count_sum"] += raw_count
                entry_stats["active_candidate_count_sum"] += active_count
                entry_stats["ranked_candidate_count_sum"] += ranked_count
                entry_stats["selected_signal_count_sum"] += selected_count
                entry_stats["metrics_cast_error_count"] += int(
                    raw_cast_error or active_cast_error or ranked_cast_error or selected_cast_error
                )
            if new_entry_signals:
                for signal in new_entry_signals:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler, i)
            # (2) 추가 매수 신호 생성 및 실행 (신규 진입으로 자금/슬롯이 소진된 후의 상태 기준)
            additional_buy_signals = self.strategy.generate_additional_buy_signals(current_date, self.portfolio, self.data_handler, trading_dates, i)
            if additional_buy_signals:
                for signal in additional_buy_signals:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler, i)

            # --- 2. 일별 포트폴리오 가치 및 상태 기록 ---
            total_value = self.portfolio.get_total_value(current_date, self.data_handler)
            self.portfolio.record_daily_snapshot(current_date, total_value)
            capture_positions_snapshot = bool(
                getattr(self.portfolio, "capture_daily_positions_snapshot", False)
            )
            positions_df = None
            if capture_positions_snapshot or self.logger.isEnabledFor(logging.DEBUG):
                positions_df = self.portfolio.get_positions_snapshot(
                    current_date,
                    self.data_handler,
                    total_value,
                )
                if capture_positions_snapshot:
                    self.portfolio.record_positions_snapshot(current_date, positions_df)

            # --- 3. [신규 로직] 당일 발생한 모든 거래에 최종 포트폴리오 가치 업데이트 ---
            num_trades_after = len(self.portfolio.trade_history)
            if num_trades_after > num_trades_before:
                for i in range(num_trades_before, num_trades_after):
                    self.portfolio.trade_history[i].total_portfolio_value = total_value

            # --- 4. [핵심] 일일 포트폴리오 스냅샷 로그 출력 ---
            # 상세 스냅샷은 DEBUG에서만 생성/출력합니다(기본 INFO에서는 tqdm 진행바만 유지).
            if total_value > 0 and self.logger.isEnabledFor(logging.DEBUG):
                stock_value = total_value - self.portfolio.cash
                # total_value가 0이 되는 엣지 케이스 방지
                cash_ratio = (self.portfolio.cash / total_value) * 100 if total_value else 0
                stock_ratio = (stock_value / total_value) * 100 if total_value else 0

                header = f"\n{'='*120}\n"
                footer = f"\n{'='*120}"

                date_str = pd.to_datetime(current_date).strftime("%Y-%m-%d")
                summary_str = (
                    f"Date: {date_str} | Day {i+1}/{len(trading_dates)}\n"
                    f"{'-'*120}\n"
                    f"Total Value: {total_value:,.0f} | "
                    f"Cash: {self.portfolio.cash:,.0f} ({cash_ratio:.1f}%) | "
                    f"Stocks: {stock_value:,.0f} ({stock_ratio:.1f}%)\n"
                    f"Holdings Count: {len(self.portfolio.positions)} Stocks"
                )

                log_message = header + summary_str

                if positions_df is None:
                    positions_df = self.portfolio.get_positions_snapshot(
                        current_date,
                        self.data_handler,
                        total_value,
                    )
                if not positions_df.empty:
                    positions_df["Avg Buy Price"] = positions_df["Avg Buy Price"].map("{:,.0f}".format)
                    positions_df["Current Price"] = positions_df["Current Price"].map("{:,.0f}".format)
                    positions_df["Unrealized P/L"] = positions_df["Unrealized P/L"].map("{:,.0f}".format)
                    positions_df["Total Value"] = positions_df["Total Value"].map("{:,.0f}".format)
                    positions_df["P/L Rate"] = positions_df["P/L Rate"].map("{:.2%}".format)
                    positions_df["Weight"] = positions_df["Weight"].map("{:.2%}".format)

                    log_message += f"\n{'-'*120}\n[Current Holdings]\n"
                    log_message += positions_df.to_string()

                log_message += footer
                self.logger.debug(log_message)

        opportunity_days = entry_stats["entry_opportunity_days"]
        empty_entry_rate = (
            float(entry_stats["empty_entry_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        tier1_coverage = (
            float(entry_stats["tier1_source_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        tier2_fallback_rate = (
            float(entry_stats["tier2_fallback_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        tier2_blocked_rate = (
            float(entry_stats["tier2_blocked_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        no_candidates_rate = (
            float(entry_stats["no_candidates_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        no_signal_date_rate = (
            float(entry_stats["no_signal_date_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        source_lookup_error_rate = (
            float(entry_stats["lookup_error_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        source_missing_rate = (
            float(entry_stats["source_missing_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        source_unknown_rate = (
            float(entry_stats["unknown_source_days"]) / float(opportunity_days)
            if opportunity_days > 0
            else 0.0
        )
        candidate_eval_days = int(entry_stats["candidate_eval_days"])
        avg_raw_candidates = (
            float(entry_stats["raw_candidate_count_sum"]) / float(candidate_eval_days)
            if candidate_eval_days > 0
            else 0.0
        )
        avg_active_candidates = (
            float(entry_stats["active_candidate_count_sum"]) / float(candidate_eval_days)
            if candidate_eval_days > 0
            else 0.0
        )
        avg_ranked_candidates = (
            float(entry_stats["ranked_candidate_count_sum"]) / float(candidate_eval_days)
            if candidate_eval_days > 0
            else 0.0
        )
        avg_selected_signals = (
            float(entry_stats["selected_signal_count_sum"]) / float(candidate_eval_days)
            if candidate_eval_days > 0
            else 0.0
        )
        self.last_run_metrics = {
            "entry_opportunity_days": int(opportunity_days),
            "candidate_eval_days": candidate_eval_days,
            "empty_entry_days": int(entry_stats["empty_entry_days"]),
            "empty_entry_day_rate": round(empty_entry_rate, 4),
            "tier1_coverage": round(tier1_coverage, 4),
            "tier2_fallback_rate": round(tier2_fallback_rate, 4),
            "tier2_blocked_rate": round(tier2_blocked_rate, 4),
            "no_candidates_rate": round(no_candidates_rate, 4),
            "no_signal_date_rate": round(no_signal_date_rate, 4),
            "source_lookup_error_days": int(entry_stats["lookup_error_days"]),
            "source_lookup_error_rate": round(source_lookup_error_rate, 4),
            "source_missing_days": int(entry_stats["source_missing_days"]),
            "source_missing_rate": round(source_missing_rate, 4),
            "source_unknown_days": int(entry_stats["unknown_source_days"]),
            "source_unknown_rate": round(source_unknown_rate, 4),
            "metrics_cast_error_count": int(entry_stats["metrics_cast_error_count"]),
            "avg_raw_candidates": round(avg_raw_candidates, 2),
            "avg_active_candidates": round(avg_active_candidates, 2),
            "avg_ranked_candidates": round(avg_ranked_candidates, 2),
            "avg_selected_signals": round(avg_selected_signals, 2),
        }
        setattr(self.portfolio, "run_metrics", self.last_run_metrics)
        self.logger.info("[EntryMetrics] %s", self.last_run_metrics)
        self.logger.info("백테스팅이 완료되었습니다.")
        return self.portfolio
