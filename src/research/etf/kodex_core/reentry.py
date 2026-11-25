"""분할 장부 재진입 일정(리뷰 3절 표). R2 MonthlyEntry의 '바로 다음 달 한 번' 동작을 대체한다.

| 사건 | M | D |
| 마지막 분할 보유분 매도 체결 | 매도한 달의 다음 달 첫 국내 거래일 시가 | 매도 다음 국내 거래일 시가 |
| 예정 재진입 불성립(비용 조건·현금 부족·거래정지·거절) | 시도한 달의 다음 달 첫 거래일 | 시도 다음 거래일 |
| 1차 재진입 체결 | 대기 종료 | 대기 종료 |
코어 수량은 분할 엔진에 없으므로 전량 매도 판정에서 자동으로 제외된다.
"""
from __future__ import annotations

from ..engine import Reentry
from ..m5_parameters import CostGuardRejection
from .calendar import KrCalendar


class ReentrySchedule:
    def __init__(self, mode: str, calendar: KrCalendar):
        if mode not in {"M", "D"}:
            raise ValueError("reentry mode must be M or D")
        self.mode, self.calendar = mode, calendar
        self.anchor = self.target = None
        self.guard = None  # 엔진이 현재 변동성으로 비용 조건을 검사하는 함수를 연결한다.
        self.attempts: list[dict] = []

    def next_after(self, day):
        if self.mode == "M":
            return self.calendar.first_session_next_month(day)
        return self.calendar.next_session(day)

    def __call__(self, instrument, bar, last_exit):
        if last_exit != self.anchor:
            self.anchor, self.target = last_exit, self.next_after(last_exit)
        day = bar.session_date
        if self.target is None or day < self.target:
            return None
        if day > self.target:
            raise ValueError(f"scheduled reentry session {self.target} has no bar; judgment held")
        self.attempts.append({"date": day, "outcome": "submitted"})
        try:
            self.guard(instrument, bar)
        except CostGuardRejection as error:
            self.failed(day, f"cost_guard: {error}")
            return Reentry(unavailable=str(error))
        return Reentry()

    def failed(self, day, reason):
        if not self.attempts or self.attempts[-1]["date"] != day:
            raise RuntimeError("reentry failure without a scheduled attempt")
        self.attempts[-1]["outcome"] = reason
        self.target = self.next_after(day)

    def filled(self, day):
        self.attempts[-1]["outcome"] = "filled"
        self.target = None
