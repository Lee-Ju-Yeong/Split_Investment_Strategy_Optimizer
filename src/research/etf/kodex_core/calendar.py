"""국내 거래일 달력. 평가·재진입·결제 일정은 실제 국내 세션만 사용한다(v1 5절)."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, time
from types import MappingProxyType
from zoneinfo import ZoneInfo

from ..engine_market import _RULES


KST = ZoneInfo("Asia/Seoul")
EVALUATION_KST = time(18)


@dataclass(frozen=True)
class KrCalendar:
    """times[day] = (open_utc, close_utc). 달력에 없는 날을 휴장으로 추정해 채우지 않는다.
    times는 생성 시 복사한 읽기 전용 사본이다(D1: 반환한 달력을 바꿔 다음 실행에 섞을 수 없다)."""

    times: dict
    sessions: tuple = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "times", MappingProxyType({day: tuple(pair) for day, pair in self.times.items()}))
        days = tuple(sorted(self.times))
        if not days or any(type(day) is not date for day in days):
            raise ValueError("KR calendar needs actual session dates")
        object.__setattr__(self, "sessions", days)

    def next_session(self, day: date) -> date | None:
        index = bisect_right(self.sessions, day)
        return self.sessions[index] if index < len(self.sessions) else None

    def first_session_next_month(self, day: date) -> date | None:
        first = date(day.year + day.month // 12, day.month % 12 + 1, 1)
        index = bisect_left(self.sessions, first)
        return self.sessions[index] if index < len(self.sessions) else None

    def first_session_of_year(self, year: int) -> date | None:
        index = bisect_left(self.sessions, date(year, 1, 1))
        if index < len(self.sessions) and self.sessions[index].year == year:
            return self.sessions[index]
        return None

    def missing_years(self, start: date, end: date) -> list[int]:
        """시작~만기 사이에 국내 거래일이 하나도 없는 연도(D5: 달력 연도 누락은 자료 보류)."""
        years = {day.year for day in self.window(start, end)}
        return [year for year in range(start.year, end.year + 1) if year not in years]

    def year_firsts(self, start: date, end: date) -> list[date]:
        """start 다음 해부터 각 해의 첫 국내 거래일(연 1회 변동성 재계산일)."""
        firsts = {}
        for day in self.window(start, end):
            firsts.setdefault(day.year, day)
        return [day for year, day in sorted(firsts.items()) if year > start.year]

    def offset(self, start: date, sessions: int) -> date:
        index = bisect_left(self.sessions, start)
        if index + sessions >= len(self.sessions) or self.sessions[index] != start:
            raise ValueError("start offset leaves the calendar or start is not a session")
        return self.sessions[index + sessions]

    def window(self, start: date, end: date) -> list[date]:
        return [day for day in self.sessions if start <= day <= end]

    def evaluation_times(self, start: date, end: date) -> list[datetime]:
        times = [datetime.combine(day, EVALUATION_KST, KST) for day in self.window(start, end)]
        if any(at <= self.times[day][1] for at, day in zip(times, self.window(start, end))):
            raise ValueError("evaluation time must follow the domestic close")
        return times

    def settlement_times(self) -> dict:
        return {day: pair[1] for day, pair in self.times.items()}


def tick_table_through() -> date:
    return date.fromisoformat(_RULES["tick_rules"]["KR"]["periods"][-1]["through"])


def tick_table_from() -> date:
    return date.fromisoformat(_RULES["tick_rules"]["KR"]["periods"][0]["from"])


def coverage_gaps(calendar: KrCalendar, end: date, *, tail: int = 2, start: date | None = None) -> list[str]:
    """호가표·달력 범위(시작~만기, T+2 결제용 뒤 두 거래일). 빈 목록이 아니면 판정 보류."""
    gaps = []
    if start is not None and start < tick_table_from():
        gaps.append(f"KR tick table starts {tick_table_from()} after {start}")
    if end > tick_table_through():
        gaps.append(f"KR tick table ends {tick_table_through()} before {end}")
    after = [day for day in calendar.sessions if day > end]
    if len(after) < tail:
        gaps.append(f"KR calendar lacks {tail} sessions after {end}")
    elif after[tail - 1] > tick_table_through():
        gaps.append(f"KR tick table ends {tick_table_through()} before T+{tail} settlement {after[tail - 1]}")
    return gaps
