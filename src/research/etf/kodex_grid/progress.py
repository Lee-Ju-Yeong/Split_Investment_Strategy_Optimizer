"""공개 진행 로그. 시각·단계·단위 이름·기술 상태·소요시간(오류면 기술 오류 코드)만 쓴다.
남은 시간은 이 로그의 기술적 완료시간만으로 다시 잡을 수 있다."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import re


DONE, SKIPPED, ERROR = "done", "skipped", "error"
LINE = re.compile(r"^(?P<at>\S+) (?P<stage>\S+) (?P<unit>\S+) (?P<status>done|skipped|error) "
                  r"(?P<seconds>\d+\.\d)s(?: (?P<code>E_[A-Z]+))?$")
ASSUMPTION = ("끝난 단위의 평균 실행 시간이 남은 단위에도 이어지고 동시 실행한 작업이 서로 느려지지 않는다는 가정의 추정이다. "
              "상한이 아니다.")


class Progress:
    """진행 로그 한 줄: 수익률·매매 건수·판정 상태·순위·영역 선택은 쓰지 않는다."""

    def __init__(self, folder, stage, echo=print):
        self.path, self.stage, self.echo = Path(folder) / "progress.log", stage, echo

    def __call__(self, name, code, seconds) -> None:
        """code: done / skipped / 기술 오류 코드(E_…)."""
        stamp, failed = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), code not in (DONE, SKIPPED)
        line = f"{stamp} {self.stage} {name} {ERROR if failed else code} {seconds:.1f}s"
        line += f" {code}" if failed else ""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self.echo(line)


def read_log(path) -> list:
    """진행 로그의 줄들. 꼴이 다른 줄이 있으면 공개 로그가 아니므로 멈춘다."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        found = LINE.match(line)
        if found is None:
            raise ValueError("progress log has a line that is not a technical progress record")
        rows.append({**found.groupdict(), "seconds": float(found["seconds"])})
    return rows


def remaining(path, *, planned: int, workers: int) -> dict:
    """기술적 완료시간만으로 잡은 남은 시간(조건부 추정)."""
    rows = read_log(path)
    latest = {row["unit"]: row["status"] for row in rows}
    seconds = [row["seconds"] for row in rows if row["status"] == DONE]
    left = planned - sum(status in (DONE, SKIPPED) for status in latest.values())
    mean = sum(seconds) / len(seconds) if seconds else None
    hours = None if mean is None else math.ceil(left / workers) * mean / 3600
    return {"planned": planned, "finished": planned - left, "remaining": left,
            "in_error": sorted(unit for unit, status in latest.items() if status == ERROR),
            "measured_units": len(seconds), "mean_done_seconds": mean,
            "max_done_seconds": max(seconds) if seconds else None, "workers": workers,
            "remaining_hours": hours, "assumption": ASSUMPTION}
