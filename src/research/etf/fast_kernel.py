"""M3 C++ 매매 후보 계산. 현금·세금은 M2 원장을 그대로 사용한다."""
from __future__ import annotations

import ctypes
from datetime import timedelta
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from . import engine_strategy as reference
from .engine import run_engine
from .engine_market import _RULES, _period
from .engine_types import Candidate, ZERO
from .fast_build import ABI_VERSION, LIBRARY, SOURCE, fingerprint

ROUNDING_MODES = ("ROUND_HALF_EVEN", "ROUND_UP", "ROUND_DOWN", "ROUND_CEILING", "ROUND_FLOOR",
                  "ROUND_HALF_UP", "ROUND_HALF_DOWN", "ROUND_05UP")


@lru_cache(maxsize=4096)
def _priority(symbol, levels):
    ordered = sorted(levels, key=lambda level: sha256(f"{symbol}:{level}".encode()).digest())
    return {level: rank for rank, level in enumerate(ordered)}


def _ratio(value):
    return str(Fraction(value))


def _payload(context, nodes, held, rules, first_entry):
    commission = (Decimal(_RULES["commission_rates"][context.instrument.market])
                  if context.commission is None else context.commission)
    values = [reference._duration_microseconds(context.bar), context.penetration, context.slippage,
              int(first_entry), ROUNDING_MODES.index(getcontext().rounding), _ratio(commission),
              _ratio(context.fee("SELL", Decimal(1))), _ratio(context.tax)]
    values.extend(_ratio(price) for _, price in nodes)
    needs_ticks = bool(held) or context.slippage != 0
    bands = (_period(context.bar.session_date, _RULES["tick_rules"][context.instrument.market]["periods"], "호가")["bands"]
             if needs_ticks else [])
    values.append(len(bands))
    for band in bands:
        values.extend([int(band["below"] is None), _ratio(band["below"] or "0"), _ratio(band["tick"])])
    values.append(len(rules))
    for rule in rules:
        values.extend([rule.level, _ratio(rule.budget), _ratio(rule.drop_rate)])
    priority = _priority(context.instrument.symbol, tuple(position.level for position in held))
    values.append(len(held))
    for position in held:
        values.extend([position.level, priority[position.level], position.quantity,
                       _ratio(position.entry_price), _ratio(position.buy_fee), _ratio(position.take_rate)])
    return " ".join(map(str, values)).encode("ascii")


class FastKernel:
    """현재 소스와 일치하는 명시 빌드만 읽는다. 커널은 내부 상태 없이 공유할 수 있다."""

    def __init__(self, build_dir):
        directory = Path(build_dir).resolve()
        manifest = json.loads((directory / "build.json").read_text(encoding="utf-8"))
        if manifest["abi"] != ABI_VERSION or manifest["source_sha256"] != fingerprint(SOURCE):
            raise ValueError("커널 소스/ABI가 다릅니다. 현재 소스를 새 폴더에 빌드하세요")
        if manifest["library_sha256"] != fingerprint(directory / LIBRARY):
            raise ValueError("커널 바이너리 해시가 빌드 기록과 다릅니다")
        self.manifest = manifest
        self.library = ctypes.CDLL(str(directory / LIBRARY))
        self.library.etf_kernel_abi.restype = ctypes.c_int
        if self.library.etf_kernel_abi() != ABI_VERSION:
            raise ValueError("커널 ABI가 다릅니다")
        self.library.etf_candidate.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.library.etf_candidate.restype = ctypes.c_int

    def _compute(self, payload):
        output = ctypes.create_string_buffer(max(65536, len(payload) * 2))
        status = self.library.etf_candidate(payload, output, len(output))
        if status == 1:
            raise ValueError(f"M3 커널 입력/가격 오류: {output.value.decode('utf-8', errors='replace')}")
        if status:
            raise RuntimeError(f"M3 커널 실패({status}): {output.value.decode('utf-8', errors='replace')}")
        return output.value.decode("ascii")

    def candidate_for_bar(self, instrument, bar, positions, rules, *, path="O-H-L-C", first_entry=False,
                          penetration_ticks=1, slippage_ticks=0, commission_rate=None,
                          sec_rate_override=None, product_sell_tax_rate=ZERO):
        """M2 입력 검사를 공유하고 C++에서 독립적으로 매매·체결 후보를 계산한다."""
        if instrument.symbol != bar.symbol or type(first_entry) is not bool:
            raise ValueError("상품/일봉 종목 또는 first_entry가 잘못됐습니다")
        with localcontext() as arithmetic:
            arithmetic.prec = 80
            context = reference._options(instrument, bar, penetration_ticks, slippage_ticks,
                                         commission_rate, sec_rate_override, product_sell_tax_rate)
            nodes = reference._path_nodes(bar, path)
            held = reference._validated_positions(instrument, bar, positions, rules)
            if bar.volume == ZERO or bar.limit_locked or any(p.opened_at == bar.open_at for p in held):
                return None
            if not held and (not first_entry or not rules):
                return None
            return _decode(self._compute(_payload(context, nodes, held, rules, first_entry)), bar)


def _decode(result, bar):
    if result == "NONE":
        return None
    side, level, quantity, numerator, denominator, micros, reason = result.split()
    price = _exact_decimal(int(numerator), int(denominator))
    return Candidate(bar.symbol, side, int(level), int(quantity), price,
                     bar.open_at + timedelta(microseconds=int(micros)), reason)


def _exact_decimal(numerator, denominator):
    """유한소수인 체결가를 현재 Decimal 정밀도로 줄이지 않고 복원한다."""
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("커널이 유한소수가 아닌 가격을 반환했습니다")
    places = max(twos, fives)
    integer = numerator * 2 ** (places - twos) * 5 ** (places - fives)
    return Decimal((int(integer < 0), tuple(map(int, str(abs(integer)))), -places))


def run_fast_engine(kernel: FastKernel, *args, **kwargs):
    """M2와 같은 완전성 검사를 마친 입력을 사용한다. 실제 입력은 strict M1 연결부를 거친다."""
    if "_candidate_provider" in kwargs:
        raise TypeError("커널은 run_fast_engine의 kernel 인수로 지정하세요")
    return run_engine(*args, **kwargs, _candidate_provider=kernel.candidate_for_bar)
