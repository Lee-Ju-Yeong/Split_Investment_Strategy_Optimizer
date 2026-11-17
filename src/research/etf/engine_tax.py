"""합성 자료용 ETF 세무 원장. 실제 자료의 세후 판정을 허용하는 모듈이 아니다.

계산 정의는 docs/research/etf_tax_rules.md 2.2, 3절의 명시 가정을 따른다.
국내 정식 분류는 metadata.py의 ``국내주식형`` / ``기타``이며 인정 별칭은
KR_TAX_CATEGORY_ALIASES에 열거한다. 미국 ETF는 회사형 양도소득이라는 연구 가정이다.
국내 실제 가격 항도 이동평균, 분배 때 실제 가격 항은 유지, 외국납부세액 공제와
금융소득종합과세 추가세액은 제외한다. 이 해석들을 확정 법률로 취급하지 않는다.

buy/sell은 결제 사건에 도달한 뒤, 그때 알려진 결제일 세무환율로 호출한다.
buy의 settled_at은 해당 사건 시각이고 sell의 settled_at은 at과 같아야 한다.
미래 결제환율을 미리 넘기는 예약 API는 제공하지 않는다. 모든 사건 시각은 단조적이다.
국내 세액과 미국 연간 국세/지방세는 각각 원 미만 절사한다(연구 반올림 가정).
미국 분배 원천징수는 Decimal의 정확한 15%로 두며 센트 반올림을 하지 않는다.
valuation_summary의 total은 합성 세후 조정 보고값으로, 현재까지 실현된 진행연도
추정세도 포함한다. 원장은 확정 미납액과 보고용 추정세를 반환하며 미래 손익은 예측하지
않는다. 현금 준비금에 따른 매수 차단·부족 시 환전·강제 매도·종료 청산 곡선은 구현하지
않는다. 이 결과는 실제 세후 판정이나 완결된 세후 성과 보고의 승인을 뜻하지 않는다.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN

from .engine_types import Instrument, Position, aware, decimal

ZERO = Decimal("0")
ONE = Decimal("1")
KR_RATE = Decimal(".154")
US_DIVIDEND_RATE = Decimal(".15")
US_DEDUCTION = Decimal("2500000")
KST = timezone(timedelta(hours=9))
KR_TAX_CATEGORY_ALIASES = {
    "국내주식형": "국내주식형", "kr_equity": "국내주식형",
    "domestic_equity": "국내주식형", "기타": "기타",
    "kr_other": "기타", "domestic_other": "기타",
}
ASSUMPTIONS = (
    "2026-11-15 문서의 현행 고정 연구 모델; 실제 세후 판정 보류",
    "미국 ETF 회사형 양도소득 가정; 법적 형태별 분류 미확정",
    "같은 계좌 전체 미국 종목 합산, 다른 과세 주식소득 없음, 연 250만원 공제",
    "국내 기타 ETF 실제 가격/과표 이동평균 및 분배 미과세 손익 이월 해석",
    "국내 주식형 분배 전액 과세; 외국납부세액 공제/종합과세 추가세액 제외",
    "국내 세액/미국 연간 국세·지방세는 원 미만 절사; 미국 분배 세액은 반올림 없음",
    "납부 예정일은 휴일 미조정; 실제 납부일과 납부액은 호출자가 명시",
    "국내 기타 ETF 동일 종목 같은 날 복수 매도 합산은 미지원하여 세후에서 거부",
)


def _number(value, name, *, positive=False):
    if isinstance(value, float):
        raise ValueError(f"{name}: float 대신 Decimal/문자/정수를 사용하세요")
    result = decimal(value)
    if result < ZERO or (positive and result == ZERO):
        raise ValueError(f"{name}: {'양수' if positive else '0 이상'}여야 합니다")
    return result


def _won(value):
    return value.quantize(ONE, rounding=ROUND_DOWN)


def _year(at):
    return at.astimezone(KST).year


@dataclass
class _Holding:
    instrument: Instrument
    quantity: int = 0
    price_cost: Decimal = ZERO
    foreign_cost: Decimal = ZERO
    krw_cost: Decimal = ZERO
    effective_basis: Decimal | None = None


class TaxLedger:
    """계좌 하나의 원장. pre_tax는 기록만 남기고 현금 세금/미납액은 0이다.

    advance(at)는 이전 과세연도의 미국 세액을 확정하고 진행 연도 추정세는 별도
    보고한다. 원장은 현금을 준비금으로 묶거나 매수를 차단하지 않는다. pay_tax는
    KRW 현금을 직접 바꾸지 않으므로 호출자가 반환된 배분만큼 현금에서 차감해야 한다.
    snapshot은 복사본이다.
    """

    def __init__(self, mode="pre_tax", *, synthetic=False):
        if mode not in {"pre_tax", "after_tax"}:
            raise ValueError("mode는 pre_tax/after_tax여야 합니다")
        if type(synthetic) is not bool or (mode == "after_tax" and not synthetic):
            raise ValueError("after_tax는 synthetic=True인 완전한 합성 자료만 허용합니다")
        self.mode, self.synthetic = mode, synthetic
        self._at = None
        self._holdings, self._lots, self._seen_buys = {}, {}, set()
        self._realized, self._financial, self._annual = {}, {}, {}
        self._receivables, self._distribution_keys = {}, set()
        self._daily_sales = set()
        self._events, self._payments, self._missing = [], [], set()

    def _time(self, at):
        aware(at)
        if self._at is not None and at < self._at:
            raise ValueError("세금 원장 시간을 과거로 되돌릴 수 없습니다")
        return at

    def _instrument(self, instrument):
        if not isinstance(instrument, Instrument):
            raise ValueError("Instrument가 필요합니다")
        holding = self._holdings.get(instrument.symbol)
        if holding is not None and holding.instrument != instrument:
            raise ValueError("같은 종목의 시장·통화·과세분류를 바꿀 수 없습니다")
        category = KR_TAX_CATEGORY_ALIASES.get(instrument.tax_category)
        if self.mode == "after_tax" and instrument.market == "KR" and category is None:
            raise ValueError("국내 ETF tax_category는 국내주식형/기타 또는 인정 별칭이어야 합니다")
        if self.mode == "after_tax" and instrument.price_basis not in {"raw", "split_adjusted", "split-adjusted"}:
            raise ValueError("세후 원장은 분배금을 반영하지 않은 raw/split_adjusted 가격만 허용합니다")
        return category

    def _basis(self, instrument, tax_basis):
        if tax_basis is not None:
            return _number(tax_basis, "tax_basis")
        if instrument.market == "KR" and self._instrument(instrument) == "기타":
            if self.mode == "after_tax":
                raise ValueError("국내 기타 ETF의 과표기준가가 필요합니다")
        return None

    def _taxes(self, year):
        base = max(ZERO, self._realized.get(year, ZERO) - US_DEDUCTION)
        return base, _won(base * Decimal(".20")), _won(base * Decimal(".02"))

    def advance(self, at):
        """시각을 전진하고 이전 연도 세액만 확정한다. 예정일은 휴일 미조정이다."""
        self._time(at)
        for year in sorted(self._realized):
            if year < _year(at) and year not in self._annual:
                base, income, local = self._taxes(year)
                self._annual[year] = {
                    "taxable_krw": base, "income_tax_krw": income, "local_tax_krw": local,
                    "unpaid_income_krw": income if self.mode == "after_tax" else ZERO,
                    "unpaid_local_krw": local if self.mode == "after_tax" else ZERO,
                    "income_due_date": date(year + 1, 5, 31),
                    "local_due_date": date(year + 1, 7, 31), "holiday_adjusted": False,
                    "finalized_at": at,
                }
        self._at = at

    def buy(self, instrument, position, *, fx_rate=ONE, tax_basis=None, settled_at=None):
        """전략 차수는 중복 방지용으로만 보존하고 세무 원가는 종목별로 합친다."""
        category = self._instrument(instrument)
        if position.symbol != instrument.symbol:
            raise ValueError("상품과 보유 종목이 다릅니다")
        at = self._time(settled_at if settled_at is not None else position.opened_at)
        if at < position.opened_at:
            raise ValueError("결제는 매수 체결보다 빠를 수 없습니다")
        fx, basis = _number(fx_rate, "fx_rate", positive=True), self._basis(instrument, tax_basis)
        key = (position.symbol, position.level)
        identity = (*key, position.opened_at)
        if key in self._lots or identity in self._seen_buys:
            raise ValueError("같은 보유 차수를 중복 반영할 수 없습니다")
        self.advance(at)
        holding = self._holdings.setdefault(instrument.symbol, _Holding(instrument))
        self._add_cost(holding, position, fx, basis)
        self._lots[key] = (position, position.quantity)
        self._seen_buys.add(identity)
        if instrument.market == "KR" and (category is None or (category == "기타" and basis is None)):
            self._missing.add(f"{instrument.symbol}: 과세분류/과표기준가 미확보")
        self._events.append({"kind": "buy", "symbol": instrument.symbol, "at": at})

    @staticmethod
    def _add_cost(holding, position, fx, basis):
        old_quantity, quantity = holding.quantity, position.quantity
        if old_quantity == 0:
            holding.effective_basis = basis
        elif holding.effective_basis is not None and basis is not None:
            holding.effective_basis = ((holding.effective_basis * old_quantity + basis * quantity)
                                       / (old_quantity + quantity))
        else:
            holding.effective_basis = None
        holding.quantity += quantity
        holding.price_cost += position.principal
        holding.foreign_cost += position.principal + position.buy_fee
        holding.krw_cost += (position.principal + position.buy_fee) * fx

    def _sale_position(self, instrument, position):
        key = (instrument.symbol, position.level)
        if position.symbol != instrument.symbol or key not in self._lots:
            raise ValueError("원장에 없는 보유분을 매도할 수 없습니다")
        original, remaining = self._lots[key]
        if (position.opened_at, position.entry_price, position.buy_fee, position.take_rate) != (
                original.opened_at, original.entry_price, original.buy_fee, original.take_rate):
            raise ValueError("등록된 보유 원가·취득시각과 매도 보유분이 다릅니다")
        if position.quantity > remaining:
            raise ValueError("보유 수량을 초과하여 매도할 수 없습니다")
        return key, remaining

    def sell(self, instrument, position, price, fee, at, *, fx_rate=ONE,
             tax_basis=None, settled_at=None):
        """즉시 원천징수 세액(자기 통화)을 반환한다. 미국 양도세는 반환하지 않는다."""
        category = self._instrument(instrument)
        self._time(at)
        if settled_at is not None and aware(settled_at) != at:
            raise ValueError("sell은 실제 결제 사건 시각 at=settled_at에 호출하세요")
        if at < position.opened_at:
            raise ValueError("매수보다 앞선 매도입니다")
        price, fee = _number(price, "price", positive=True), _number(fee, "fee")
        fx, basis = _number(fx_rate, "fx_rate", positive=True), self._basis(instrument, tax_basis)
        key, remaining = self._sale_position(instrument, position)
        if fee > price * position.quantity:
            raise ValueError("매도 수수료가 매도 대금을 초과합니다")
        day_key = (instrument.symbol, at.astimezone(KST).date())
        if self.mode == "after_tax" and instrument.market == "KR" and category == "기타" and day_key in self._daily_sales:
            raise ValueError("같은 날 복수 국내 기타 ETF 매도 합산 입력은 아직 지원하지 않습니다")
        self.advance(at)
        if instrument.market == "KR" and category == "기타" and basis is None:
            self._missing.add(f"{instrument.symbol}: 매도 과표기준가 미확보")
        holding = self._holdings[instrument.symbol]
        withholding = self._sell_tax(holding, position.quantity, price, fee, fx, basis, at)
        self._remove_cost(holding, position.quantity)
        if remaining == position.quantity:
            del self._lots[key]
        else:
            self._lots[key] = (self._lots[key][0], remaining - position.quantity)
        self._daily_sales.add(day_key)
        return withholding

    def _sell_tax(self, holding, quantity, price, fee, fx, basis, at):
        instrument = holding.instrument
        event = {"kind": "sell", "symbol": instrument.symbol, "at": at, "taxable": ZERO}
        withholding = ZERO
        if instrument.market == "US":
            gain = (price * quantity - fee) * fx - holding.krw_cost * quantity / holding.quantity
            self._realized[_year(at)] = self._realized.get(_year(at), ZERO) + gain
            event["realized_krw"] = gain
        elif instrument.market == "KR" and KR_TAX_CATEGORY_ALIASES.get(instrument.tax_category) == "기타":
            if basis is not None and holding.effective_basis is not None:
                actual_gain = price - holding.price_cost / holding.quantity
                taxable = max(ZERO, min(actual_gain, basis - holding.effective_basis)) * quantity
                event["taxable"] = taxable
                self._financial[_year(at)] = self._financial.get(_year(at), ZERO) + taxable
                withholding = _won(taxable * KR_RATE)
        event["withholding"] = withholding if self.mode == "after_tax" else ZERO
        self._events.append(event)
        return event["withholding"]

    @staticmethod
    def _remove_cost(holding, quantity):
        for name in ("price_cost", "foreign_cost", "krw_cost"):
            total = getattr(holding, name)
            setattr(holding, name, total - total * quantity / holding.quantity)
        holding.quantity -= quantity
        if holding.quantity == 0:
            holding.effective_basis = None

    def accrue_dividend(self, instrument, gross, ex_at, pay_at, *, tax_basis=None, dividend_id=None):
        """배당락 보유 수량으로 미수금 확정. 기타 ETF의 이월 기준도 이때 갱신한다."""
        self._instrument(instrument)
        self._time(ex_at)
        if aware(pay_at) < ex_at:
            raise ValueError("분배금 지급일은 배당락일보다 빠를 수 없습니다")
        gross = _number(gross, "gross")
        basis = self._basis(instrument, tax_basis)
        holding = self._holdings.get(instrument.symbol)
        if holding is None or holding.quantity == 0:
            raise ValueError("배당락 시점 보유 수량이 필요합니다")
        if instrument.market == "GOLD" and gross != ZERO:
            raise ValueError("KRX 금현물에는 ETF 분배금을 적용하지 않습니다")
        identity = (instrument.symbol, ex_at, pay_at)
        receipt_id = dividend_id if dividend_id is not None else f"dividend-{len(self._receivables) + 1}"
        if not isinstance(receipt_id, str) or not receipt_id or receipt_id in self._receivables:
            raise ValueError("분배금 식별자가 비어 있거나 중복입니다")
        if identity in self._distribution_keys:
            raise ValueError("같은 배당락·지급 사건을 중복 반영할 수 없습니다")
        tax, taxable, next_basis = self._dividend_tax(holding, gross, basis)
        self.advance(ex_at)
        if instrument.market == "KR" and KR_TAX_CATEGORY_ALIASES.get(instrument.tax_category) == "기타" and basis is None:
            self._missing.add(f"{instrument.symbol}: 분배 직전 과표기준가 미확보")
        holding.effective_basis = next_basis
        applied = tax if self.mode == "after_tax" else ZERO
        self._receivables[receipt_id] = {
            "id": receipt_id, "symbol": instrument.symbol, "currency": instrument.currency,
            "ex_at": ex_at, "pay_at": pay_at, "quantity": holding.quantity,
            "gross": gross, "taxable": taxable, "withholding": applied,
            "net": gross - applied, "paid_at": None,
        }
        self._distribution_keys.add(identity)
        return receipt_id

    def _dividend_tax(self, holding, gross, basis):
        instrument, next_basis = holding.instrument, holding.effective_basis
        if instrument.market == "US":
            return gross * US_DIVIDEND_RATE, gross, next_basis
        if instrument.market == "GOLD":
            return ZERO, ZERO, next_basis
        if KR_TAX_CATEGORY_ALIASES.get(instrument.tax_category) == "기타":
            if basis is None or next_basis is None:
                return ZERO, ZERO, None
            per_share = gross / holding.quantity
            taxable_per_share = max(ZERO, min(per_share, basis - next_basis))
            taxable = taxable_per_share * holding.quantity
            return _won(taxable * KR_RATE), taxable, next_basis - per_share + taxable_per_share
        return _won(gross * KR_RATE), gross, next_basis

    def dividend(self, instrument, gross, at, *, receivable_id=None, fx_rate=ONE, tax_basis=None):
        """지급일 원천징수. 미수금 없이 호출하면 배당락=지급일인 합성 사례다."""
        self._instrument(instrument)
        self._time(at)
        gross, fx = _number(gross, "gross"), _number(fx_rate, "fx_rate", positive=True)
        if receivable_id is None:
            receivable_id = self.accrue_dividend(instrument, gross, at, at, tax_basis=tax_basis)
        if receivable_id not in self._receivables:
            raise ValueError("등록되지 않은 미수 분배금입니다")
        receipt = self._receivables[receivable_id]
        if receipt["symbol"] != instrument.symbol or receipt["gross"] != gross:
            raise ValueError("미수 분배금의 종목·금액이 다릅니다")
        if receipt["paid_at"] is not None or at < receipt["pay_at"]:
            raise ValueError("중복 지급이거나 아직 지급일이 아닙니다")
        self.advance(at)
        receipt["paid_at"] = at
        krw_gross = gross * fx if instrument.market == "US" else gross
        self._financial[_year(at)] = self._financial.get(_year(at), ZERO) + krw_gross
        self._events.append({"kind": "dividend", "at": at, "receivable_id": receivable_id,
                             "gross": gross, "withholding": receipt["withholding"]})
        return receipt["withholding"]

    def pay_tax(self, amount_krw, at):
        """확정 미국 미납세액을 납기순으로 납부하고 배분 내역을 반환한다(원화)."""
        self._time(at)
        amount = _number(amount_krw, "amount_krw")
        available = self._unpaid()
        if self.mode == "after_tax":
            available += sum((sum(self._taxes(year)[1:], ZERO) for year in self._realized
                              if year < _year(at) and year not in self._annual), ZERO)
        if amount > available:
            raise ValueError("확정된 미납 세액보다 많은 금액을 낼 수 없습니다")
        self.advance(at)
        remaining, allocations = amount, []
        schedule = sorted((row[f"{part}_due_date"], year, part) for year, row in self._annual.items()
                          for part in ("income", "local"))
        for _, year, part in schedule:
            row, key = self._annual[year], f"unpaid_{part}_krw"
            paid = min(remaining, row[key])
            if paid:
                row[key] -= paid
                remaining -= paid
                allocations.append({"year": year, "component": part, "amount_krw": paid})
        self._payments.append({"at": at, "amount_krw": amount, "allocations": allocations})
        return deepcopy(allocations)

    def _unpaid(self):
        return sum((row["unpaid_income_krw"] + row["unpaid_local_krw"]
                    for row in self._annual.values()), ZERO)

    def valuation_summary(self) -> dict[str, Decimal]:
        """현재 원장의 평가용 부채만 읽는다. 시각 전진이나 전체 원장 복사는 없다.

        estimated는 진행연도에 이미 실현된 손익만 현재 세율·공제로 계산하므로
        이후 같은 해 손실이 생기면 줄어든다. total은 확정 미납액과 이 추정액의
        합인 조정 세후 보고값이다. 납부 때 줄어든 현금과 부채를 함께 반영하면
        이 조정 평가액이 연속된다. 이 메서드는 현금·예산·매수 가능액을 변경하지
        않으며 준비금에 따른 매수 차단을 실행하지 않는다.
        비용은 거래 사건 수와 무관하고 확정된 과세연도 수에만 비례한다.
        """
        unpaid, estimated = ZERO, ZERO
        if self.mode == "after_tax":
            unpaid = self._unpaid()
            if self._at is not None:
                _, income, local = self._taxes(_year(self._at))
                estimated = income + local
        return {"unpaid_tax_krw": unpaid, "estimated_tax_krw": estimated,
                "total_tax_liability_krw": unpaid + estimated}

    def snapshot(self, *, at=None):
        """과거로 되돌리지 않는다. at을 명시하면 advance와 같이 전진한다.

        tax_reserve_krw는 확정 미납액의 보고용 이름이며 실제 현금 구속이나
        준비금에 따른 매수 차단을 뜻하지 않는다.
        """
        if at is not None:
            self.advance(at)
        holdings = {symbol: {
            "quantity": row.quantity, "price_cost": row.price_cost,
            "foreign_cost": row.foreign_cost, "krw_cost": row.krw_cost,
            "average_price": row.price_cost / row.quantity if row.quantity else ZERO,
            "effective_tax_basis": row.effective_basis,
        } for symbol, row in self._holdings.items()}
        receivable_net = {"KRW": ZERO, "USD": ZERO}
        for row in self._receivables.values():
            if row["paid_at"] is None:
                receivable_net[row["currency"]] += row["net"]
        estimated = {year: {"taxable_krw": self._taxes(year)[0],
                            "income_tax_krw": self._taxes(year)[1],
                            "local_tax_krw": self._taxes(year)[2]}
                     for year in self._realized if year not in self._annual}
        return deepcopy({
            "mode": self.mode, "synthetic": self.synthetic, "actual_after_tax_approved": False,
            "as_of": self._at, "holdings": holdings, "us_realized_krw": self._realized,
            "estimated_us_tax": estimated, "annual_us_tax": self._annual,
            "unpaid_tax_krw": self._unpaid(), "tax_reserve_krw": self._unpaid(),
            "financial_income_krw": self._financial,
            "financial_income_over_threshold": [year for year, total in self._financial.items()
                                                if total > Decimal("20000000")],
            "receivables": self._receivables, "receivable_net": receivable_net,
            "payments": self._payments, "events": self._events,
            "missing_inputs": sorted(self._missing), "assumptions": ASSUMPTIONS,
        })
