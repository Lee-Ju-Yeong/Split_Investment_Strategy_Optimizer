"""ETF 분류 메타데이터 (M1 데이터 단계, 계획 3.2절 보류 분할과 7절 D1·D7).

입력:
- ChatGPT 연구 자료의 `etf_universe_with_price_audit_2026-11-15.csv`(국내 1,188·미국 5,787행, 현재 상장).
- 폐지 국내 ETF: `<MAGICSPLIT_ETF_DATA_ROOT>/kr_universe/kr_etf_history_scoped.csv`의 `delisted=True` 행.
  KRX 월말 스냅샷(UTF-8 BOM)에서 만든 표다. 이름은 폐지 직전 이름(`last_name`)을 쓰고, 마지막으로 보인
  스냅샷의 이름과 같은지 확인한다(다르면 멈춘다).
출력: `<MAGICSPLIT_ETF_DATA_ROOT>/metadata/etf_metadata.csv`.

폐지 ETF도 생존 ETF와 같은 `classify()`로 판정한다. 다만 원천에 없는 값 두 가지는 이렇게 채운다.
- 네이버 탭: 폐지 ETF는 네이버 목록에 없다. KRX 기초지수명으로 탭을 추정한다(`infer_tab_from_krx_index`,
  `naver_tab_source=krx_index_name`). 같은 추정을 생존 ETF 1,188개에 적용해 실제 탭과 비교하면 자산군이
  1,181개 같고 D1 판정은 모두 같았다(2026-11-15). 다른 7개 중 4개는 기초지수명이 없어 `기타`로 검토 표시된다.
- 이름 힌트: 이력표의 `flags`(별도 이름 규칙: inverse·lev2x·option)를 원천 힌트 자리에 넣는다. 규칙과
  어긋나면 생존 ETF와 똑같이 `*_hint_unresolved`로 D1에서 뺀다.
`listing_status`(listed/delisted)와 `has_price_history`는 가격 저장소 CURRENT 묶음의 inventory와 맞는지
확인한다(다르거나 묶음을 찾지 못하면 쓰지 않고 멈춘다). 그래서 가격 저장소를 먼저 돌린다.

판단은 상품명·운용사 분류(issuer_category)·네이버 ETF 탭 번호·기초지수명 규칙으로만 한다.
규칙이 애매하면 값을 추측해 채우지 않고 `needs_review=True`와 `review_reasons`를 남긴다.

인코딩
------
- 받은 CSV는 UTF-8(BOM 포함)이다. `encoding="utf-8-sig"`로 읽으면 국내 상품명 1,188개가 모두 정상이다.
  엑셀·PowerShell 기본 코드페이지로 열면 깨져 보일 수 있다.
- 국내 원본(`universe/raw/kr_naver.json`)은 CP949로 인코딩돼 있다. CSV 상품명은 이 원본을 CP949로
  해석한 값과 1,188개 모두 일치한다(2026-11-15 확인).
- 종목코드는 `0000D0`, `069500`처럼 앞자리 0과 영문을 포함하므로 반드시 문자열로 읽는다.

열 정의
-------
- `leverage_multiple`: 기초 대비 일간 목표 배수. 1, 2, 3, -1, -2, -3(폐지 국내 ETF에는 1.5도 있다).
  이름으로 확정할 수 없으면 빈 값. 국내 상장은 이름에 배수 숫자(`1.5X`, `2X`)가 있으면 그 값을 쓰고,
  숫자 없는 `레버리지`는 국내 규정상 2배, `인버스`는 -1배로 본다(`인버스2X`는 -2배). 미국은 ProShares 이름 규칙
  (Ultra 2, UltraPro 3, Short -1, UltraShort -2, UltraPro Short -3)과 일반 `2X/3X` 표기를 쓴다.
- `is_inverse`, `is_etn`, `is_options_strategy`(커버드콜·버퍼·바이라이트·목표헤지 등),
  `is_futures_based`(**이름·운용사 분류에 선물 근거가 있을 때만 True**. False는 "근거 없음"이지
  "선물을 쓰지 않음"의 확인이 아니다), `is_single_stock`(국내 `단일종목` 상품).
- `asset_class` (판단 대상은 상장 시장이 아니라 **노출 자산**):
  - 국내주식: 한국 주식(지수·업종·테마·단일종목). 미국 상장 한국 ETF(EWY)도 포함.
  - 해외주식: 한국 외 주식(미국 주식 포함). 금광·원자재 생산기업 같은 주식형 테마 포함.
  - 국내외주식: 이름에 한국과 해외 주식을 함께 담는다고 드러난 상품(한중·한미·한국대만 등).
  - 채권: 국채·회사채·종합채권·물가채·하이일드·만기매칭형 등(아래 단기금리 제외).
  - 단기금리: 머니마켓, CD·KOFR·SOFR 금리, 초단기·단기 국채·통안채, T-bill, 변동금리·초단기 채권.
  - 금: 금 현물·선물·금 2배(금광 주식은 해외주식).
  - 원자재: 금 외 원자재(은·구리·원유·천연가스·농산물·팔라듐·탄소배출권·원자재 지수).
  - 통화: 환율 자체에 투자(달러선물, 엔선물, CurrencyShares, 달러 인덱스).
  - 리츠: 상장 리츠·부동산 주식(모기지 리츠 포함).
  - 우선증권: 우선주·하이브리드 증권(미국 상장 우선증권 ETF 등). 국내 `우선주` 지수는 국내주식.
  - 혼합: 주식+채권(채권혼합·TDF·TRF·자산배분·멀티에셋) 또는 주식+금 등 서로 다른 자산 결합.
  - 변동성: VIX 선물. 대체전략: 헤지펀드 복제·합병차익. 가상자산: 비트코인 등.
  - 기타: 규칙으로 정하지 못함(항상 needs_review).
- `region`: 노출 지역(한국, 미국, 중국, 일본, 선진국(미국 제외), 미국 외 글로벌, 글로벌 …).
- `market`: 상장 시장(KR/US).
- `index_family`: 같은 기초지수 또는 거의 같은 지수를 따르는 상품 묶음 ID. 국내·미국 상장을
  가로질러 같은 묶음을 쓴다(예: SPY·VOO·TIGER 미국S&P500·SSO·SPXU는 모두 `US_EQ_LARGE_BLEND`).
  같은 묶음에 넣는 변형: 환헤지(H)·합성·선물·TR·레버리지·인버스·액티브(같은 지수명)·커버드콜/버퍼
  (같은 기초지수)·ESG 배제형 변형, 그리고 채권혼합 상품은 주식 쪽 기초지수 묶음.
  시가총액 가중 광역지수는 거의 같은 지수로 본다(예: 한국 KOSPI·KOSPI200·KRX300·MSCI Korea,
  미국 S&P500·Russell 1000/3000·Total Market). 동일가중·팩터·배당·섹터 지수는 따로 둔다.
- `index_family_basis`: rule(명시 지수 규칙), theme(테마 키워드로 묶음 — 거의 같은 노출이라는 판단이
  들어감), name(정규화한 상품명 키 — 다른 상품과 이름 키가 같을 때만 묶임).
- `kr_tax_class`(국내 상장만): 국내주식형 / 기타. 근거는 `kr_tax_basis`.
  국내주식형 = 국내 주식 또는 국내 주가지수 장내파생(선물·옵션)만으로 노출을 만드는 상품(레버리지·
  인버스·커버드콜 포함). 해외주식·채권·혼합·원자재·금·통화·단기금리·국내외주식은 기타.
  합성(스왑)·단일종목 레버리지·국내 리츠는 분류가 불확실해 needs_review.
- `d1_eligible`, `d1_exclusion_reasons`: 계획 D1의 상품 성격 필터만 적용한다(인버스·ETN·옵션 전략·
  3배 이상 제외, 2배까지 포함, 배수 미확인 제외). 원천 이름 힌트(인버스·레버리지·옵션)는 1인데 규칙이 0이면
  규칙이 놓친 제외 대상일 수 있으므로 `*_hint_unresolved`로 제외하고 needs_review로 남긴다. 이력 3년·거래대금 조건은 가격 자료로 따로 계산한다.
  만기형(이름의 `26-04` 같은 만기)은 여기서 빼지 않는다.
- `listing_status`: listed(현재 상장) / delisted(폐지 국내 ETF). 미국은 현재 상장 목록뿐이다(생존 편향 한계).
- `naver_tab_source`: naver(네이버 목록의 탭) / krx_index_name(KRX 기초지수명으로 추정) / 빈 값(탭 없음).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from . import paths

UNIVERSE_FILENAME = "etf_universe_with_price_audit_2026-11-15.csv"
OUTPUT_FILENAME = "etf_metadata.csv"

# 네이버 ETF 목록 API의 etfTabCode (원본 raw/kr_naver.json, CSV의 raw_category_code)
NAVER_TABS = {
    "1": "국내 시장지수",
    "2": "국내 업종/테마",
    "3": "국내 파생",
    "4": "해외 주식",
    "5": "원자재",
    "6": "채권",
    "7": "기타",
}

REQUIRED_COLUMNS = (
    "market", "symbol", "name", "raw_category_code", "issuer", "issuer_category",
    "underlying_benchmark", "history_download_status", "explicit_etn_name",
    "leveraged_name_hint", "inverse_name_hint", "options_strategy_name_hint",
)

OUTPUT_COLUMNS = [
    "market", "symbol", "name", "issuer", "has_price_history", "naver_tab",
    "leverage_multiple", "is_inverse", "is_etn", "is_options_strategy", "is_futures_based",
    "is_single_stock", "asset_class", "region", "index_family", "index_family_basis",
    "kr_tax_class", "kr_tax_basis", "d1_eligible", "d1_exclusion_reasons",
    "needs_review", "review_reasons", "listing_status", "naver_tab_source",
]

KR_HISTORY_FILENAME = "kr_etf_history_scoped.csv"
KR_UNIVERSE_DIR = "kr_universe"
KR_SNAPSHOT_SUBDIR = "snapshots"
LISTED, DELISTED = "listed", "delisted"
# 이력표 flags(이름 규칙) → 원천 이름 힌트 자리
_FLAG_TO_HINT = {"lev2x": "leveraged_name_hint", "inverse": "inverse_name_hint", "option": "options_strategy_name_hint"}


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------
def load_universe(path: str | Path) -> pd.DataFrame:
    """유니버스 CSV를 문자열로 읽는다(앞자리 0 보존, UTF-8 BOM 처리)."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"유니버스 CSV에 필요한 열이 없습니다: {missing}")
    bad = df[df["name"].str.contains("�", regex=False)]
    if len(bad):
        raise ValueError(f"상품명에 대체 문자(U+FFFD)가 있습니다. 인코딩을 확인하세요: {bad['symbol'].tolist()[:5]}")
    dup = df[df.duplicated(["market", "symbol"], keep=False)]
    if len(dup):
        raise ValueError(f"(market, symbol) 중복: {dup[['market', 'symbol']].values.tolist()[:5]}")
    if df.empty:
        raise ValueError(f"유니버스 CSV에 데이터 행이 없습니다: {path}")
    return df


# 폐지 국내 ETF는 네이버 탭이 없다. KRX 기초지수명으로 네이버 탭에 해당하는 값을 추정한다.
# 순서: 채권·단기금리(6) → 원자재(5) → 해외 표시만 있음(4) → 국내 표시·한글 지수명·국내 지수 회사(2) → 모름("").
_IDX_BOND = re.compile(r"채권|국고채|국채|통안채|회사채|금융채|BOND|MONEY ?MARKET|금리|KOFR|CD\s?\d|단기|머니|TREASURY", re.I)
_IDX_COMMODITY = re.compile(r"GSCI|원유|(?<!철강)금속|GOLD|금현물|골드|은선물|구리|농산물|COMMODITY|WTI|CRUDE", re.I)
_IDX_KOREA = re.compile(r"KOREA|코리아|한국|KRX|코스피|코스닥|KOSPI|KOSDAQ|K-|^K|\bK[A-Z가-힣]", re.I)
_IDX_FOREIGN = re.compile(
    r"S&P|MSCI|NASDAQ|나스닥|DOW ?JONES|\bDJ\b|NYSE|HANG ?SENG|항셍|\bCSI\b|NIKKEI|니케이|TOPIX|STOXX|FTSE|RUSSELL|"
    r"BNY|INDXX|BLOOMBERG|MORNINGSTAR|NIFTY|미국|중국|차이나|일본|인도|베트남|글로벌|GLOBAL|\bU\.?S\.?\b|CHINA|"
    r"JAPAN|INDIA|EURO|유럽|선진|신흥|WORLD|ASIA|아시아|브라질|BRAZIL|BRIC|필리핀|PHILIPPINES", re.I)
_IDX_KR_VENDOR = re.compile(r"MKF|FNGUIDE|FNKOREA|WISE|ISELECT|AKROS|KEDI|DEEPSEARCH|NICE|KAP", re.I)


def infer_tab_from_krx_index(index_name: str) -> str:
    """KRX 기초지수명 → 네이버 탭 번호 추정("6" 채권, "5" 원자재, "4" 해외 주식, "2" 국내, "" 모름).

    1과 2(시장지수·업종테마)는 분류 규칙에서 차이가 없으므로 국내는 "2"로 둔다.
    """
    ix = (index_name or "").strip()
    if not ix:
        return ""
    if _IDX_BOND.search(ix):
        return "6"
    if _IDX_COMMODITY.search(ix):
        return "5"
    korea = bool(_IDX_KOREA.search(ix))
    if _IDX_FOREIGN.search(ix) and not korea:
        return "4"
    if korea or re.search(r"[가-힣]", ix) or _IDX_KR_VENDOR.search(ix):
        return "2"
    return ""


def _snapshot_names(snapshots_dir: Path, date: str) -> dict[str, str]:
    path = snapshots_dir / f"{date.replace('-', '')}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"폐지 ETF 이름 대조용 KRX 스냅샷이 없습니다: {path}")
    snap = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, usecols=["ISU_SRT_CD", "ISU_ABBRV"])
    return dict(zip(snap["ISU_SRT_CD"], snap["ISU_ABBRV"]))


def load_kr_delisted(history_path: str | Path, snapshots_dir: str | Path | None = None) -> pd.DataFrame:
    """폐지 국내 ETF를 유니버스 CSV와 같은 열로 만든다(`classify()` 입력).

    - 이력표는 UTF-8(BOM)로 엄격하게 읽는다. 이름에 대체 문자(U+FFFD)가 있으면 멈춘다.
    - `snapshots_dir`를 주면 마지막으로 보인 스냅샷의 이름(ISU_ABBRV)과 `last_name`이 같은지 확인한다.
    - `history_download_status`는 Naver 일봉이 있으면(`has_daily=True`) downloaded. 가격 저장소와의
      일치는 `check_price_bundle()`이 확인한다.
    """
    history = pd.read_csv(history_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    needed = {"code", "last_name", "index_name", "last_seen", "delisted", "flags", "has_daily"}
    missing = sorted(needed - set(history.columns))
    if missing:
        raise ValueError(f"폐지 이력표에 필요한 열이 없습니다: {missing}")
    if history["code"].duplicated().any():
        raise ValueError("폐지 이력표에 종목코드 중복이 있습니다")
    for column, allowed in (("delisted", {"True", "False"}), ("has_daily", {"True", "False"})):
        if not history[column].isin(allowed).all():
            raise ValueError(f"폐지 이력표 {column} 값은 True/False여야 합니다")
    delisted = history[history["delisted"] == "True"]
    if delisted.empty:
        raise ValueError(f"폐지 이력표에 폐지 ETF가 없습니다: {history_path}")
    bad = delisted[delisted["last_name"].str.contains("�", regex=False) | (delisted["last_name"].str.strip() == "")]
    if len(bad):
        raise ValueError(f"폐지 ETF 이름이 비었거나 대체 문자(U+FFFD)가 있습니다: {bad['code'].tolist()[:5]}")
    if snapshots_dir is not None:
        cache: dict[str, dict[str, str]] = {}
        for code, name, seen in delisted[["code", "last_name", "last_seen"]].itertuples(index=False):
            names = cache.setdefault(seen, _snapshot_names(Path(snapshots_dir), seen))
            if names.get(code) != name:
                raise ValueError(f"{code}: 이력표 이름({name!r})이 {seen} KRX 스냅샷 이름({names.get(code)!r})과 다릅니다")
    rows = []
    for item in delisted.to_dict("records"):
        flags = set(filter(None, item["flags"].split(";")))
        unknown = flags - set(_FLAG_TO_HINT) - {"term"}
        if unknown:
            raise ValueError(f"{item['code']}: 알 수 없는 flags {sorted(unknown)}")
        tab = infer_tab_from_krx_index(item["index_name"])
        row = {
            "market": "KR", "symbol": item["code"], "name": item["last_name"], "raw_category_code": tab,
            "issuer": "", "issuer_category": "", "underlying_benchmark": item["index_name"],
            "history_download_status": "downloaded" if item["has_daily"] == "True" else "no_naver_daily",
            "explicit_etn_name": "0", "listing_status": DELISTED,
            "naver_tab_source": "krx_index_name" if tab else "",
        }
        for flag, hint in _FLAG_TO_HINT.items():
            row[hint] = "1" if flag in flags else "0"
        rows.append(row)
    return pd.DataFrame(rows)


def combine_universe(listed: pd.DataFrame, delisted: pd.DataFrame) -> pd.DataFrame:
    """현재 상장 유니버스와 폐지 국내 ETF를 합친다. 같은 (market, symbol)이 양쪽에 있으면 멈춘다."""
    listed = listed.copy()
    listed["listing_status"] = LISTED
    both = pd.concat([listed, delisted], ignore_index=True).fillna("")
    dup = both[both.duplicated(["market", "symbol"], keep=False)]
    if len(dup):
        raise ValueError(f"현재 상장 목록과 폐지 목록에 같은 종목이 있습니다: {dup['symbol'].unique().tolist()[:5]}")
    return both


def check_price_bundle(meta: pd.DataFrame, bundle: Path) -> dict:
    """분류표의 `has_price_history`·`listing_status`가 가격 묶음 inventory와 같은지 확인한다(다르면 ValueError)."""
    result = {}
    for market in sorted(meta["market"].unique()):
        path = Path(bundle) / f"{market.lower()}_inventory.csv"
        if not path.is_file():
            raise ValueError(f"가격 묶음에 {market} inventory가 없습니다: {path}")
        inv = pd.read_csv(path, dtype={"symbol": str}, keep_default_na=False)
        status = dict(zip(inv["symbol"], inv["listing_status"])) if "listing_status" in inv else {}
        priced = meta[(meta["market"] == market) & meta["has_price_history"]]
        meta_set, inv_set = set(priced["symbol"]), set(inv["symbol"])
        if meta_set != inv_set:
            raise ValueError(
                f"{market}: has_price_history와 가격 묶음 종목이 다릅니다 "
                f"(분류표에만 {sorted(meta_set - inv_set)[:5]}, 가격에만 {sorted(inv_set - meta_set)[:5]})")
        if status:
            wrong = [s for s, st in zip(priced["symbol"], priced["listing_status"]) if status.get(s) != st]
            if wrong:
                raise ValueError(f"{market}: listing_status가 가격 묶음과 다릅니다: {wrong[:5]}")
        result[market] = len(meta_set)
    return result


# ---------------------------------------------------------------------------
# 이름 정규화
# ---------------------------------------------------------------------------
def kr_rest(name: str) -> str:
    """국내 상품명에서 브랜드(첫 단어)를 떼고 공백을 없앤 대문자 문자열."""
    parts = name.strip().split(" ", 1)
    rest = parts[1] if len(parts) == 2 else parts[0]
    return re.sub(r"\s+", "", rest).upper()


_KR_MIX_TOKENS = re.compile(
    r"미국30년국채혼합|미국채커버드콜밸런스|미국채타겟커버드콜혼합|미국채커버드콜혼합|커버드콜OTM채권혼합|"
    r"커버드콜채권혼합|위클리커버드콜채권혼합|채권혼합50|미국채혼합50|국채혼합50|채권혼합|미국채혼합|혼합50|혼합"
)
_KR_MODIFIERS = [
    r"\(합성H\)", r"\(합성\)", r"\(H\)",
    r"단일종목", r"레버리지", r"인버스2X", r"인버스", r"2X", r"(?<![\d.])[1-9]\.\d+X",  # 1.5X(폐지 HANARO)
    r"선물", r"액티브", r"적격", r"ENHANCED", r"엔화노출",
    r"(?:데일리|위클리|타겟|고정|변동성확대시|월간)*(?:타겟)?(?:데일리)?커버드콜(?:OTM|ATM)?",
    r"버퍼\d+월", r"월간목표헤지", r"\d호$",
    r"TR(?=$|[^A-Z])",
]
_KR_VENDOR = re.compile(r"SOLACTIVE|INDXX|FACTSET|ISELECT$")


def kr_base(rest: str) -> str:
    """혼합·파생·운용 방식 표시를 뗀 국내 상품명 핵심(기초지수 이름에 가까운 부분)."""
    s = _KR_MIX_TOKENS.sub("", rest)
    s = s.replace("플러스", "PLUS").replace("+", "PLUS")
    for _ in range(2):
        for pat in _KR_MODIFIERS:
            s = re.sub(pat, "", s)
    return s


_US_BRANDS = re.compile(
    r"^(STATE STREET SPDR |STATE STREET |SPDR |ISHARES |INVESCO |VANGUARD |SCHWAB |PROSHARES )"
)
_US_LEV_PREFIX = re.compile(r"^(ULTRAPRO SHORT|ULTRASHORT|ULTRAPRO|ULTRA|SHORT)\b\s*")


def us_core(name: str) -> str:
    """미국 상품명에서 운용사·형식 단어를 뗀 대문자 핵심. ProShares 배수 접두어도 뗀다."""
    s = re.sub(r"\s+", " ", name.upper()).strip()
    for _ in range(2):
        s = _US_BRANDS.sub("", s)
    if name.upper().startswith(("PROSHARES", "ULTRAPRO", "ULTRASHORT")):
        s = _US_LEV_PREFIX.sub("", s)
    s = re.sub(r"^(PORTFOLIO|CORE) ", "", s)
    s = re.sub(r" SELECT SECTOR SPDR ETF$", " SELECT SECTOR", s)
    s = re.sub(r"( EXCHANGE TRADED FUND| INDEX FUND| ETF TRUST| ETF SHARES| INDEX ETF| ETFO?| FUND| TRUST,? SERIES 1| TRUST SHARES| TRUST| SHARES| INDEX| PORTFOLIO)+$", "", s)
    s = re.sub(r" ETF$", "", s)
    return s.strip()


def _name_key(text: str) -> str:
    s = _KR_VENDOR.sub("", text)
    s = re.sub(r"[^0-9A-Z가-힣&+]", "", s.upper())
    return s


# ---------------------------------------------------------------------------
# 배수·인버스·ETN·옵션·선물
# ---------------------------------------------------------------------------
_US_OPTIONS = re.compile(
    # BUFFER는 접두어·숫자가 붙는 이름이 많아(Buffer10, Buffer100, Buffered, DynaBuffer) 단어 경계를 두지 않는다.
    r"COVERED CALL|BUY-?WRITE|BUFFER|PREMIUM INCOME|OPTION INCOME|\bOPTIONS?\b|AUTOCALLABLE|DEFINED OUTCOME|"
    r"EQUITY PREMIUM|\bCOLLAR\b|PUT ?WRITE|HIGH INCOME ETF|TARGET INCOME|YIELDMAX|\bFLOOR\b|\bBARRIER\b|0DTE"
)
_KR_OPTIONS = re.compile(r"커버드콜|버퍼|목표헤지|콜옵션|풋옵션")
_US_SHORT_NON_INVERSE = re.compile(r"SHORT[- ](TERM|DURATION|MATURITY|DATED)")
_KR_MULTIPLE = re.compile(r"(?<![\d.])([1-9](?:\.\d+)?)X(?![A-Z])")


def classify_leverage(market: str, name: str, issuer: str) -> tuple[float | None, bool, list[str]]:
    """(배수, 인버스 여부, 검토 사유)."""
    reasons: list[str] = []
    if market == "KR":
        rest = kr_rest(name)
        if re.search(r"롱.*숏", rest):
            reasons.append("롱숏 상품: 순 배수가 이름 규칙으로 정해지지 않음")
            return None, False, reasons
        # 이름의 배수 숫자가 우선이다(폐지 ETF 'HANARO 200선물레버리지1.5X'는 1.5배).
        m = _KR_MULTIPLE.search(rest)
        if "인버스" in rest:
            return -(float(m.group(1)) if m else 1.0), True, reasons
        if m:
            return float(m.group(1)), False, reasons
        if "레버리지" in rest:
            return 2.0, False, reasons
        return 1.0, False, reasons

    up = re.sub(r"\s+", " ", name.upper())
    is_proshares = issuer == "ProShares" or up.startswith(("PROSHARES", "ULTRAPRO "))
    if is_proshares:
        body = re.sub(r"^PROSHARES ", "", up)
        m = re.match(r"(ULTRAPRO SHORT|ULTRASHORT|ULTRAPRO|ULTRA|SHORT)\b", body)
        if "VIX" in body and m:
            reasons.append("ProShares VIX 레버리지·인버스 상품: 현재 배수가 이름 규칙(Ultra=2, Short=-1)과 다를 수 있어 운용사 자료 확인 필요")
            return None, m.group(1) != "ULTRA", reasons
        if "DECLINE OF" in body:
            return -1.0, True, reasons
        mult = {"ULTRAPRO SHORT": -3.0, "ULTRASHORT": -2.0, "SHORT": -1.0, "ULTRAPRO": 3.0, "ULTRA": 2.0}
        if m:
            v = mult[m.group(1)]
            return v, v < 0, reasons
        return 1.0, False, reasons

    # 일반 규칙(가격 자료가 없는 다른 운용사 상품에 주로 해당)
    if re.search(r"LONG/SHORT|LONG-SHORT", up):
        reasons.append("롱숏 전략: 순 노출 배수가 이름 규칙으로 정해지지 않음")
        return None, False, reasons
    inverse =bool(re.search(r"\b(INVERSE|BEAR|BEARISH)\b", up))
    short_word = bool(re.search(r"\bSHORT\b", up)) and not _US_SHORT_NON_INVERSE.search(up) \
        and not re.search(r"ULTRA[- ]SHORT|LONG/SHORT|LONG-SHORT", up)
    inverse = inverse or short_word
    m = re.search(r"(?<![\d.])-?([1-9](?:\.\d+)?)\s*[X×]\b", up) or re.search(r"\b([1-9](?:\.\d+)?)X\b", up)
    if m:
        mult = float(m.group(1))
        return (-mult if inverse else mult), inverse, reasons
    if short_word:
        reasons.append("이름의 Short가 인버스인지 단기 채권인지 확인 필요")
    if re.search(r"\bLEVERAGED\b", up):
        reasons.append("레버리지 표현은 있으나 배수 표기가 없음")
        return None, inverse, reasons
    return (-1.0 if inverse else 1.0), inverse, reasons


def is_etn(market: str, name: str, explicit_flag: str) -> bool:
    if explicit_flag == "1":
        return True
    if market == "US":
        return bool(re.search(r"\bETNS?\b|EXCHANGE[- ]TRADED NOTES?", name.upper()))
    return "ETN" in name.upper()


def is_options_strategy(market: str, name: str) -> bool:
    if market == "KR":
        return bool(_KR_OPTIONS.search(kr_rest(name)))
    return bool(_US_OPTIONS.search(name.upper()))


_US_FUTURES_NAME = re.compile(
    r"FUTURES|\bDB\b|OPTIMUM YIELD DIVERSIFIED COMMODITY|GSCI COMMODITY|ROLL SELECT COMMODITY|K-1 FREE CRUDE"
)


def is_futures_based(market: str, name: str, issuer: str) -> bool:
    if market == "KR":
        return "선물" in kr_rest(name)
    up = name.upper()
    if _US_FUTURES_NAME.search(up):
        return True
    # ProShares 원자재·통화 상품은 선물·선도로 노출을 만든다(이름에 원자재·통화명이 있음).
    if issuer == "ProShares" and re.search(r"\b(GOLD|SILVER|CRUDE OIL|NATURAL GAS|EURO|YEN)\b", up):
        return True
    return False


# ---------------------------------------------------------------------------
# 자산군·지역
# ---------------------------------------------------------------------------
_KR_FOREIGN = re.compile(
    r"미국|글로벌|차이나|중국|인도|일본|유럽|유로|베트남|대만|아시아|선진국|신흥국|이머징|라틴|멕시코|러시아|독일|DAX|"
    r"싱가포르|테슬라|엔비디아|팔란티어|애플|구글|마이크로소프트|브로드컴|TSMC|샤오미|BYD|알리바바|버크셔|스페이스X|"
    r"일라이릴리|S&P500|S&P글로벌|나스닥|항셍|토탈월드|빌리어네어|서학개미|MSCIEM|MSCI선진국|EM선물"
)
_KR_DOMESTIC_HINT = re.compile(r"코리아|한국|KOREA|^K-|코스닥")

# 국내 탭 1·2(국내)인데 이름만으로는 해외처럼 보이는 상품: 탭 판단을 따른다(근거 확인 결과).
_KR_DOMESTIC_OVERRIDE = {
    "150460": "TIGER 중국소비테마는 중국 소비 수혜 국내 주식 테마(네이버 탭 2 국내 업종/테마)",
}


def _kr_asset_class(rest: str, base: str, tab: str, reasons: list[str]) -> str:
    if re.search(r"TDF|TRF|자산배분|멀티에셋|주식혼합|채권혼합|미국채혼합|국채혼합|혼합50|커버드콜혼합|커버드콜밸런스|&GOLD$|주식&골드|인컴TOP10", rest) \
            or rest.endswith("혼합") or "리츠부동산인프라채권" in rest:
        if "인컴TOP10" in rest:
            reasons.append("ACE 글로벌인컴TOP10: 편입 자산 구성(주식·리츠·채권)이 이름만으로 불명확")
        return "혼합"
    if re.search(r"CD\d?년?금리|KOFR|SOFR|머니마켓|단기자금|단기채권|단기통안채|통안채1년|초단기|전단채|단기금융채|"
                 r"단기특수은행채|단기국공채|단기변동금리|달러단기채권|엔화초단기국채|CD금리", rest):
        return "단기금리"
    if re.search(r"달러선물|엔선물", rest):
        return "통화"
    if re.search(r"기업|채굴|생산기업", rest):
        return "해외주식" if _KR_FOREIGN.search(rest) else "국내주식"
    if "리츠" in rest or "부동산" in rest or "리얼티" in rest:
        if "리얼티인컴" in rest:
            reasons.append("RISE 글로벌리얼티인컴: 리츠 단일 종목·리츠 바스켓 여부를 이름만으로 확인할 수 없음")
        return "리츠"
    if re.fullmatch(r"(KRX)?금현물|골드|국제금|금|KRX금", base) or base.startswith(("골드", "금현물", "국제금")):
        return "금"
    # 금속: 폐지 TIGER 금속선물(H)(S&P GSCI Industrial Metals)
    if re.fullmatch(r"은|금은|구리|구리실물|금속|WTI원유|원유|3대농산물|농산물|콩|팔라듐", base) or "탄소배출권" in base:
        return "원자재"
    if "우선증권" in rest:
        return "우선증권"
    if re.search(r"채권|국고채|국채|국공채|회사채|금융채|은행채|특수채|통안채|물가채|하이일드|투자등급|\d\d-\d\d", rest) or tab == "6":
        return "채권"
    if re.search(r"한중|한국대만|한미|한국고배당&미국|한국형글로벌", rest):
        return "국내외주식"
    foreign = bool(_KR_FOREIGN.search(rest)) and not re.search(r"MSCIKOREA|S&P코리아", rest)
    if tab in ("1", "2", "3"):
        if foreign and _KR_DOMESTIC_HINT.search(rest):
            foreign = False  # 예: WON K-글로벌수급상위, KODEX 코스닥글로벌(국내 지수)
        if foreign:
            reasons.append(f"네이버 탭 {tab}({NAVER_TABS[tab]})과 이름의 해외 표시가 다름: 이름 기준 해외주식으로 분류")
            return "해외주식"
        return "국내주식"
    if tab == "4":
        return "해외주식"
    if foreign:
        return "해외주식"
    if tab in ("5", "7"):
        if re.search(r"AI|로봇|반도체|컬처|팔란티어", rest):
            reasons.append(f"네이버 탭 {tab}의 주식형 상품: 국내/해외 여부를 이름만으로 판단하기 어려움")
            return "기타"
        return "국내주식"
    return "기타"


_US_COUNTRIES = {
    "SOUTH KOREA": "한국", "KOREA": "한국", "BRAZIL": "브라질", "MEXICO": "멕시코", "CANADA": "캐나다",
    "AUSTRALIA": "호주", "GERMANY": "독일", "FRANCE": "프랑스", "UNITED KINGDOM": "영국", "BRITISH": "영국",
    "ITALY": "이탈리아", "SPAIN": "스페인", "SWITZERLAND": "스위스", "SWISS": "스위스", "NETHERLANDS": "네덜란드",
    "SWEDEN": "스웨덴", "DENMARK": "덴마크", "NORWAY": "노르웨이", "FINLAND": "핀란드", "BELGIUM": "벨기에",
    "AUSTRIA": "오스트리아", "IRELAND": "아일랜드", "ISRAEL": "이스라엘", "SINGAPORE": "싱가포르",
    "MALAYSIA": "말레이시아", "THAILAND": "태국", "INDONESIA": "인도네시아", "PHILIPPINES": "필리핀",
    "TAIWAN": "대만", "TURKEY": "튀르키예", "SOUTH AFRICA": "남아프리카공화국", "SAUDI ARABIA": "사우디아라비아",
    "QATAR": "카타르", "UAE": "아랍에미리트", "CHILE": "칠레", "PERU": "페루", "POLAND": "폴란드",
    "NEW ZEALAND": "뉴질랜드", "HONG KONG": "홍콩", "VIETNAM": "베트남", "ARGENTINA": "아르헨티나",
    "CANADIAN DOLLAR": "캐나다", "AUSTRALIAN DOLLAR": "호주", "SWISS FRANC": "스위스",
}


def _us_region(up: str) -> str:
    if re.search(r"EX[- ]JAPAN", up):
        return "아시아태평양"
    up = re.sub(r"EX[- ]CHINA", "", up)
    for word, region in sorted(_US_COUNTRIES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", up):
            return region
    if re.search(r"CHINA|CHINESE|GOLDEN DRAGON", up):
        return "중국"
    if re.search(r"JAPAN|NIKKEI|JPX|\bYEN\b", up):
        return "일본"
    if re.search(r"\bINDIA\b", up):
        return "인도"
    if re.search(r"EURO STOXX|EUROZONE|\bEURO\b|\bEMU\b", up):
        return "유로존"
    if re.search(r"EUROPE|EUROPEAN", up):
        return "유럽"
    if re.search(r"LATIN AMERICA", up):
        return "중남미"
    if re.search(r"EMERGING|\bEM\b|\bBIC\b", up):
        return "신흥국"
    if re.search(r"ASIA|PACIFIC", up):
        return "아시아태평양"
    if re.search(r"EAFE|DEVELOPED|KOKUSAI", up):
        return "선진국(미국 제외)" if "KOKUSAI" not in up else "글로벌"
    if re.search(r"EX[- ]U\.?S|INTERNATIONAL|\bINTL\b|ALL[- ]WORLD EX|GLOBAL EX", up):
        return "미국 외 글로벌"
    if re.search(r"ACWI|\bWORLD\b|GLOBAL|ALL[- ]WORLD", up):
        return "글로벌"
    return "미국"


_KR_REGIONS = [
    (r"한중", "한국+중국"), (r"한국대만", "한국+대만"), (r"한미|한국고배당&미국", "한국+미국"),
    (r"한국형글로벌", "한국+글로벌"),
    (r"아시아", "아시아"), (r"차이나|중국|항셍|CSI|과창판|심천|샤오미|BYD|알리바바", "중국"),
    (r"일본|TOPIX|니케이|NIKKEI|엔선물|엔화", "일본"), (r"인도네시아", "인도네시아"),
    (r"인도|NIFTY|니프티", "인도"), (r"유로스탁스", "유로존"), (r"독일|DAX", "독일"), (r"유럽", "유럽"),
    (r"베트남|VN30", "베트남"), (r"멕시코", "멕시코"), (r"러시아", "러시아"), (r"대만|TSMC", "대만"),
    (r"싱가포르", "싱가포르"), (r"라틴", "중남미"), (r"신흥국|이머징|MSCIEM", "신흥국"),
    (r"단기선진", "선진국(미국 제외)"),
    (r"미국|나스닥|S&P500|다우존스|러셀|필라델피아|테슬라|엔비디아|팔란티어|애플|구글|마이크로소프트|브로드컴|버크셔|"
     r"스페이스X|일라이릴리|빌리어네어|서학개미|달러|SOFR", "미국"),
    (r"토탈월드|글로벌|월드|선진국|MSCI선진국", "글로벌"),
]


def _kr_region(rest: str, asset_class: str) -> str:
    if asset_class in ("국내주식",):
        return "한국"
    if asset_class in ("금", "원자재") and not re.search(r"유럽", rest):
        return "글로벌"
    for pat, region in _KR_REGIONS:
        if re.search(pat, rest):
            return region
    if asset_class in ("채권", "단기금리", "혼합", "리츠"):
        return "한국"
    return "미확인"


def _us_asset_class(name: str, core: str, issuer_category: str) -> str:
    up = name.upper()
    cat = issuer_category.upper()
    if re.search(r"BITCOIN|ETHER(EUM)?\b|CRYPTO|SOLANA|\bXRP\b", up):
        return "가상자산"
    if "VIX" in up:
        return "변동성"
    if re.search(r"HEDGE REPLICATION|MERGER", up):
        return "대체전략"
    if re.search(r"CURRENCYSHARES|USD INDEX", up) or re.fullmatch(r"(EURO|YEN|YEN NEW)", core):
        return "통화"
    if re.search(r"\bGOLD\b", up) and not re.search(r"MINER|PRODUCER", up):
        return "금"
    if re.search(r"\bSILVER\b|CRUDE|NATURAL GAS|COMMODITY|\bOIL FUND\b|\bDB (AGRICULTURE|BASE METALS|ENERGY|PRECIOUS METALS|OIL)", up) \
            and not re.search(r"MINER|PRODUCER|EQUIPMENT|SERVICES|EXPLORATION", up):
        return "원자재"
    if re.search(r"ALLOCATION|MULTI-ASSET|BALANCED|CEF INCOME COMPOSITE|REAL RETURN", up) or "MULTI-ASSET" in cat or "BALANCED" in cat:
        return "혼합"
    if re.search(r"\bREIT\b|REAL ESTATE|REALTY", up):
        return "리츠"
    if "PREFERRED" in up:
        return "우선증권"
    if re.search(r"T-BILL|0-1 YEAR TREASURY|1-3 MONTH|TREASURY FLOATING RATE|ULTRA SHORT|ULTRA-SHORT|FLOATING RATE BOND|"
                 r"INVESTMENT GRADE FLOATING RATE|VARIABLE RATE INVESTMENT GRADE|SHORT DURATION BOND ACTIVE|SHORT TERM BOND ETF|"
                 r"MONEY MARKET", up) or "ULTRASHORT BOND" in cat:
        return "단기금리"
    if re.search(r"BOND|TREASURY|\bTIPS\b|INFLATION-PROTECTED|\bMUNI\b|MUNICIPAL|CORPORATE|HIGH YIELD(?! EQUITY)|AGGREGATE|\bMBS\b|"
                 r"INVESTMENT GRADE|"
                 r"MORTGAGE-BACKED|GNMA|CMBS|AGENCY|SENIOR LOAN|BANK LOAN|FLOATING RATE|CREDIT|FIXED INCOME|IBONDS|"
                 r"BULLETSHARES|FALLEN ANGELS|CONVERTIBLE|SOVEREIGN DEBT|INFLATION EXPECTATIONS|INCOME SECURITIES", up) \
            or cat.startswith("FIXED INCOME") or "FIXED INCOME SECTOR" in cat:
        return "채권"
    if re.search(r"\bSOUTH KOREA\b", up):
        return "국내주식"
    return "해외주식"


# ---------------------------------------------------------------------------
# 기초지수 묶음
# ---------------------------------------------------------------------------
# 국내 상장 상품: 혼합·파생 표시를 뗀 핵심 이름(kr_base) → 묶음
KR_BASE_FAMILY = {
    # 한국 대형 광역지수(시가총액 가중)는 거의 같은 지수로 본다.
    "": "KR_EQ_LARGE",  # KODEX 레버리지, TIGER 인버스 등(KOSPI200 기준)
    **{k: "KR_EQ_LARGE" for k in (
        "200", "K200", "코스피", "코스피200", "코스피100", "코스피50", "KRX300", "KRX100", "MSCIKOREA",
        "코스피대형주", "MSCIKOREAESG유니버설", "200ESG")},
    "코스닥150": "KR_KOSDAQ150", "코스닥": "KR_KOSDAQ150",
    "코리아밸류업": "KR_VALUEUP",
    "삼성전자": "STOCK_KR_SAMSUNG_ELEC", "SK하이닉스": "STOCK_KR_SK_HYNIX",
    "미국S&P500": "US_EQ_LARGE_BLEND", "미국500": "US_EQ_LARGE_BLEND", "미국대형주500": "US_EQ_LARGE_BLEND",
    "미국S&P500ESG": "US_EQ_LARGE_BLEND",
    "미국S&P500성장주": "US_EQ_LARGE_GROWTH",
    "미국러셀2000": "US_EQ_SMALL_BLEND",
    "미국나스닥100": "US_NASDAQ100",
    "미국나스닥100ETF": "US_NASDAQ100",  # 폐지 TIGER 미국나스닥100ETF선물
    "미국다우존스30": "US_DOW30",
    "미국배당다우존스": "US_DIV_DJ_US_DIVIDEND100", "미국배당": "US_DIV_DJ_US_DIVIDEND100",
    "미국배당100": "US_DIV_DJ_US_DIVIDEND100", "미국S&P500모멘텀": "US_EQ_SP500_MOMENTUM",
    "글로벌금채굴기업": "GOLD_MINERS",
    "미국S&P500배당귀족": "US_DIV_SP500_ARISTOCRATS",
    "미국S&P500동일가중": "US_EQ_SP500_EQUAL_WEIGHT",
    "미국S&P500테크놀로지": "US_SECTOR_TECH", "미국S&P500금융": "US_SECTOR_FINANCIALS",
    "미국S&P500헬스케어": "US_SECTOR_HEALTH", "미국S&P500에너지": "US_SECTOR_ENERGY",
    "미국S&P500유틸리티": "US_SECTOR_UTILITIES", "미국S&P500필수소비재": "US_SECTOR_STAPLES",
    "미국S&P500경기소비재": "US_SECTOR_DISCRETIONARY", "미국S&P500산업재": "US_SECTOR_INDUSTRIALS",
    "미국S&P500커뮤니케이션": "US_SECTOR_COMMUNICATION", "S&P글로벌헬스케어": "US_SECTOR_HEALTH",
    "미국S&P바이오": "US_SP_BIOTECH", "미국나스닥바이오": "US_NASDAQ_BIOTECH",
    "미국부동산리츠": "US_REIT", "미국MSCI리츠": "US_REIT",
    "S&P글로벌인프라": "GLOBAL_INFRASTRUCTURE",
    "미국30년국채": "US_TREASURY_LONG", "미국장기국채": "US_TREASURY_LONG", "미국30년국채울트라": "US_TREASURY_LONG",
    "미국채30년": "US_TREASURY_LONG", "미국30년국채스트립": "US_TREASURY_LONG",
    "미국10년국채": "US_TREASURY_INTERMEDIATE", "미국채10년": "US_TREASURY_INTERMEDIATE",
    "미국종합채권ESG": "US_AGG_BOND", "ISHARES미국투자등급회사채": "US_CORP_IG", "미국투자등급회사채": "US_CORP_IG",
    "ISHARES미국하이일드": "US_HIGH_YIELD", "미국하이일드": "US_HIGH_YIELD",
    "ISHARES미국인플레이션국채": "US_TIPS",
    "미국단기투자등급회사채": "US_CORP_IG_SHORT", "미국단기회사채(AAA~A)": "US_CORP_IG_SHORT",
    "미국장기우량회사채": "US_CORP_IG_LONG", "미국고정배당우선증권": "US_PREFERRED",
    "미국달러SOFR금리": "USD_CASH", "미국달러단기채권": "USD_CASH", "미국초단기(3개월이하)국채": "USD_CASH",
    "미국머니마켓": "USD_CASH",
    "골드": "GOLD", "금현물": "GOLD", "KRX금현물": "GOLD", "금": "GOLD", "국제금": "GOLD",
    "금은": "PRECIOUS_METALS_MIX", "은": "SILVER",
    "WTI원유": "CRUDE_OIL", "원유": "CRUDE_OIL", "구리": "COPPER", "구리실물": "COPPER",
    "3대농산물": "AGRICULTURE", "농산물": "AGRICULTURE", "팔라듐": "PALLADIUM",
    "미국달러": "FX_USD_KRW", "일본엔": "FX_JPY", "일본엔화초단기국채": "FX_JPY",
    "MSCI선진국": "GLOBAL_EQ", "선진국MSCI": "GLOBAL_EQ", "글로벌MSCI": "GLOBAL_EQ",
    "신흥국MSCI": "EM_EQ", "MSCIEM": "EM_EQ", "이머징마켓MSCI": "EM_EQ",
    "일본TOPIX": "COUNTRY_JAPAN", "일본TOPIX100": "COUNTRY_JAPAN", "일본NIKKEI225": "COUNTRY_JAPAN",
    "일본니케이225": "COUNTRY_JAPAN",
    "차이나H": "CHINA_OFFSHORE_LARGE", "차이나HSCEI": "CHINA_OFFSHORE_LARGE", "중국MSCICHINA": "CHINA_OFFSHORE_LARGE",
    "차이나항셍30": "CHINA_OFFSHORE_LARGE",
    "중국본토CSI300": "CHINA_A_LARGE", "차이나CSI300": "CHINA_A_LARGE", "차이나A50": "CHINA_A_LARGE",
    "중국본토대형주CSI100": "CHINA_A_LARGE",
    "차이나항셍테크": "CHINA_TECH", "과창판STAR50": "CHINA_STAR50", "중국과창판STAR50": "CHINA_STAR50",
    "차이나과창판STAR50": "CHINA_STAR50", "차이나심천CHINEXT": "CHINA_CHINEXT", "심천차이넥스트": "CHINA_CHINEXT",
    "인도NIFTY50": "COUNTRY_INDIA", "인도니프티50": "COUNTRY_INDIA",
    "유로스탁스50": "EUROPE_EQ", "유로스탁스": "EUROPE_EQ", "독일DAX": "COUNTRY_GERMANY",
    "베트남VN30": "COUNTRY_VIETNAM", "MSCI인도네시아": "COUNTRY_INDONESIA", "MSCI멕시코": "COUNTRY_MEXICO",
    "러시아MSCI": "COUNTRY_RUSSIA", "라틴35": "LATAM_EQ", "아시아TOP50": "ASIA50",
    "국고채3년": "KR_KTB_3Y", "국채3년": "KR_KTB_3Y",
    "국고채10년": "KR_KTB_10Y", "국채10년": "KR_KTB_10Y",
    "국고채30년": "KR_KTB_30Y", "국채30년": "KR_KTB_30Y", "KIS국고채30년": "KR_KTB_30Y", "국고채30년스트립": "KR_KTB_30Y",
    "중장기국공채": "KR_GOVT_MID_LONG", "중장기국채": "KR_GOVT_MID_LONG",
}

# 국내 상장 상품 정규식 규칙(KR_BASE_FAMILY에 없을 때, 순서대로). (묶음, 패턴, 근거)
KR_PATTERN_FAMILY: list[tuple[str, str, str]] = [
    ("KRW_CASH", r"CD.*금리|KOFR|머니마켓|단기자금|^단기채권|단기통안채|통안채1년|초단기채권|전단채|단기금융채|단기특수은행채|"
                 r"단기국공채|단기변동금리|단기채권알파|단기채권PLUS|단기채권ESG", "rule"),
    ("KR_AGG_BOND", r"^(ESG)?종합채권\((AA-|A-)이상\)$|^장기종합채권", "rule"),
    ("KR_AGG_BOND_SHORT_MID", r"^(중기|중단기)종합채권", "rule"),
    ("KR_CORP_BOND", r"^(중기)?우량회사채|^우량회사채|^금융채$|^중단기회사채|^중기채권", "rule"),
    ("KR_ROLLING_1Y_CREDIT", r"\d+월(만기)?자동연장", "rule"),
    ("KR_INFLATION_LINKED", r"물가채", "rule"),
    ("KR_SAMSUNG_HYNIX_MIX", r"삼성전자&?SK하이닉스", "rule"),
    ("US_TREASURY_LONG", r"^미국30년국채|^미국장기국채", "rule"),
    ("CARBON_ALLOWANCES", r"탄소배출권", "theme"),
]


def _kr_target_maturity(base: str) -> tuple[str, str] | None:
    """만기매칭형 채권. 신용채(회사채·금융채·은행채·특수채)는 만기 연도별로 묶는다.
    국고채는 만기가 먼 것(2046년 이후)은 30년 국고채, 2031~2040년은 10년 국고채 묶음과 거의 같은
    금리 노출로 보고 합친다(theme). 그 밖에는 만기 연도별 묶음."""
    m = re.match(r"^(\d\d)-(\d\d)(국고채|회사채|금융채|은행채|특수채)", base)
    if not m:
        return None
    year = 2000 + int(m.group(1))
    if m.group(3) == "국고채":
        if year >= 2046:
            return "KR_KTB_30Y", "theme"
        if 2031 <= year <= 2040:
            return "KR_KTB_10Y", "theme"
        return f"KR_TARGET_KTB_{year}", "rule"
    return f"KR_TARGET_CREDIT_{year}", "rule"


# 국내 주식 테마(이름 키워드로 거의 같은 노출을 묶음). 순서가 우선순위다.
KR_DOMESTIC_THEMES: list[tuple[str, str]] = [
    ("KR_THEME_SEMICONDUCTOR", r"반도체|HBM|하이닉스밸류체인|MLCC|AI기판"),
    ("KR_THEME_BATTERY", r"2차전지|배터리|양극재|전고체"),
    ("KR_THEME_SHIPBUILDING", r"조선"),
    ("KR_THEME_DEFENSE", r"방산"),
    ("KR_THEME_NUCLEAR", r"원자력|SMR"),
    ("KR_THEME_POWER", r"전력"),
    ("KR_THEME_ROBOT", r"로봇|휴머노이드|피지컬AI"),
    ("KR_THEME_BIO_HEALTH", r"바이오|헬스케어|의료|신약|제약"),
    ("KR_THEME_SECURITIES", r"증권|초대형IB"),
    ("KR_THEME_INSURANCE", r"보험"),
    ("KR_THEME_FINANCIALS", r"은행|금융|금융지주"),
    ("KR_THEME_GAME", r"게임"),
    ("KR_THEME_MEDIA_ENT", r"KPOP|K-POP|엔터|컨텐츠|콘텐츠|미디어|웹툰|컬처"),
    ("KR_THEME_COSMETICS", r"화장품|뷰티"),
    ("KR_THEME_AUTO", r"자동차|현대차|모빌리티|자율주행|스마트카|전기&수소차"),
    ("KR_THEME_SAMSUNG_GROUP", r"삼성그룹"),
    ("KR_THEME_SAMSUNG_SK_GROUP", r"삼성SK그룹"),
    ("KR_THEME_CLIMATE_GREEN", r"기후변화|탄소효율|그린뉴딜|신재생|친환경에너지|(?<!필)수소|태양광"),
    ("KR_THEME_INTERNET_SOFTWARE", r"인터넷|플랫폼|소프트웨어|메타버스|E커머스|5G|네트워크인프라"),
    ("KR_THEME_IT", r"^200IT$|^IT$|^IT플러스$|^IT PLUS$|^ITPLUS$"),
    ("KR_THEME_HIGH_DIVIDEND", r"고배당|배당"),
    ("KR_THEME_LOW_VOL", r"로우볼|저변동|최소변동성"),
    ("KR_SECTOR_CONSTRUCTION", r"건설"),
    ("KR_SECTOR_STEEL_MATERIALS", r"철강"),
    ("KR_SECTOR_ENERGY_CHEMICAL", r"에너지화학"),
    ("KR_SECTOR_INDUSTRIALS", r"산업재|중공업|기계장비"),
    ("KR_SECTOR_CONSUMER", r"경기소비재|생활소비재|필수소비재|내수"),
    ("KR_SECTOR_COMMUNICATION", r"커뮤니케이션"),
    ("KR_THEME_SPACE", r"우주"),
    ("KR_STYLE_MOMENTUM", r"모멘텀"),
    ("KR_STYLE_VALUE", r"^가치주$|^밸류PLUS$|^우량가치$"),
    ("KR_TOP_N_LARGE", r"^(200)?TOP\d+(동일가중|PLUS)?$|^코리아TOP10$|^TOP5PLUS$"),
]

# 해외(국내 상장 해외주식) 테마. 순서가 우선순위다.
FOREIGN_THEMES: list[tuple[str, str, str]] = [
    # (묶음, 국내 상장 이름 패턴, 미국 상장 핵심 이름 패턴)
    ("CHINA_SEMIS", r"(차이나|중국|한중).*반도체|차이나AI반도체", r"^$"),
    ("JAPAN_SEMIS", r"일본반도체", r"^$"),
    ("US_GLOBAL_SEMIS", r"반도체|HBM|메모리|DRAM|낸드|TSMC|MLCC|AI기판", r"SEMICONDUCTOR"),
    ("CHINA_THEME_CLEAN_ENERGY", r"차이나.*(클린에너지|태양광)", r"^$"),
    ("CHINA_THEME_EV_BATTERY", r"(차이나|한중).*(전기차|2차전지)", r"^$"),
    ("CHINA_THEME_BIO", r"차이나.*바이오", r"^$"),
    ("CHINA_THEME_ROBOT", r"차이나.*(휴머노이드|로봇)", r"^$"),
    ("CHINA_TECH", r"(차이나|중국).*(테크|AI|인터넷|플랫폼|빅테크|소프트웨어)|항셍테크", r"CHINA TECHNOLOGY"),
    ("GLOBAL_THEME_NUCLEAR", r"원자력|SMR", r"URANIUM|NUCLEAR"),
    ("GLOBAL_THEME_QUANTUM", r"양자", r"QUANTUM"),
    ("GLOBAL_THEME_SPACE", r"우주(?!항공방산)", r"FINAL FRONTIERS|SPACE"),
    ("US_AEROSPACE_DEFENSE", r"미국방산|미국우주항공방산", r"AEROSPACE & DEFENSE"),
    ("EUROPE_DEFENSE", r"유럽방산", r"^$"),
    ("GLOBAL_THEME_ROBOT", r"휴머노이드|로봇|피지컬AI", r"ROBOTICS"),
    ("GLOBAL_THEME_OBESITY", r"비만", r"OBESITY"),
    ("GLOBAL_THEME_POWER_INFRA", r"전력|GRID", r"GRID"),
    # '(?<!내)수소': '차이나내수소비'의 '수소'를 수소 테마로 잡지 않는다(폐지 454180).
    ("GLOBAL_THEME_CLEAN_ENERGY", r"클린에너지|친환경그린|(?<!내)수소|태양광", r"CLEAN ENERGY|^SOLAR$|KENSHO GRID|CLEAN POWER"),
    ("US_THEME_NATGAS_INFRA", r"천연가스", r"^$"),
    ("US_THEME_AI_SOFTWARE", r"미국AI소프트웨어", r"TECH-SOFTWARE SECTOR|SOFTWARE & SERVICES"),
    ("US_BIGTECH", r"미국.*(빅테크|테크TOP|AI테크TOP|테크TOP\d+)|^미국테크TOP", r"^$"),
]

# 미국 상장 상품 핵심 이름(us_core) 규칙. (묶음, 포함 패턴, 제외 패턴)
_US_STYLE_EXCLUDE = (
    r"EQUAL|DIVIDEND|ARISTOCRAT|MOMENTUM|QUALITY|LOW VOL|MIN VOL|MINIMUM VOL|HIGH BETA|REVENUE|GARP|QVM|PURE|TOP 50|"
    r"\bEX-|FACTOR|STRATEGIC|SIZE|GENDER|FUNDAMENTAL|RAFI|BUYBACK|DORSEY|ZACKS|YIELD|MULTIFACTOR|TILT|CORE PLUS|"
    r"SPIN-OFF|DOWNSIDE|ANALYST|PRICING POWER|MVP|CONCENTRATED|RISING RATES|INFRASTRUCTURE|MICROCAP|"
    r"TECHNOLOGY|FINANCIAL|HEALTH|ENERGY|UTILIT|MATERIAL|INDUSTRIAL|CONSUMER|REAL ESTATE|COMMUNICATION|"
    r"SEMICONDUCTOR|BIOTECH|BANK|INSURANCE|EMERGING|EAFE|INTERNATIONAL|DEVELOPED|GLOBAL|WORLD|JAPAN|EUROPE|"
    r"CHINA|INDIA|BRAZIL|CANADA|BOND|TREASURY"
)

US_RULES: list[tuple[str, str, str]] = [
    ("USD_CASH", r"T-BILL|0-1 YEAR TREASURY|1-3 MONTH|TREASURY FLOATING RATE|ULTRA SHORT|ULTRA-SHORT|^FLOATING RATE BOND|"
                 r"INVESTMENT GRADE FLOATING RATE|VARIABLE RATE INVESTMENT GRADE|SHORT DURATION BOND ACTIVE|ULTRA SHORT TERM BOND", r"MUNI"),
    ("US_TREASURY_LONG", r"20\+? ?YEAR TREASURY|LONG[- ]TERM (U\.S\. )?TREASURY|EXTENDED DURATION TREASURY|10-20 YEAR TREASURY|"
                         r"LEHMAN 20 +YEAR TREASURY|20 YEAR TREASURY", ""),
    ("US_TREASURY_INTERMEDIATE", r"7[- ]10 YEAR TREASURY|3-7 YEAR TREASURY|INTERMEDIATE[- ]TERM (U\.S\. )?TREASURY|"
                                 r"^U\.S\. TREASURY BOND$|EQUAL WEIGHT 0-30 YEAR TREASURY", ""),
    ("US_TREASURY_SHORT", r"1-3 YEAR TREASURY|SHORT[- ]TERM (U\.S\. )?TREASURY", r"INTERNATIONAL"),
    ("US_TIPS_SHORT", r"0-5 (YR|YEAR) (US )?TIPS|SHORT-TERM INFLATION-PROTECTED|1.{0,3}10 YEAR TIPS", ""),
    ("US_TIPS", r"^TIPS BOND$|^U\.S\. TIPS$|^TIPS$", ""),
    ("US_AGG_BOND", r"AGGREGATE BOND|TOTAL BOND MARKET|UNIVERSAL USD BOND|^DOUBLELINE TOTAL RETURN TACTICAL$|^TOTAL RETURN BOND$|"
                    r"^SYSTEMATIC BOND$", r"INTERNATIONAL|GLOBAL|WORLD|SHORT DURATION"),
    ("US_CORP_IG_SHORT", r"(0-5|1-5|1-3) YEAR (INVESTMENT GRADE |USD )?CORPORATE|SHORT[- ]TERM CORPORATE", ""),
    ("US_CORP_IG_LONG", r"LONG[- ]TERM CORPORATE|(^|[^-\d])10\+? +YEAR INVESTMENT GRADE CORPORATE", r"HEDGED"),
    ("US_BOND_SHORT_BROAD", r"^SHORT-TERM BOND$|^1-5 YEAR USD BOND$", ""),
    ("US_BOND_INTERMEDIATE_BROAD", r"^INTERMEDIATE-TERM BOND$|^5-10 YEAR USD BOND$|^INTERMEDIATE GOVERNMENT/CREDIT BOND$", ""),
    ("US_BOND_LONG_BROAD", r"^LONG-TERM BOND$|^10\+? +YEAR USD BOND$", ""),
    ("US_CORP_IG", r"INVESTMENT GRADE CORPORATE BOND$|IBOXX \$ INVESTMENT GRADE|^CORPORATE BOND$|TOTAL CORPORATE BOND|"
                   r"BROAD USD INVESTMENT GRADE|5-10 YEAR (INVESTMENT GRADE )?CORPORATE|INTERMEDIATE[- ]TERM CORPORATE|"
                   r"AAA A RATED CORPORATE|INVESTMENT GRADE SYSTEMATIC|ESG AWARE USD CORPORATE",
                   r"HEDGED|INTERNATIONAL|EM |CONVERTIBLE|BULLETSHARES|IBONDS"),
    ("US_HIGH_YIELD_SHORT", r"0-5 YEAR HIGH YIELD|SHORT TERM HIGH YIELD", ""),
    ("US_HIGH_YIELD", r"HIGH YIELD (CORPORATE )?BOND$|IBOXX \$ HIGH YIELD|BROAD USD HIGH YIELD|ESG ADVANCED HIGH YIELD|"
                      r"HIGH YIELD SYSTEMATIC|^HIGH YIELD$|^HIGH YIELD ETF$",
                      r"HEDGED|EURO|EMERGING|EM |MUNI|GLOBAL|INTL|EX-US|DIVIDEND|EQUITY"),
    ("US_MUNI_SHORT", r"SHORT-TERM NATIONAL MUNI|SHORT TERM MUNICIPAL|SHORT MATURITY MUNICIPAL", ""),
    ("US_MUNI", r"NATIONAL MUNI|^NUVEEN ICE MUNICIPAL BOND$|TAX-EXEMPT BOND|NATIONAL AMT-FREE MUNICIPAL", ""),
    ("US_MBS", r"\bMBS\b|MORTGAGE[- ]BACKED|GNMA", r"CMBS"),
    ("EM_USD_BOND", r"USD EMERGING MARKETS BOND|EMERGING MARKETS GOVERNMENT BOND|EMERGING MARKETS SOVEREIGN DEBT", ""),
    ("EM_LOCAL_BOND", r"EM LOCAL CURRENCY|EMERGING MARKETS LOCAL BOND", ""),
    ("INTL_TREASURY_SHORT", r"1-3 YEAR INTERNATIONAL TREASURY|SHORT TERM INTERNATIONAL TREASURY", ""),
    ("INTL_TREASURY", r"INTERNATIONAL TREASURY BOND", ""),
    ("INTL_AGG_BOND_HEDGED", r"TOTAL INTERNATIONAL BOND|INTERNATIONAL AGGREGATE BOND", ""),
    ("US_BANK_LOAN", r"SENIOR LOAN", ""),
    ("US_PREFERRED", r"PREFERRED", ""),
    ("US_CONVERTIBLE", r"CONVERTIBLE", ""),
    # 원자재·통화·변동성
    ("GOLD", r"^GOLD\b", r"MINER"),
    ("SILVER", r"^SILVER\b", r"MINER"),
    ("PRECIOUS_METALS_MIX", r"^DB PRECIOUS METALS", ""),
    ("CRUDE_OIL", r"CRUDE OIL|^DB OIL", ""),
    ("NATURAL_GAS", r"NATURAL GAS$", ""),
    ("AGRICULTURE", r"^DB AGRICULTURE", ""),
    ("BASE_METALS", r"^DB BASE METALS", ""),
    ("ENERGY_COMMODITY", r"^DB ENERGY", ""),
    ("BROAD_COMMODITY", r"DB COMMODITY INDEX TRACKING|OPTIMUM YIELD DIVERSIFIED COMMODITY|GSCI COMMODITY|ROLL SELECT COMMODITY", ""),
    ("FX_USD_INDEX", r"DB USD INDEX", ""),
    ("FX_EUR", r"CURRENCYSHARES EURO|^EURO$", ""),
    ("FX_JPY", r"CURRENCYSHARES JAPANESE YEN|^YEN( NEW)?$", ""),
    ("FX_GBP", r"CURRENCYSHARES BRITISH POUND", ""),
    ("FX_CAD", r"CURRENCYSHARES CANADIAN DOLLAR", ""),
    ("FX_AUD", r"CURRENCYSHARES AUSTRALIAN DOLLAR", ""),
    ("FX_CHF", r"CURRENCYSHARES SWISS FRANC", ""),
    ("VIX_SHORT_TERM_FUTURES", r"VIX SHORT[- ]TERM FUTURES", ""),
    ("VIX_MID_TERM_FUTURES", r"VIX MID[- ]TERM FUTURES", ""),
    # 미국 주식 지수
    ("US_NASDAQ100", r"^QQQ\b|NASDAQ[- ]100", r"EQUAL|NEXT"),
    ("US_DOW30", r"^DOW30$|DOW JONES INDUSTRIAL AVERAGE$", ""),
    ("US_DIV_DJ_US_DIVIDEND100", r"^US DIVIDEND EQUITY$", ""),
    ("US_DIV_SP500_ARISTOCRATS", r"S&P 500 DIVIDEND ARISTOCRATS", ""),
    ("US_DIV_GROWTH", r"^DIV APPRECIATION$|^DIVIDEND GROWTH$", ""),
    ("US_EQ_SP500_EQUAL_WEIGHT", r"^S&P 500 EQUAL WEIGHT$", ""),
    ("US_EQ_SP500_MOMENTUM", r"^S&P 500 MOMENTUM$", ""),
    ("US_SECTOR_TECH", r"^(TECHNOLOGY SELECT SECTOR|INFORMATION TECH|DOW JONES U\.S\. TECHNOLOGY SECTOR|TECHNOLOGY|"
                       r"EXPANDED TECH SECTOR|GLOBAL TECH)$", ""),
    ("US_SECTOR_FINANCIALS", r"^(FINANCIAL SELECT SECTOR|FINANCIALS|U\.S\. FINANCIAL|GLOBAL FINANCIAL)$", ""),
    ("US_SECTOR_HEALTH", r"^(HEALTH CARE SELECT SECTOR|HEALTH CARE|U\.S\. HEALTHCARE|GLOBAL HEALTHCARE)$", ""),
    ("US_SECTOR_ENERGY", r"^(ENERGY SELECT SECTOR|ENERGY|U\.S\. ENERGY|GLOBAL ENERGY)$", ""),
    ("US_SECTOR_UTILITIES", r"^(UTILITIES SELECT SECTOR|UTILITIES|U\.S\. UTILITIES|GLOBAL UTILITIES)$", ""),
    ("US_SECTOR_STAPLES", r"^(CONSUMER STAPLES SELECT SECTOR|CONSUMER STAPLES|U\.S\. CONSUMER STAPLES|GLOBAL CONSUMER STAPLES)$", ""),
    ("US_SECTOR_DISCRETIONARY", r"^(CONSUMER DISCRETIONARY SELECT SECTOR|CONSUMER DISCRETION(ARY)?|U\.S\. CONSUMER DISCRETIONARY|"
                                r"GLOBAL CONSUMER DISCRETIONARY)$", ""),
    ("US_SECTOR_INDUSTRIALS", r"^(INDUSTRIAL SELECT SECTOR|INDUSTRIALS|U\.S\. INDUSTRIALS|GLOBAL INDUSTRIALS)$", ""),
    ("US_SECTOR_MATERIALS", r"^(MATERIALS SELECT SECTOR|MATERIALS|U\.S\. BASIC MATERIALS|GLOBAL MATERIALS)$", ""),
    ("US_SECTOR_COMMUNICATION", r"^(COMMUNICATION SERVICES SELECT SECTOR|COMMUNICATION SERVICES|GLOBAL COMM SERVICES)$", ""),
    ("US_REIT", r"^(REAL ESTATE SELECT SECTOR|REAL ESTATE|U\.S\. REAL ESTATE|U\.S\. REIT|DOW JONES REIT|SELECT U\.S\. REIT)$", ""),
    ("GLOBAL_REIT", r"^(GLOBAL REIT|DOW JONES GLOBAL REAL ESTATE)$", ""),
    ("INTL_REIT", r"^(GLOBAL EX-U\.S\. REAL ESTATE|DOW JONES INTERNATIONAL REAL ESTATE|INTERNATIONAL DEVELOPED REAL ESTATE)$", ""),
    ("US_NASDAQ_BIOTECH", r"^(NASDAQ )?BIOTECHNOLOGY$", ""),
    ("US_SP_BIOTECH", r"^S&P BIOTECH$", ""),
    ("US_BANKS", r"^(S&P BANK|KBW BANK)$", ""),
    ("US_REGIONAL_BANKS", r"^(S&P REGIONAL BANKING|U\.S\. REGIONAL BANKS)$", ""),
    ("US_HOMEBUILDERS", r"^(S&P HOMEBUILDERS|U\.S\. HOME CONSTRUCTION)$", ""),
    ("US_OIL_EXPLORATION", r"OIL & GAS EXPLORATION & PRODUCTION|ENERGY EXPLORATION & PRODUCTION", ""),
    ("US_OIL_SERVICES", r"OIL & GAS EQUIPMENT & SERVICES|OIL EQUIPMENT & SERVICES|^OIL & GAS SERVICES$", ""),
    ("US_AEROSPACE_DEFENSE", r"AEROSPACE & DEFENSE", ""),
    ("US_INSURANCE", r"^(S&P INSURANCE|U\.S\. INSURANCE)$", ""),
    ("US_PHARMA", r"^(S&P PHARMACEUTICALS|U\.S\. PHARMACEUTICAL|PHARMACEUTICALS)$", ""),
    ("US_MED_DEVICES", r"^(U\.S\. MEDICAL DEVICES|S&P HEALTH CARE EQUIPMENT)$", ""),
    ("US_TRANSPORTS", r"^(S&P TRANSPORTATION|U\.S\. TRANSPORTATION)$", ""),
    ("US_TELECOM", r"^(S&P TELECOM|U\.S\. TELECOMMUNICATIONS)$", ""),
    ("NA_NATURAL_RESOURCES", r"NORTH AMERICAN NATURAL RESOURCES", ""),
    ("METALS_MINING", r"METALS & MINING", ""),
    ("GOLD_MINERS", r"GOLD MINERS", ""),
    ("GLOBAL_INFRASTRUCTURE", r"^(S&P GLOBAL INFRASTRUCTURE|GLOBAL INFRASTRUCTURE|DJ BROOKFIELD GLOBAL INFRASTRUCTURE)$", ""),
    ("GLOBAL_WATER", r"WATER", ""),
    ("GLOBAL_TIMBER", r"TIMBER", ""),
    ("LISTED_PRIVATE_EQUITY", r"LISTED PRIVATE EQUITY", ""),
    # 국제 주식
    ("GLOBAL_EQ", r"^MSCI ACWI$|TOTAL WORLD STOCK|^MSCI WORLD$|MSCI GLOBAL STOCK MARKET|LOW CARBON OPTIMIZED MSCI ACWI|"
                  r"MSCI ACWI CLIMATE PARIS ALIGNED|MSCI KOKUSAI", ""),
    ("INTL_EQ", r"^(CURRENCY HEDGED |ESG AWARE )?MSCI EAFE$|^MSCI EAFE FOSSIL FUEL RESERVES FREE$|^FTSE DEVELOPED MARKETS$|"
                r"^INTERNATIONAL EQUITY$|^DEVELOPED WORLD EX-US$|^MSCI INTERNATIONAL DEVELOPED MARKETS$|"
                r"^(CURRENCY HEDGED )?MSCI ACWI EX[- ]U\.?S\.?$|^FTSE ALL WORLD EX US$|TOTAL INTERNATIONAL STOCK|"
                r"^ESG INTERNATIONAL STOCK$", ""),
    ("INTL_SMALL_EQ", r"^(CURRENCY HEDGED )?MSCI EAFE SMALL-CAP$|^S&P INTERNATIONAL SMALL CAP$|^FTSE ALL-WLD EX-US SMCP IDX$|"
                      r"^INTERNATIONAL SMALL-CAP EQUITY$", ""),
    ("EM_EQ", r"^(CURRENCY HEDGED )?MSCI EMERGING( MARKETS)?$|^MSCI EMERGING MARKETS$|^FTSE EMERGING MARKETS$|"
              r"^EMERGING MARKETS EQUITY$|^EMERGING MARKETS$|^ESG AWARE MSCI EM$|^MSCI EMERGING MARKETS FOSSIL FUEL RESERVES FREE$", ""),
    ("EUROPE_EQ", r"EURO STOXX 50|^(CURRENCY HEDGED )?MSCI EUROZONE$|^MSCI EUROPE$|^EUROPE$|^FTSE ?EUROPEAN$|^FTSE EUROPE$", ""),
    ("COUNTRY_JAPAN", r"^(CURRENCY HEDGED )?MSCI JAPAN$|JPX-NIKKEI 400", ""),
    ("CHINA_OFFSHORE_LARGE", r"^MSCI CHINA$|CHINA LARGE-CAP|FTSE CHINA 50|^S&P CHINA$", ""),
    ("CHINA_A_LARGE", r"^MSCI CHINA A$", ""),
    ("COUNTRY_INDIA", r"^MSCI INDIA$|^INDIA 50$|^INDIA$", ""),
    ("KR_EQ_LARGE", r"^MSCI SOUTH KOREA$", ""),
    ("PACIFIC_EQ", r"^FTSE PACIFIC$|^MSCI PACIFIC$", ""),
    ("LATAM_EQ", r"LATIN AMERICA 40", ""),
    ("ASIA50", r"^ASIA 50$", ""),
]


def _us_style_family(core: str, region: str) -> str | None:
    """미국 광역 주식의 크기×스타일 묶음(US_EQ_{LARGE|MID|SMALL}_{BLEND|GROWTH|VALUE})."""
    if region != "미국":
        return None
    if re.search(_US_STYLE_EXCLUDE, core):
        return None
    size = None
    if re.search(r"MID[- ]?CAP|MIDCAP|S&P 400|RUSSELL MID|EXTENDED MARKET|RUSSELL 2500", core):
        size = "MID"
    elif re.search(r"SMALL[- ]?CAP|SMALLCAP|S&P 600|RUSSELL ?2000|SM CAP", core):
        size = "SMALL"
    elif re.search(r"S&P ?500|RUSSELL 1000|RUSSELL 3000|LARGE[- ]CAP|MEGA CAP|TOP 200|S&P 100\b|TOTAL STOCK MARKET|"
                   r"TOTAL U\.S\. STOCK MARKET|BROAD MARKET|S&P (COMPOSITE )?1500|^1000$|MSCI USA|DOW JONES U\.S\.$|"
                   r"KLD 400|ESG U\.S\. STOCK|^MORNINGSTAR (GROWTH|VALUE)$|^S&P U\.S\. (GROWTH|VALUE)$", core):
        size = "LARGE"
    if size is None:
        return None
    style = "GROWTH" if "GROWTH" in core else "VALUE" if "VALUE" in core else "BLEND"
    return f"US_EQ_{size}_{style}"


_US_COUNTRY_RE = re.compile(r"^(?:CURRENCY HEDGED )?MSCI (?P<c>[A-Z ]+?)(?: CAPPED| INDEX)?$")


def _family_us(name: str, core: str, region: str, asset_class: str) -> tuple[str, str]:
    if asset_class == "단기금리":
        return "USD_CASH", "rule"
    m = re.search(r"BULLETSHARES (\d{4}) (HIGH YIELD )?CORPORATE|IBONDS DEC (\d{4}) TERM (HIGH YIELD )?CORPORATE", core)
    if m:
        year = m.group(1) or m.group(3)
        hy = bool(m.group(2) or m.group(4))
        return f"US_TARGET_{'HY' if hy else 'CORP'}_{year}", "rule"
    for family, pat, excl in US_RULES:
        if re.search(pat, core) and not (excl and re.search(excl, core)):
            return family, "rule"
    style = _us_style_family(core, region)
    if style:
        return style, "rule"
    m = _US_COUNTRY_RE.match(core)
    if m and m.group("c") in _US_COUNTRIES and "SMALL" not in core:
        return f"COUNTRY_{m.group('c').replace(' ', '_')}", "rule"
    if m and m.group("c") in ("GERMANY",):
        return "COUNTRY_GERMANY", "rule"
    for family, _kr_pat, us_pat in FOREIGN_THEMES:
        if us_pat != r"^$" and re.search(us_pat, core):
            return family, "theme"
    return "NAME:" + _name_key(core), "name"


def _family_kr(rest: str, base: str, asset_class: str, region: str) -> tuple[str, str]:
    if base in KR_BASE_FAMILY:
        fam = KR_BASE_FAMILY[base]
        if base == "" and asset_class != "국내주식":
            fam = None
        if fam:
            return fam, "rule"
    tm = _kr_target_maturity(base)
    if tm:
        return tm
    for family, pat, basis in KR_PATTERN_FAMILY:
        if re.search(pat, base):
            return family, basis
    foreign = bool(_KR_FOREIGN.search(rest)) or asset_class in ("해외주식", "국내외주식")
    if asset_class == "리츠" and region == "한국":
        return "KR_REIT", "theme"
    if asset_class == "혼합" and "리츠" in rest and not foreign:
        return "KR_REIT", "theme"
    if asset_class == "국내주식" or (asset_class == "혼합" and not foreign):
        for family, pat in KR_DOMESTIC_THEMES:
            if re.search(pat, base):
                return family, "theme"
    if asset_class in ("해외주식", "국내외주식") or (asset_class == "혼합" and foreign):
        for family, kr_pat, _us in FOREIGN_THEMES:
            if re.search(kr_pat, base):
                return family, "theme"
    return "NAME:" + _name_key(base), "name"


# ---------------------------------------------------------------------------
# 과세 분류 (국내 상장만)
# ---------------------------------------------------------------------------
def _kr_tax(rest: str, asset_class: str, is_single: bool, futures: bool, reasons: list[str]) -> tuple[str, str]:
    if asset_class == "국내주식":
        if "(합성" in rest:
            reasons.append("국내주식 합성(스왑) 상품: 국내주식형 비과세 해당 여부 확인 필요")
            return "국내주식형", "국내 주식 노출이나 합성(스왑) 구조 — 확인 필요"
        if is_single and not futures:
            reasons.append("국내 단일종목 레버리지(선물 표기 없음): 노출 수단과 과세 분류 확인 필요")
            return "국내주식형", "국내 단일종목 노출, 노출 수단 미확인 — 확인 필요"
        if re.search(r"롱.*숏", rest):
            return "국내주식형", "국내 주가지수 장내선물 롱숏"
        basis = "국내 주식 또는 국내 주가지수 장내파생(선물·옵션)으로 노출"
        return "국내주식형", basis
    if asset_class == "리츠" and not _KR_FOREIGN.search(rest):
        reasons.append("국내 리츠·부동산 인프라 ETF: 상장 리츠·인프라펀드 편입분의 과세 분류 확인 필요")
        return "기타", "국내 리츠·인프라 편입(국내주식형 아님으로 가정, 확인 필요)"
    return "기타", f"{asset_class} 노출: 국내주식형이 아님(매매차익과 과표기준가 증가분 중 작은 금액에 15.4%)"


# ---------------------------------------------------------------------------
# 한 상품 분류
# ---------------------------------------------------------------------------
def classify(row: Mapping[str, str]) -> dict:
    market = row["market"]
    symbol = row["symbol"]
    name = row["name"]
    issuer = row.get("issuer", "") or ""
    tab = row.get("raw_category_code", "") or ""
    reasons: list[str] = []

    lev, inverse, lev_reasons = classify_leverage(market, name, issuer)
    reasons += lev_reasons
    etn = is_etn(market, name, row.get("explicit_etn_name", "0"))
    options = is_options_strategy(market, name)
    futures = is_futures_based(market, name, issuer)

    if market == "KR":
        rest = kr_rest(name)
        base = kr_base(rest)
        single = "단일종목" in rest
        if symbol in _KR_DOMESTIC_OVERRIDE:
            asset_class = "국내주식"
        else:
            asset_class = _kr_asset_class(rest, base, tab, reasons)
        region = _kr_region(rest, asset_class)
        family, basis = _family_kr(rest, base, asset_class, region)
        tax_class, tax_basis = _kr_tax(rest, asset_class, single, futures, reasons)
        if single:
            reasons.append("단일종목 레버리지: D1에 명시되지 않은 상품 유형, 포함 여부 결정 필요")
        if base in ("MSCI선진국", "선진국MSCI"):
            reasons.append("선진국MSCI: MSCI World(미국 포함)인지 EAFE(미국 제외)인지 이름만으로 불명확")
    else:
        core = us_core(name)
        single = False
        asset_class = _us_asset_class(name, core, row.get("issuer_category", "") or "")
        region = _us_region(name.upper())
        if asset_class in ("금", "원자재", "변동성", "가상자산"):
            region = "글로벌" if asset_class != "변동성" else "미국"
        family, basis = _family_us(name, core, region, asset_class)
        tax_class, tax_basis = "", ""
        if asset_class == "변동성":
            reasons.append("VIX 선물 상품: D1에 명시되지 않은 자산군, 포함 여부 결정 필요")
        if re.search(r"DOWNSIDE HEDGED|HEDGE REPLICATION|INFLATION EXPECTATIONS|DECLINE OF", name.upper()):
            reasons.append("헤지·차익형 전략: 노출 구조가 일반 지수형과 달라 자산군·묶음 확인 필요")

    # KRX 기초지수명으로 추정한 탭이 국내(1·2·3)가 아닌데 이름 규칙이 국내주식으로 떨어지면(탭 5·7의 기본값),
    # 추정 근거와 결과가 어긋난 것이므로 확정하지 않고 검토로 남긴다.
    if market == "KR" and row.get("naver_tab_source") == "krx_index_name" and tab not in ("1", "2", "3") \
            and asset_class == "국내주식":
        reasons.append(f"KRX 기초지수명으로 추정한 탭 {tab}({NAVER_TABS.get(tab, '?')})과 자산군(국내주식)이 다름")
    if asset_class == "기타":
        reasons.append("자산군을 규칙으로 정하지 못함")
    if region == "미확인":
        reasons.append("노출 지역을 규칙으로 정하지 못함")

    # 원천 이름 힌트와의 불일치는 검토 사유로 남긴다(규칙 판단을 값으로 쓴다).
    hint_lev = row.get("leveraged_name_hint", "0") == "1"
    hint_inv = row.get("inverse_name_hint", "0") == "1"
    hint_opt = row.get("options_strategy_name_hint", "0") == "1"
    # 규칙이 명시 키워드(버퍼·목표헤지·Bearish·Decline of 등)로 True를 낸 경우는 규칙 어휘가 더 넓은 것이므로
    # 표시하지 않는다. 원천 힌트는 1인데 규칙이 0인 경우만 검토 사유로 남긴다.
    rule_lev = lev is not None and abs(lev) >= 2
    # 원천 레버리지 힌트는 Ultra 단어에도 1이 된다(예: Ultra Short Duration 채권, RDIV "Ultra Dividend").
    # ProShares가 아닌 상품의 Ultra는 배수 근거가 아니므로 이 불일치는 표시하지 않는다.
    up_name = name.upper()
    ultra_short_bond = market == "US" and issuer != "ProShares" and "ULTRA" in up_name \
        and not re.search(r"\d(\.\d+)?\s*X\b|LEVERAGED", up_name)
    # 규칙이 이름의 배수 숫자(1X·1.5x 등)를 직접 읽었으면 원천 레버리지 힌트는 더할 정보가 없다.
    explicit_multiple = bool(re.search(r"\d(\.\d+)?\s*X\b", up_name)) if market == "US" \
        else bool(_KR_MULTIPLE.search(kr_rest(name)))
    # Ultra Short·Ultrashort는 초단기 채권이다(원천 인버스 힌트가 Short 단어로 1이 됨).
    ultra_short_word = market == "US" and bool(re.search(r"ULTRA[- ]?SHORT", up_name)) and issuer != "ProShares"
    # 이 불일치는 규칙이 놓친 제외 대상일 수 있으므로 해소 전까지 D1에서도 뺀다(fail closed).
    # 롱숏(배수 미확인)은 leverage_unknown으로 이미 제외되므로 따로 표시하지 않는다.
    unresolved: list[str] = []
    if hint_inv and not inverse and not ultra_short_word and lev is not None:
        reasons.append("원천 인버스 힌트(1)와 규칙 판단(0)이 다름")
        unresolved.append("inverse_hint_unresolved")
    if hint_lev and not rule_lev and lev is not None and not ultra_short_bond and not explicit_multiple:
        reasons.append(f"원천 레버리지 힌트(1)와 규칙 배수({lev:g})가 다름")
        unresolved.append("leverage_hint_unresolved")
    if hint_opt and not options:
        reasons.append("원천 옵션 전략 힌트(1)와 규칙 판단(0)이 다름")
        unresolved.append("options_hint_unresolved")

    d1 = []
    if inverse:
        d1.append("inverse")
    if etn:
        d1.append("etn")
    if options:
        d1.append("options_strategy")
    if lev is None:
        d1.append("leverage_unknown")
    elif abs(lev) >= 3:
        d1.append("leverage_3x_or_more")
    d1 += unresolved

    return {
        "market": market,
        "symbol": symbol,
        "name": name,
        "issuer": issuer,
        "has_price_history": row.get("history_download_status", "") == "downloaded",
        "naver_tab": tab,
        "leverage_multiple": lev,
        "is_inverse": inverse,
        "is_etn": etn,
        "is_options_strategy": options,
        "is_futures_based": futures,
        "is_single_stock": single,
        "asset_class": asset_class,
        "region": region,
        "index_family": family,
        "index_family_basis": basis,
        "kr_tax_class": tax_class,
        "kr_tax_basis": tax_basis,
        "d1_eligible": not d1,
        "d1_exclusion_reasons": ";".join(d1),
        "needs_review": bool(reasons),
        "review_reasons": " | ".join(dict.fromkeys(reasons)),
        "listing_status": row.get("listing_status") or LISTED,
        "naver_tab_source": row.get("naver_tab_source") or ("naver" if tab else ""),
    }


def build_metadata(universe: pd.DataFrame) -> pd.DataFrame:
    rows = [classify(r) for r in universe.to_dict("records")]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def summarize(meta: pd.DataFrame) -> dict:
    priced = meta[meta["has_price_history"]]
    fam_sizes = priced.groupby("index_family").size().sort_values(ascending=False)
    lev2 = priced[priced["leverage_multiple"].abs() == 2]
    return {
        "rows": len(meta),
        "priced_rows": len(priced),
        "priced_by_market": priced.groupby("market").size().to_dict(),
        "asset_class": priced.groupby(["market", "asset_class"]).size().to_dict(),
        "families_priced": int(fam_sizes.size),
        "families_priced_multi": int((fam_sizes > 1).sum()),
        "largest_families": fam_sizes.head(15).to_dict(),
        "family_basis": priced.groupby("index_family_basis").size().to_dict(),
        "needs_review_priced": int(priced["needs_review"].sum()),
        "needs_review_all": int(meta["needs_review"].sum()),
        "leverage_2x_priced_by_market": lev2.groupby("market").size().to_dict(),
        "leverage_2x_long_priced_by_market": lev2[lev2["leverage_multiple"] > 0].groupby("market").size().to_dict(),
        "d1_eligible_priced_by_market": priced[priced["d1_eligible"]].groupby("market").size().to_dict(),
        "kr_tax_class": priced[priced["market"] == "KR"].groupby("kr_tax_class").size().to_dict(),
        "listing_status": meta.groupby(["market", "listing_status"]).size().to_dict(),
        "priced_by_listing_status": priced.groupby(["market", "listing_status"]).size().to_dict(),
        "d1_eligible_priced_by_listing_status":
            priced[priced["d1_eligible"]].groupby(["market", "listing_status"]).size().to_dict(),
        "needs_review_priced_by_listing_status":
            priced[priced["needs_review"]].groupby(["market", "listing_status"]).size().to_dict(),
    }


def write_atomic(meta: pd.DataFrame, out_path: Path) -> None:
    """같은 폴더의 임시 파일에 쓴 뒤 교체한다. 실패하면 기존 산출물을 그대로 둔다."""
    if meta.empty:
        raise ValueError("메타데이터가 비어 있어 쓰지 않습니다.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".tmp", dir=out_path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        meta.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def default_universe_path() -> Path:
    return paths.raw_root().parent / UNIVERSE_FILENAME


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETF 분류 메타데이터 생성")
    parser.add_argument("--universe", type=Path, default=None,
                        help=f"유니버스 CSV 경로(기본: $%s/../%s)" % (paths.ENV_RAW_ROOT, UNIVERSE_FILENAME))
    parser.add_argument("--output", type=Path, default=None,
                        help=f"출력 CSV 경로(기본: $%s/%s/%s)" % (paths.ENV_DATA_ROOT, paths.METADATA_DIR, OUTPUT_FILENAME))
    parser.add_argument("--kr-history", type=Path, default=None,
                        help="폐지 국내 ETF 이력표(기본: $%s/%s/%s). 같은 폴더의 snapshots/로 이름을 대조한다."
                        % (paths.ENV_DATA_ROOT, KR_UNIVERSE_DIR, KR_HISTORY_FILENAME))
    parser.add_argument("--prices-bundle", type=Path, default=None,
                        help="has_price_history·listing_status를 대조할 가격 묶음 폴더(기본: 가격 저장소 CURRENT)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    universe_path = args.universe or default_universe_path()
    listed = load_universe(universe_path)
    history_path = args.kr_history or (paths.data_root() / KR_UNIVERSE_DIR / KR_HISTORY_FILENAME)
    snapshots = Path(history_path).parent / KR_SNAPSHOT_SUBDIR
    if not snapshots.is_dir():
        raise FileNotFoundError(f"폐지 ETF 이름 대조용 KRX 스냅샷 폴더가 없습니다: {snapshots}")
    delisted = load_kr_delisted(history_path, snapshots)
    meta = build_metadata(combine_universe(listed, delisted))
    bundle = args.prices_bundle
    if bundle is None:
        from .price_store import current_bundle  # 가격 저장소 묶음 위치는 이 함수로만 얻는다
        # CURRENT가 없거나 손상됐으면 FileNotFoundError로 멈춘다(대조 없이 쓰지 않는다).
        bundle = current_bundle(paths.data_root() / paths.PRICES_DIR)
    checked = check_price_bundle(meta, bundle)
    print(f"price bundle check OK: {bundle} {checked}")
    summary = summarize(meta)  # 검증·요약이 끝난 뒤에만 쓴다.
    out_path = args.output or (paths.output_dir(paths.METADATA_DIR) / OUTPUT_FILENAME)
    write_atomic(meta, out_path)
    print(f"wrote {out_path} ({len(meta)} rows)")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
