"""연구 자료 위치. 대용량 자료는 저장소 밖에 둔다."""
from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_ROOT = "MAGICSPLIT_ETF_DATA_ROOT"
ENV_RAW_ROOT = "MAGICSPLIT_ETF_RAW_ROOT"

# M1 산출물 하위 폴더. 모듈마다 자기 폴더에만 쓴다(병렬 작업 충돌 방지).
PRICES_DIR = "prices"          # price_store.py: 정규화 일봉·기업행사·품질 표시
METADATA_DIR = "metadata"      # metadata.py: 레버리지·자산군·기초지수 묶음·과세 분류
KRX_GOLD_DIR = "krx_gold"      # krx_gold.py: KRX 금현물 일별 시세
FX_DIR = "fx"                  # fx.py: 원/달러 환율과 이용 가능 시각
PROTOCOL_DIR = "protocol"      # protocol.py: 프로토콜 JSON·탐색/보류 분할


def _required_dir(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} 환경 변수에 저장소 밖 자료 폴더 절대 경로를 지정하세요.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{env_name}는 절대 경로여야 합니다: {value}")
    return path


def data_root() -> Path:
    """M1 산출물 루트(쓰기)."""
    return _required_dir(ENV_DATA_ROOT)


def raw_root() -> Path:
    """ChatGPT 연구 원천 자료 루트(읽기 전용), 예: .../2026-11-15-chatgpt/extracted."""
    return _required_dir(ENV_RAW_ROOT)


def output_dir(name: str) -> Path:
    path = data_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
