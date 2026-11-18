"""확정 M1 가격 묶음의 경로 조회. 수집·자료 처리 의존성 없이 읽기만 한다."""
from pathlib import Path


RUNS_DIR = "runs"
CURRENT_POINTER = "CURRENT"
MANIFEST = "manifest.json"


def current_bundle(out: Path) -> Path:
    """CURRENT가 가리키는 확정 묶음 폴더. 기존 price_store API에서도 재노출한다."""
    pointer = out / CURRENT_POINTER
    if not pointer.is_file():
        raise FileNotFoundError(f"확정된 가격 묶음이 없습니다: {pointer}")
    bundle = out / RUNS_DIR / pointer.read_text(encoding="utf-8").strip()
    if not (bundle / MANIFEST).is_file():
        raise FileNotFoundError(f"CURRENT가 가리키는 묶음에 {MANIFEST}가 없습니다: {bundle}")
    return bundle
