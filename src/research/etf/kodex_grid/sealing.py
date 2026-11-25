"""봉인 도구: 한 번만 쓰기, 출력 폴더의 단일 작성자 잠금, 작업 출력 가두기, 공개용 기술 오류 코드.

결과 파일은 한 번 쓰면 다른 내용으로 바꾸지 않는다. 조립 전에 공개되는 것은 단위 이름·시간·기술 오류 코드뿐이고,
작업 내부의 표준출력·표준오류·예외 전문은 봉인 위치에만 남긴다.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading

from ..kodex_core.metrics import plain


PUBLIC, SEALED = 0o644, 0o600
ERROR_CODES = {"IdentityError": "E_IDENTITY", "InputIdentityError": "E_IDENTITY", "PermitError": "E_IDENTITY",
               "InputIntegrityError": "E_INTEGRITY", "OverwriteError": "E_OVERWRITE", "MemoryError": "E_MEMORY"}
NFS_TRACE = re.compile(r"\.nfs[0-9a-f]{24}")  # 네트워크 파일시스템이 지운 파일 자리에 남기는 흔적의 이름 형식
_HELD = {}  # 폴더 잠금: 경로 -> _Hold. 포크한 작업 자식은 부모의 항목을 물려받는다
_GUARD = threading.Lock()  # _HELD를 보고 바꾸는 구간


class OverwriteError(RuntimeError):
    """이미 기록된 파일을 다른 내용으로 바꾸려 했다. 덮어쓰지 않고 멈춘다."""


class BusyError(RuntimeError):
    """같은 출력 폴더에 다른 실행이 쓰고 있다. 기다리지 않고 멈춘다(기존 실행은 그대로 둔다)."""


def dump(record) -> str:
    return json.dumps(plain(record), ensure_ascii=False, indent=1, sort_keys=True)


def encoded(text: str) -> bytes:
    """파일에 쓰는 바이트. 기록의 해시와 동일성은 모두 이 바이트로 약속한다."""
    return text.encode("utf-8")


def bytes_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(text: str) -> str:
    return bytes_digest(encoded(text))


def file_digest(path) -> str:
    return bytes_digest(Path(path).read_bytes())


def _temporary(path: Path, text: str, mode: int) -> Path:
    """같은 폴더의 임시 파일에 약속한 바이트를 그대로 다 쓰고 디스크에 내린다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "wb") as stream:
        stream.write(encoded(text))
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def write_marker(path: Path, text: str) -> None:
    """바꿔 써도 되는 기술 표시(오류 표시 등). 임시 파일에 쓴 뒤 이름을 바꾼다."""
    os.replace(_temporary(Path(path), text, PUBLIC), path)


def write_once(path: Path, text: str, *, mode: int = PUBLIC) -> bool:
    """없으면 임시 파일에 다 쓴 뒤 이름을 바꿔 새로 만든다. 이미 있으면 바이트가 같을 때만 통과하고, 다르면 멈춘다.
    존재 확인과 생성 사이의 경쟁은 출력 폴더의 작성자 잠금이 막는다(하드 링크는 네트워크 파일시스템에 흔적 파일을 남긴다)."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != encoded(text):
            raise OverwriteError(f"{path.name} already exists with different content; it is not overwritten")
        return False
    os.replace(_temporary(path, text, mode), path)
    return True


def listed(folder: Path) -> set:
    """폴더의 파일 이름. 네트워크 파일시스템의 흔적(.nfs + 16진수 24자)만 결과가 아닌 것으로 뺀다."""
    folder = Path(folder)
    if not folder.is_dir():
        return set()
    return {path.name for path in folder.iterdir() if NFS_TRACE.fullmatch(path.name) is None}


def clear_temporaries(folder: Path) -> None:
    """끊긴 쓰기가 남긴 임시 파일을 지운다. 폴더 잠금을 잡은 뒤에만 부른다."""
    for leftover in Path(folder).rglob(".*.tmp"):
        leftover.unlink()


@dataclass
class _Hold:
    """한 폴더의 작성자 잠금: 잠금 파일, 잡은 프로세스와 스레드, 같은 호출 흐름의 겹침 수, 이 잠금 구간의 표시."""

    handle: object
    owner: tuple
    depth: int
    token: str


def _flow() -> tuple:
    return os.getpid(), threading.get_ident()


def _key(folder) -> str:
    return str(Path(folder).resolve())


def _flocked(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    handle = open(folder / ".writer.lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise BusyError(f"another run is writing to {folder}; this run stops and leaves it alone") from None
    return handle


def _enter(folder) -> _Hold:
    """같은 호출 흐름(같은 프로세스·같은 스레드)의 겹침만 허용한다. 같은 프로세스의 다른 스레드는 독립 작성자다."""
    key = _key(folder)
    with _GUARD:
        hold = _HELD.get(key)
        if hold is not None and hold.owner == _flow():
            hold.depth += 1
            return hold
        if hold is not None and hold.owner[0] == os.getpid():
            raise BusyError(f"another thread is writing to {folder}; this run stops and leaves it alone")
        hold = _HELD[key] = _Hold(_flocked(Path(folder)), _flow(), 1, os.urandom(8).hex())
        return hold


def _leave(folder, hold: _Hold) -> None:
    with _GUARD:
        hold.depth -= 1
        if hold.depth == 0:
            _HELD.pop(_key(folder), None)
            fcntl.flock(hold.handle, fcntl.LOCK_UN)
            hold.handle.close()


@contextmanager
def writer_lock(folder):
    """출력 폴더마다 작성자는 하나다. 잠금을 잡은 호출 흐름 안에서만 겹쳐 잡을 수 있다."""
    hold = _enter(folder)
    try:
        yield
    finally:
        _leave(folder, hold)


def writer_token(folder):
    """이 호출 흐름이 그 폴더의 작성자 잠금 안에 있으면 그 잠금 구간의 표시를, 아니면 None을 돌려준다.
    잠금을 잡은 프로세스가 포크한 작업 자식은 부모의 잠금 아래에 있다. 같은 프로세스의 다른 스레드는 아니다."""
    hold = _HELD.get(_key(folder))
    if hold is None:
        return None
    forked_worker = hold.owner[0] == os.getppid() and hold.owner[0] != os.getpid()
    return hold.token if hold.owner == _flow() or forked_worker else None


def flush_standard() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()


@contextmanager
def _descriptors_to(target: int):
    """파일 기술자 1·2(자식 프로세스와 C 수준 출력 포함)를 잠시 target으로 돌린다."""
    flush_standard()
    saved = [os.dup(1), os.dup(2)]
    os.dup2(target, 1)
    os.dup2(target, 2)
    try:
        yield
    finally:
        flush_standard()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


@contextmanager
def captured_output(path: Path):
    """작업 내부의 표준출력·표준오류를 봉인 파일로만 보낸다(순차·병렬 공통)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, SEALED), "w", encoding="utf-8") as sink:
        with _descriptors_to(sink.fileno()), redirect_stdout(sink), redirect_stderr(sink):
            try:
                yield sink
            finally:
                sink.flush()


def error_code(error: BaseException) -> str:
    """공개해도 되는 기술 오류 코드. 예외의 문구는 싣지 않는다."""
    name = type(error).__name__
    if name in ERROR_CODES:
        return ERROR_CODES[name]
    return "E_OS" if isinstance(error, OSError) else "E_ENGINE"


def seal_of(body: dict) -> str:
    """기록 본문의 자기 해시(seal 칸 제외). 손으로 고친 기록을 가려낸다."""
    return digest(dump({key: value for key, value in body.items() if key != "seal"}))


def with_seal(body: dict) -> dict:
    return {**body, "seal": seal_of(body)}


def read_sealed(path: Path, schema: str) -> dict:
    """자기 해시가 맞는 기록만 읽는다. 없거나 꼴이 다르면 빈 사전이 아니라 오류다."""
    record = json.loads(Path(path).read_bytes())
    if not isinstance(record, dict) or record.get("schema") != schema or record.get("seal") != seal_of(record):
        raise ValueError(f"{Path(path).name}: not an intact {schema} record")
    return record
