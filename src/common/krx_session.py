"""
krx_session.py

KRX 호출을 하나의 로그인 세션, 하나의 속도 제한기, 하나의 실행 잠금으로 모은다.

왜 필요한가:
- 이 서버는 2026-02-11부터 KRX egress 403 차단을 겪었고, 2026-02-27 이후에는
  로그인 세션 없이는 응답이 비거나 LOGOUT이 온다
  (docs/operations/krx_hotspot_db_tunnel_runbook.md).
- pykrx(webio.Get/Post)와 market_cap/short_selling 수집기의 직접 requests 경로가
  서로 다른 세션·속도로 호출하면 차단 위험이 커진다.

규칙:
- 자동 리디렉션을 따라가지 않는다(추가 전송이 제한기·예산을 우회하지 않게).
- 전송부터 응답 판정·휴식·재로그인까지 한 잠금 안에서 처리한다(요청 겹침 없음).
- 같은 호스트의 다른 실행과 동시에 돌지 않도록 파일 잠금을 잡는다.
- 차단·예산 초과·예상 밖 응답으로 한 번 멈추면 그 세션은 이후 모든 전송을 거부한다.
- 자격 값·쿠키·응답 본문·서버 오류 문자열·하위 예외 원문은 출력·예외 메시지에 넣지 않는다.
  메시지는 고정 분류명과 정수 HTTP 상태만 쓴다.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import random
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Optional
from urllib.parse import urlparse

import requests


KRX_HOST_SUFFIX = "krx.co.kr"
LOGIN_PAGE_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
LOGIN_JSP_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
LOGIN_POST_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
LOGIN_OK_CODE = "CD001"
LOGIN_DUPLICATE_CODE = "CD011"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
BLOCK_LIKE_STATUS_CODES = frozenset({403, 429, 500, 502, 503, 504})
BLOCK_BODY_MARKER = "LOGOUT"
BLOCK_BODY_SCAN_CHARS = 300
CREDENTIALS_FILE_ENV = "KRX_CREDENTIALS_FILE"
LOGIN_ID_ENV = "KRX_LOGIN_ID"
LOGIN_PW_ENV = "KRX_LOGIN_PW"
LOCK_FILE_ENV = "KRX_SESSION_LOCK"
DEFAULT_LOCK_FILE = "/tmp/split-krx-session.lock"


class KrxSessionError(RuntimeError):
    """KRX 세션 계층 오류의 공통 부모. 메시지는 고정 분류명만 담는다."""


class KrxLoginError(KrxSessionError):
    """로그인 실패."""


class KrxBlockedError(KrxSessionError):
    """차단 유사 응답이 허용 횟수에 도달해 IP 보호를 위해 중단."""


class KrxBudgetExceededError(KrxSessionError):
    """설정한 최대 KRX 호출 수를 넘으려 해서 중단."""


class KrxResponseError(KrxSessionError):
    """리디렉션·예상 밖 HTTP 상태처럼 성공으로 볼 수 없는 응답."""


class KrxTransportError(KrxSessionError):
    """연결·시간 초과 등 전송 단계 오류(하위 예외 원문은 버린다)."""


class KrxLockError(KrxSessionError):
    """다른 실행이 KRX 세션 잠금을 잡고 있음."""


@dataclass(frozen=True)
class KrxThrottlePolicy:
    """기본값 근거: KRX runbook 3절 수집 명령(workers 1, 4s+0~2s, 40회마다 45~70s)."""

    base_delay_seconds: float = 4.0
    jitter_max_seconds: float = 2.0
    macro_pause_every: int = 40
    macro_pause_min_seconds: float = 45.0
    macro_pause_max_seconds: float = 70.0
    block_cooldown_seconds: float = 600.0
    max_block_events: int = 1  # 사용자 결정(11-22): 403·429·5xx·LOGOUT 첫 응답에서 즉시 중단
    max_krx_calls: Optional[int] = None
    request_timeout_seconds: float = 20.0
    other_host_delay_seconds: float = 1.0

    def __post_init__(self):
        numbers = (
            self.base_delay_seconds, self.jitter_max_seconds, self.macro_pause_min_seconds,
            self.macro_pause_max_seconds, self.block_cooldown_seconds,
            self.request_timeout_seconds, self.other_host_delay_seconds,
        )
        if any(value < 0 for value in numbers) or self.macro_pause_min_seconds > self.macro_pause_max_seconds:
            raise ValueError("invalid_throttle_policy")
        if self.macro_pause_every < 0 or self.max_block_events < 1:
            raise ValueError("invalid_throttle_policy")
        if self.max_krx_calls is not None and self.max_krx_calls < 0:
            raise ValueError("invalid_throttle_policy")


@dataclass(frozen=True)
class KrxCredentials:
    login_id: str = field(repr=False)
    login_pw: str = field(repr=False)

    def __repr__(self):
        return "KrxCredentials(<hidden>)"


@dataclass(frozen=True)
class KrxClock:
    """테스트에서 시간·대기·난수를 바꿔 끼우기 위한 묶음."""

    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    uniform: Callable[[float, float], float] = random.uniform


def _parse_env_file(path: Path) -> dict:
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        values[key] = value.strip().strip('"').strip("'")
    return values


def _assert_private_file(path: Path) -> None:
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise KrxLoginError("credential_file_not_private")


def load_krx_credentials(
    credentials_file: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> KrxCredentials:
    """우선순위: 인자 파일 > KRX_CREDENTIALS_FILE 파일 > KRX_LOGIN_ID/PW 환경변수."""
    env = os.environ if environ is None else environ
    file_path = credentials_file or env.get(CREDENTIALS_FILE_ENV)
    source = env
    if file_path:
        path = Path(file_path)
        if not path.is_file():
            raise KrxLoginError("credential_file_missing")
        _assert_private_file(path)
        source = _parse_env_file(path)
    login_id = source.get(LOGIN_ID_ENV, "")
    login_pw = source.get(LOGIN_PW_ENV, "")
    if not login_id or not login_pw:
        raise KrxLoginError("credentials_not_found")
    return KrxCredentials(login_id=login_id, login_pw=login_pw)


def is_krx_url(url: str) -> bool:
    host = (urlparse(str(url)).hostname or "").lower()
    return host == KRX_HOST_SUFFIX or host.endswith("." + KRX_HOST_SUFFIX)


def looks_blocked(response) -> bool:
    """403/429/5xx 또는 본문 앞부분의 LOGOUT 표식. 본문은 판정에만 쓰고 보관하지 않는다."""
    if response.status_code in BLOCK_LIKE_STATUS_CODES:
        return True
    try:
        head = (response.text or "")[:BLOCK_BODY_SCAN_CHARS]
    except Exception:
        return False
    return BLOCK_BODY_MARKER in head.upper()


class KrxRateLimiter:
    """모든 KRX 요청이 지나는 관문. 호출자는 KrxSession의 직렬화 잠금 안에서 부른다."""

    def __init__(self, policy: KrxThrottlePolicy, clock: KrxClock = KrxClock()):
        self._policy = policy
        self._clock = clock
        self._next_allowed = None
        self.krx_calls = 0
        self.other_calls = 0
        self.macro_pauses = 0

    def acquire(self, is_krx: bool) -> None:
        self._check_budget(is_krx)
        self._sleep_until_allowed()
        self._count(is_krx)
        self._next_allowed = self._clock.monotonic() + self._next_gap(is_krx)

    def _check_budget(self, is_krx: bool) -> None:
        limit = self._policy.max_krx_calls
        if is_krx and limit is not None and self.krx_calls >= int(limit):
            raise KrxBudgetExceededError("krx_call_budget_exhausted")

    def _sleep_until_allowed(self) -> None:
        if self._next_allowed is None:
            return
        wait_seconds = self._next_allowed - self._clock.monotonic()
        if wait_seconds > 0:
            self._clock.sleep(wait_seconds)

    def _count(self, is_krx: bool) -> None:
        if is_krx:
            self.krx_calls += 1
        else:
            self.other_calls += 1

    def _next_gap(self, is_krx: bool) -> float:
        policy = self._policy
        if not is_krx:
            return float(policy.other_host_delay_seconds)
        gap = policy.base_delay_seconds + self._clock.uniform(0.0, policy.jitter_max_seconds)
        every = int(policy.macro_pause_every or 0)
        if every > 0 and self.krx_calls % every == 0:
            self.macro_pauses += 1
            gap += self._clock.uniform(policy.macro_pause_min_seconds, policy.macro_pause_max_seconds)
        return gap


class KrxRunLock:
    """호스트 단위 단일 실행 잠금(fcntl). 같은 계정의 두 실행이 서로 세션을 밀어내지 않게 한다."""

    def __init__(self, path: Optional[str] = None):
        self._path = Path(path or os.environ.get(LOCK_FILE_ENV) or DEFAULT_LOCK_FILE)
        self._fd = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise KrxLockError("krx_session_lock_held") from None
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


class KrxSession:
    """로그인 + 속도 제한 + 차단 감지 + 실행 잠금을 한곳에서 처리하는 requests 래퍼."""

    def __init__(
        self,
        credentials: KrxCredentials,
        policy: KrxThrottlePolicy = KrxThrottlePolicy(),
        http_session=None,
        clock: KrxClock = KrxClock(),
        run_lock: Optional[KrxRunLock] = None,
    ):
        self._credentials = credentials
        self._policy = policy
        self._http = http_session if http_session is not None else requests.Session()
        self._http.max_redirects = 0
        self._clock = clock
        self._limiter = KrxRateLimiter(policy, clock)
        self._io_lock = threading.RLock()
        self._run_lock = run_lock if run_lock is not None else KrxRunLock()
        self._tripped: Optional[KrxSessionError] = None
        self.block_events = 0
        self.logins = 0

    # ---- 수명 ----------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def close(self) -> None:
        # 닫은 세션은 다시 쓰지 못한다(잠금 없이 전송하는 경로를 막는다).
        if self._tripped is None:
            self._tripped = KrxLockError("session_closed")
        self._run_lock.release()

    @property
    def tripped(self) -> Optional[str]:
        return str(self._tripped) if self._tripped else None

    def _trip(self, error: KrxSessionError) -> KrxSessionError:
        if self._tripped is None:
            self._tripped = error
        return error

    # ---- 로그인 -------------------------------------------------------
    def login(self) -> None:
        with self._io_lock:
            self._run_lock.acquire()
            self._http.headers.update({"User-Agent": BROWSER_USER_AGENT})
            self._send_login_step("GET", LOGIN_PAGE_URL)
            self._send_login_step("GET", LOGIN_JSP_URL, headers={"Referer": LOGIN_PAGE_URL})
            error_code = self._post_login(skip_duplicate=False)
            if error_code == LOGIN_DUPLICATE_CODE:
                error_code = self._post_login(skip_duplicate=True)
            if error_code != LOGIN_OK_CODE:
                raise self._trip(KrxLoginError("login_rejected"))
            self.logins += 1

    def _send_login_step(self, method: str, url: str, **kwargs):
        # 로그인 페이지 HTML에는 LOGOUT 글자가 있을 수 있어 상태 코드만 본다.
        response = self._send(method, url, **kwargs)
        status = int(response.status_code)
        if status in BLOCK_LIKE_STATUS_CODES:
            raise self._trip(KrxBlockedError(f"login_blocked_http_{status}"))
        if status != 200:
            raise self._trip(KrxLoginError(f"login_unexpected_http_{status}"))
        return response

    def _post_login(self, skip_duplicate: bool) -> str:
        payload = {
            "mbrNm": "", "telNo": "", "di": "", "certType": "",
            "mbrId": self._credentials.login_id,
            "pw": self._credentials.login_pw,
        }
        if skip_duplicate:
            payload["skipDup"] = "Y"
        response = self._send_login_step(
            "POST", LOGIN_POST_URL, data=payload, headers={"Referer": LOGIN_PAGE_URL}
        )
        try:
            return str(response.json().get("_error_code", ""))
        except Exception:
            raise self._trip(KrxLoginError("login_non_json_response")) from None

    # ---- 일반 요청 ----------------------------------------------------
    def request(self, method: str, url: str, **kwargs):
        with self._io_lock:
            if not is_krx_url(url):
                return self._send(method, url, **kwargs)
            while True:
                response = self._send(method, url, **kwargs)
                if looks_blocked(response):
                    self._handle_block(int(response.status_code))
                    continue
                if int(response.status_code) != 200:
                    raise self._trip(KrxResponseError(f"unexpected_http_{int(response.status_code)}"))
                return response

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def _send(self, method: str, url: str, **kwargs):
        with self._io_lock:
            if self._tripped is not None:
                raise type(self._tripped)(f"session_stopped:{self._tripped}")
            self._run_lock.acquire()  # 로그인 전 요청도 실행 간 잠금을 먼저 잡는다(이미 잡았으면 그대로)
            try:
                self._limiter.acquire(is_krx_url(url))
            except KrxBudgetExceededError as exc:
                raise self._trip(exc)
            kwargs["allow_redirects"] = False
            kwargs.setdefault("timeout", self._policy.request_timeout_seconds)
            try:
                response = self._http.request(method, url, **kwargs)
            except requests.RequestException as exc:
                raise self._trip(KrxTransportError(f"transport_error:{type(exc).__name__}")) from None
            if 300 <= int(response.status_code) < 400:
                raise self._trip(KrxResponseError(f"redirect_refused_http_{int(response.status_code)}"))
            return response

    def _handle_block(self, status_code: int) -> None:
        self.block_events += 1
        if self.block_events >= int(self._policy.max_block_events):
            raise self._trip(KrxBlockedError(f"blocked_http_{status_code}_event_{self.block_events}"))
        self._clock.sleep(float(self._policy.block_cooldown_seconds))
        self.login()

    def stats(self) -> dict:
        return {
            "krx_calls": self._limiter.krx_calls,
            "other_calls": self._limiter.other_calls,
            "macro_pauses": self._limiter.macro_pauses,
            "block_events": self.block_events,
            "logins": self.logins,
            "max_krx_calls": self._policy.max_krx_calls,
            "stopped": self.tripped,
        }

    # ---- pykrx / 수집기 경로 주입 ------------------------------------
    def direct_requests_namespace(self) -> SimpleNamespace:
        """수집기 모듈의 `requests` 이름을 대신할 객체(post/get/exceptions)."""
        return SimpleNamespace(
            post=self.post, get=self.get, request=self.request, exceptions=requests.exceptions,
        )

    @contextlib.contextmanager
    def installed(self, collector_modules=()):
        """pykrx webio와 수집기 직접 호출을 이 세션으로 돌리고, 끝나면 원래대로 되돌린다."""
        from pykrx.website.comm import webio

        session = self
        saved = [(webio.Post, "read", webio.Post.read), (webio.Get, "read", webio.Get.read)]

        def _post_read(io_self, **params):
            return session.post(io_self.url, headers=io_self.headers, data=params)

        def _get_read(io_self, **params):
            return session.get(io_self.url, headers=io_self.headers, params=params)

        webio.Post.read = _post_read
        webio.Get.read = _get_read
        namespace = self.direct_requests_namespace()
        for module in collector_modules:
            saved.append((module, "requests", getattr(module, "requests")))
            module.requests = namespace
        try:
            yield self
        finally:
            for owner, name, original in reversed(saved):
                setattr(owner, name, original)


def default_collector_modules():
    """직접 requests.post로 KRX를 부르는 기존 수집기 모듈."""
    from src import market_cap_collector, short_selling_collector

    return (market_cap_collector, short_selling_collector)
