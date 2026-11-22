"""src.common.krx_session 단위 테스트. 외부 호출 0회(가짜 HTTP), pykrx 불필요(CI 편입 대상)."""

import contextlib
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common import krx_session as ks


SECRET_ID = "user-id-should-not-leak"
SECRET_PW = "pw-should-not-leak"
DATA_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def uniform(self, low, high):
        return low

    def as_clock(self):
        return ks.KrxClock(monotonic=self.monotonic, sleep=self.sleep, uniform=self.uniform)


class FakeResponse:
    def __init__(self, status_code=200, text="{}", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeHttp:
    """로그인 3단계는 성공, 데이터 요청은 data_responses 순서대로 돌려준다."""

    def __init__(self, data_responses=(), login_codes=("CD001",)):
        self.headers = {}
        self.calls = []
        self._data = list(data_responses)
        self._login_codes = list(login_codes)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if "MDCCOMS001D1" in url:
            return FakeResponse(200, payload={"_error_code": self._login_codes.pop(0)})
        if "COMS" in url:
            return FakeResponse(200, text="<html>... LOGOUT menu ...</html>")
        item = self._data.pop(0) if self._data else FakeResponse(200, text='{"OutBlock_1": []}')
        if isinstance(item, Exception):
            raise item
        return item

    def data_calls(self):
        return [c for c in self.calls if "COMS" not in c[1]]


LOCK_DIR = tempfile.TemporaryDirectory()
_LOCK_COUNTER = [0]


def tearDownModule():
    LOCK_DIR.cleanup()


class Lock:
    """테스트마다 별도 잠금 파일."""

    def __init__(self):
        _LOCK_COUNTER[0] += 1
        self.path = str(Path(LOCK_DIR.name) / f"krx-{_LOCK_COUNTER[0]}.lock")

    def make(self):
        return ks.KrxRunLock(self.path)


def make_session(policy=None, http=None, clock=None, lock=None):
    clock = clock or FakeClock()
    http = http or FakeHttp()
    lock = lock or Lock()
    creds = ks.KrxCredentials(SECRET_ID, SECRET_PW)
    session = ks.KrxSession(creds, policy or ks.KrxThrottlePolicy(), http, clock.as_clock(), lock.make())
    return session, http, clock


class TestPolicyAndLimiter(unittest.TestCase):
    def test_default_policy_matches_runbook(self):
        policy = ks.KrxThrottlePolicy()
        self.assertEqual((policy.base_delay_seconds, policy.jitter_max_seconds), (4.0, 2.0))
        self.assertEqual(policy.macro_pause_every, 40)
        self.assertEqual((policy.macro_pause_min_seconds, policy.macro_pause_max_seconds), (45.0, 70.0))
        self.assertEqual((policy.block_cooldown_seconds, policy.max_block_events), (600.0, 1))

    def test_negative_or_inverted_values_are_rejected(self):
        for kwargs in ({"base_delay_seconds": -1}, {"max_krx_calls": -1}, {"macro_pause_min_seconds": 80}, {"max_block_events": 0}):
            with self.assertRaises(ValueError):
                ks.KrxThrottlePolicy(**kwargs)

    def test_second_call_waits_base_delay_and_macro_pause_every_n(self):
        session, _, clock = make_session(ks.KrxThrottlePolicy(macro_pause_every=3))
        for _ in range(4):
            session.post(DATA_URL)
        self.assertEqual(clock.sleeps, [4.0, 4.0, 49.0])
        self.assertEqual(session.stats()["macro_pauses"], 1)

    def test_non_krx_host_is_not_counted_in_krx_budget(self):
        session, _, _ = make_session(ks.KrxThrottlePolicy(max_krx_calls=1))
        session.get("http://fchart.stock.naver.com/sise.nhn")
        session.post(DATA_URL)
        self.assertEqual((session.stats()["krx_calls"], session.stats()["other_calls"]), (1, 1))


class TestBudgetAndPermanentStop(unittest.TestCase):
    def test_budget_exceeded_stops_before_sending_and_stays_stopped(self):
        session, http, _ = make_session(ks.KrxThrottlePolicy(max_krx_calls=2))
        session.post(DATA_URL)
        session.post(DATA_URL)
        with self.assertRaises(ks.KrxBudgetExceededError):
            session.post(DATA_URL)
        with self.assertRaises(ks.KrxBudgetExceededError):
            session.get(DATA_URL)
        self.assertEqual(len(http.data_calls()), 2)
        self.assertIn("budget", session.stats()["stopped"])

    def test_login_requests_count_toward_budget(self):
        session, _, _ = make_session(ks.KrxThrottlePolicy(max_krx_calls=3))
        session.login()
        with self.assertRaises(ks.KrxBudgetExceededError):
            session.post(DATA_URL)


class TestBlockHandling(unittest.TestCase):
    def test_default_stops_on_first_block_without_cooldown_or_retry(self):
        for status in (403, 429, 500, 502, 503, 504):
            session, http, clock = make_session(http=FakeHttp([FakeResponse(status), FakeResponse(200)]))
            with self.assertRaises(ks.KrxBlockedError):
                session.post(DATA_URL)
            self.assertEqual(len(http.data_calls()), 1)
            self.assertEqual(clock.sleeps.count(600.0), 0)
            with self.assertRaises(ks.KrxBlockedError):
                session.post(DATA_URL)
            self.assertEqual(len(http.data_calls()), 1)  # 멈춘 뒤 가짜 전송 0회

    def test_logout_body_stops_on_first_response(self):
        session, http, _ = make_session(http=FakeHttp([FakeResponse(400, text="LOGOUT")]))
        with self.assertRaises(ks.KrxBlockedError):
            session.post(DATA_URL)
        self.assertEqual(len(http.data_calls()), 1)

    def test_optional_second_chance_cools_down_relogins_and_retries(self):
        http = FakeHttp([FakeResponse(403), FakeResponse(200, text='{"OutBlock_1": [1]}')])
        session, http, clock = make_session(ks.KrxThrottlePolicy(max_block_events=2), http=http)
        self.assertEqual(session.post(DATA_URL).status_code, 200)
        self.assertIn(600.0, clock.sleeps)
        self.assertEqual((session.stats()["logins"], session.stats()["block_events"]), (1, 1))

    def test_second_block_stops_and_later_calls_never_reach_network(self):
        http = FakeHttp([FakeResponse(403), FakeResponse(429)])
        session, http, clock = make_session(ks.KrxThrottlePolicy(max_block_events=2), http=http)
        with self.assertRaises(ks.KrxBlockedError):
            session.post(DATA_URL)
        sent = len(http.calls)
        for _ in range(3):  # 옛 수집기가 예외를 삼키고 다시 불러도 전송하지 않는다
            with self.assertRaises(ks.KrxBlockedError):
                session.post(DATA_URL)
        self.assertEqual(len(http.calls), sent)
        self.assertEqual(clock.sleeps.count(600.0), 1)

    def test_logout_body_and_5xx_are_block_like(self):
        self.assertTrue(ks.looks_blocked(FakeResponse(400, text="LOGOUT")))
        self.assertTrue(ks.looks_blocked(FakeResponse(400, text="logout")))
        self.assertTrue(ks.looks_blocked(FakeResponse(503)))
        self.assertFalse(ks.looks_blocked(FakeResponse(200, text='{"OutBlock_1": []}')))

    def test_blocked_login_step_stops(self):
        class BlockedLoginHttp(FakeHttp):
            def request(self, method, url, **kwargs):
                self.calls.append((method, url, kwargs))
                return FakeResponse(403)

        session, _, _ = make_session(http=BlockedLoginHttp())
        with self.assertRaises(ks.KrxBlockedError):
            session.login()


class TestRedirectsAndUnexpectedResponses(unittest.TestCase):
    def test_every_send_disables_redirect_following(self):
        session, http, _ = make_session()
        session.login()
        session.post(DATA_URL)
        self.assertTrue(all(c[2].get("allow_redirects") is False for c in http.calls))

    def test_redirect_response_stops_without_following(self):
        session, http, _ = make_session(http=FakeHttp([FakeResponse(302)]))
        with self.assertRaises(ks.KrxResponseError) as ctx:
            session.post(DATA_URL)
        self.assertEqual(str(ctx.exception), "redirect_refused_http_302")
        with self.assertRaises(ks.KrxResponseError):
            session.post(DATA_URL)
        self.assertEqual(len(http.data_calls()), 1)

    def test_other_http_error_stops_immediately(self):
        session, http, _ = make_session(http=FakeHttp([FakeResponse(404, text="secret body")]))
        with self.assertRaises(ks.KrxResponseError) as ctx:
            session.post(DATA_URL)
        self.assertNotIn("secret", str(ctx.exception))

    def test_transport_error_message_is_only_a_class_name(self):
        error = requests.exceptions.ConnectionError(f"failed with {SECRET_PW} in url")
        session, _, _ = make_session(http=FakeHttp([error]))
        with self.assertRaises(ks.KrxTransportError) as ctx:
            session.post(DATA_URL)
        self.assertEqual(str(ctx.exception), "transport_error:ConnectionError")


class TestSerialization(unittest.TestCase):
    def test_send_and_response_handling_never_overlap(self):
        active, overlaps = [0], [0]

        class SlowHttp(FakeHttp):
            def request(self, method, url, **kwargs):
                active[0] += 1
                if active[0] > 1:
                    overlaps[0] += 1
                time.sleep(0.01)
                active[0] -= 1
                return FakeResponse(200, text='{"OutBlock_1": []}')

        policy = ks.KrxThrottlePolicy(base_delay_seconds=0, jitter_max_seconds=0, macro_pause_every=0)
        session, _, _ = make_session(policy, SlowHttp(), clock=None)
        session._clock = ks.KrxClock()
        session._limiter._clock = ks.KrxClock()
        threads = [threading.Thread(target=lambda: [session.post(DATA_URL) for _ in range(3)]) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(overlaps[0], 0)
        self.assertEqual(session.stats()["krx_calls"], 12)


class TestRunLock(unittest.TestCase):
    def test_second_session_on_same_lock_is_refused_until_close(self):
        lock = Lock()
        first, _, _ = make_session(lock=lock)
        second, http2, _ = make_session(lock=lock)
        first.login()
        with self.assertRaises(ks.KrxLockError):
            second.login()
        self.assertEqual(http2.calls, [])
        first.close()
        second.login()
        second.close()


class TestLockOwnership(unittest.TestCase):
    def test_request_before_login_takes_the_run_lock(self):
        lock = Lock()
        first, http1, _ = make_session(lock=lock)
        second, http2, _ = make_session(lock=lock)
        first.post(DATA_URL)
        with self.assertRaises(ks.KrxLockError):
            second.post(DATA_URL)
        self.assertEqual(http2.calls, [])
        first.close()

    def test_closed_session_refuses_without_network(self):
        session, http, _ = make_session()
        session.login()
        session.close()
        sent = len(http.calls)
        with self.assertRaises(ks.KrxLockError):
            session.post(DATA_URL)
        self.assertEqual(len(http.calls), sent)


class TestSilentLogin(unittest.TestCase):
    def test_login_prints_nothing_and_handles_duplicate_login(self):
        session, http, _ = make_session(http=FakeHttp(login_codes=("CD011", "CD001")))
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            session.login()
        self.assertEqual(sink.getvalue(), "")
        login_posts = [c for c in http.calls if "MDCCOMS001D1" in c[1]]
        self.assertEqual(login_posts[-1][2]["data"].get("skipDup"), "Y")

    def test_login_failure_message_is_fixed_and_hides_server_code(self):
        session, _, _ = make_session(http=FakeHttp(login_codes=("SERVER-TEXT-123",)))
        with self.assertRaises(ks.KrxLoginError) as ctx:
            session.login()
        message = str(ctx.exception)
        self.assertEqual(message, "login_rejected")
        for leaked in (SECRET_PW, SECRET_ID, "SERVER-TEXT"):
            self.assertNotIn(leaked, message)

    def test_credentials_repr_and_stats_hide_values(self):
        self.assertNotIn(SECRET_PW, repr(ks.KrxCredentials(SECRET_ID, SECRET_PW)))
        session, _, _ = make_session()
        session.login()
        self.assertNotIn(SECRET_PW, str(session.stats()))


class TestCredentialLoading(unittest.TestCase):
    def test_environment_variables(self):
        creds = ks.load_krx_credentials(environ={"KRX_LOGIN_ID": "a", "KRX_LOGIN_PW": "b"})
        self.assertEqual((creds.login_id, creds.login_pw), ("a", "b"))

    def test_private_file_is_read_and_shared_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "krx.env"
            path.write_text("export KRX_LOGIN_ID='a'\nKRX_LOGIN_PW=\"b\"\n")
            os.chmod(path, 0o600)
            creds = ks.load_krx_credentials(str(path), environ={})
            self.assertEqual((creds.login_id, creds.login_pw), ("a", "b"))
            os.chmod(path, 0o644)
            with self.assertRaises(ks.KrxLoginError):
                ks.load_krx_credentials(str(path), environ={})

    def test_missing_credentials_raise(self):
        with self.assertRaises(ks.KrxLoginError):
            ks.load_krx_credentials(environ={})


if __name__ == "__main__":
    unittest.main()
