import argparse
import getpass
import json
import os
import sys
from datetime import date

import requests
from pykrx import stock
from pykrx.website.comm import webio

LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
LOGIN_JSP = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
SHORT_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
SHORT_BLD = "dbms/MDC/STAT/srt/MDCSTAT30001"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--login-id")
    p.add_argument("--login-pw")
    p.add_argument("--date", default="20260227")
    p.add_argument("--ticker", default="005930")
    return p.parse_args()


def get_creds(args):
    login_id = args.login_id or os.getenv("KRX_LOGIN_ID")
    login_pw = args.login_pw or os.getenv("KRX_LOGIN_PW")
    if not login_id:
        login_id = input("KRX login id: ").strip()
    if not login_pw:
        login_pw = getpass.getpass("KRX login pw: ")
    return login_id, login_pw


def install_shared_session(session):
    def _session_post_read(self, **params):
        return session.post(self.url, headers=self.headers, data=params, timeout=20)

    def _session_get_read(self, **params):
        return session.get(self.url, headers=self.headers, params=params, timeout=20)

    webio.Post.read = _session_post_read
    webio.Get.read = _session_get_read


def login_krx(session, login_id, login_pw):
    session.headers.update({"User-Agent": UA})
    print("[1/4] seed login page", flush=True)
    r1 = session.get(LOGIN_PAGE, headers={"User-Agent": UA}, timeout=20)
    print("  status=", r1.status_code, flush=True)

    print("[2/4] seed iframe jsp", flush=True)
    r2 = session.get(
        LOGIN_JSP,
        headers={"User-Agent": UA, "Referer": LOGIN_PAGE},
        timeout=20,
    )
    print("  status=", r2.status_code, flush=True)

    payload = {
        "mbrNm": "",
        "telNo": "",
        "di": "",
        "certType": "",
        "mbrId": login_id,
        "pw": login_pw,
    }
    headers = {"User-Agent": UA, "Referer": LOGIN_PAGE}

    print("[3/4] login POST", flush=True)
    resp = session.post(LOGIN_URL, data=payload, headers=headers, timeout=20)
    print("  status=", resp.status_code, flush=True)
    data = resp.json()
    print("  body=", data, flush=True)
    error_code = data.get("_error_code", "")

    if error_code == "CD011":
        print("[4/4] duplicate-login skipDup=Y", flush=True)
        payload["skipDup"] = "Y"
        resp = session.post(LOGIN_URL, data=payload, headers=headers, timeout=20)
        print("  status=", resp.status_code, flush=True)
        data = resp.json()
        print("  body=", data, flush=True)
        error_code = data.get("_error_code", "")

    return error_code == "CD001", data


def direct_short_probe(session, day, ticker):
    print("[probe] direct short endpoint", flush=True)
    payload = {
        "bld": SHORT_BLD,
        "locale": "ko_KR",
        "inqTpCd": "1",
        "trdDd": day,
        "isuCd": ticker,
        "isuCd2": ticker,
        "codeNmisuCd_finder_stkisu0_0": "",
        "param1isuCd_finder_stkisu0_0": "ALL",
    }
    resp = session.post(
        SHORT_URL,
        headers={"User-Agent": UA, "Referer": LOGIN_PAGE},
        data=payload,
        timeout=20,
    )
    print("  http_status=", resp.status_code, flush=True)
    print("  content_type=", resp.headers.get("Content-Type"), flush=True)
    head = resp.text[:300].replace("\n", " ")
    print("  body_head=", head, flush=True)
    try:
        data = resp.json()
        keys = sorted(list(data.keys()))
        output = data.get("output") or []
        print("  json_keys=", keys, flush=True)
        print("  output_rows=", len(output), flush=True)
    except Exception as e:
        print("  json_parse_error=", repr(e), flush=True)


def pykrx_smoke(day, ticker):
    print("[pykrx] ETF ticker list smoke", flush=True)
    try:
        etf = stock.get_etf_ticker_list(day)
        print("  etf_count=", len(etf), flush=True)
        print("  etf_sample=", etf[:5], flush=True)
    except Exception as e:
        print("  etf_error=", repr(e), flush=True)

    print("[pykrx] short status smoke", flush=True)
    try:
        df = stock.get_shorting_status_by_date(day, day, ticker)
        print("  short_rows=", 0 if df is None else len(df), flush=True)
        if df is not None and len(df):
            print(df.head(3).to_string(), flush=True)
    except Exception as e:
        print("  short_error=", repr(e), flush=True)


def main():
    args = parse_args()
    login_id, login_pw = get_creds(args)
    session = requests.Session()
    ok, data = login_krx(session, login_id, login_pw)
    print("login_ok=", ok, flush=True)
    print("cookies=", session.cookies.get_dict(), flush=True)
    if not ok:
        print("login failed; stop", flush=True)
        return 1

    install_shared_session(session)
    direct_short_probe(session, args.date, args.ticker)
    pykrx_smoke(args.date, args.ticker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
