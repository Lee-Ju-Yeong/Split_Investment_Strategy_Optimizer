"""Jekyll 생성물 또는 공개 Pages의 주요 방문 경로를 Chromium으로 확인한다."""

import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


PREFIX = "/Split_Investment_Strategy_Optimizer/"
PAGES = ("", "docs/engineering-evidence.html", "tests/", "docs/", "src/",
         "docs/running.html", "docs/testing-mysql.html")
NAV = ("소개", "구현 근거", "직접 확인", "문서 안내", "GitHub 코드")
NAV_PATHS = ("", "docs/engineering-evidence.html", "tests/", "docs/")
REPO = "https://github.com/Lee-Ju-Yeong/Split_Investment_Strategy_Optimizer"


class PreviewHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if not path.startswith(PREFIX):
            return str(Path(self.directory) / "__outside_preview__")
        return super().translate_path("/" + path[len(PREFIX):])

    def log_message(self, *_args):
        pass


@contextmanager
def serve(directory):
    if not (directory / "index.html").is_file():
        raise ValueError("Jekyll이 생성한 index.html이 없습니다.")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(PreviewHandler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}{PREFIX}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def wait_for_revision(base, revision):
    # main의 기존 Pages 배포를 기다릴 뿐 배포 권한/요청은 사용하지 않는다.
    deadline = time.monotonic() + 90
    while True:
        try:
            with urlopen(base + "?revision=" + revision, timeout=5) as response:
                html = response.read().decode("utf-8")
            if f'name="siso-revision" content="{revision}"' in html:
                return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("공개 Pages에서 이번 병합 SHA를 90초 안에 확인하지 못했습니다.")
        time.sleep(3)


def check_site(base, expected_revision=None):
    checked_links = set()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            page = context.new_page()
            page.set_default_timeout(10000)
            for width in (1280, 390):
                page.set_viewport_size({"width": width, "height": 900 if width == 1280 else 844})
                for path in PAGES:
                    address = urljoin(base, path)
                    response = page.goto(address, wait_until="load", timeout=30000)
                    if response.status != 200:
                        raise AssertionError(f"문서 응답 {response.status}: {address}")
                    if expected_revision is not None:
                        actual_revision = page.locator('meta[name="siso-revision"]').get_attribute("content")
                        if actual_revision != expected_revision:
                            raise AssertionError(f"공개 문서 SHA 불일치: {path or '/'}: {actual_revision}")
                    if page.locator("html").get_attribute("lang") != "ko-KR":
                        raise AssertionError(f"한국어 lang 누락: {path}")
                    if page.locator('meta[property="og:locale"]').get_attribute("content") != "ko_KR":
                        raise AssertionError(f"한국어 공유 메타데이터 누락: {path}")
                    if page.locator("h1").count() != 1:
                        raise AssertionError(f"H1은 문서 제목 하나여야 합니다: {path}")
                    levels = page.locator("main h1,main h2,main h3,main h4").evaluate_all(
                        "els => els.map(e => Number(e.tagName.slice(1)))")
                    if any(b > a + 1 for a, b in zip(levels, levels[1:])):
                        raise AssertionError(f"제목 단계가 건너뜁니다: {path}")
                    nav = page.locator('nav[aria-label="주요 문서"] a')
                    if nav.all_text_contents() != list(NAV):
                        raise AssertionError(f"공통 메뉴 불일치: {path}")
                    expected_hrefs = [urljoin(base, p) for p in NAV_PATHS] + [REPO]
                    actual_hrefs = nav.evaluate_all("els => els.map(e => e.href)")
                    if actual_hrefs != expected_hrefs:
                        raise AssertionError(f"공통 메뉴 목적지 불일치: {path}")
                    if page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"):
                        raise AssertionError(f"{width}px 화면 전체가 가로로 넘칩니다: {path}")
                    if not page.locator("main").is_visible():
                        raise AssertionError(f"본문이 보이지 않습니다: {path}")
                    images = page.locator("main img")
                    if images.evaluate_all("els => els.some(e => !e.alt.trim())"):
                        raise AssertionError(f"이미지 대체 설명 누락: {path}")
                    # 같은 사이트의 링크·앵커·그림·CSS만 확인. 외부 GitHub/API 접속은 별도 범위다.
                    resources = page.locator("main a[href],nav a[href],main img[src],link[rel=stylesheet]")
                    urls = resources.evaluate_all("els => els.map(e => e.href || e.src)")
                    for target in urls:
                        if not target.startswith(base) or target in checked_links:
                            continue
                        parts = urlsplit(target)
                        fetch_url = urlunsplit(parts._replace(fragment=""))
                        reply = context.request.get(fetch_url, timeout=15000)
                        if reply.status != 200:
                            raise AssertionError(f"깨진 링크 {reply.status}: {target}")
                        if parts.fragment:
                            html = reply.text()
                            from html.parser import HTMLParser

                            class Anchors(HTMLParser):
                                def __init__(self):
                                    super().__init__()
                                    self.ids = set()

                                def handle_starttag(self, tag, attrs):
                                    values = dict(attrs)
                                    self.ids.update(v for k, v in values.items() if k == "id" or (tag == "a" and k == "name"))

                            anchors = Anchors()
                            anchors.feed(html)
                            if unquote(parts.fragment) not in anchors.ids:
                                raise AssertionError(f"앵커 없음: {target}")
                        checked_links.add(target)
                    print(f"PASS {width}px: {path or '/'}", flush=True)

                # 새 페이지의 최초 키보드 초점부터 실제 Tab 순서를 확인한다.
                keyboard_page = context.new_page()
                keyboard_page.set_viewport_size({"width": width, "height": 900})
                keyboard_page.goto(base, wait_until="load", timeout=30000)
                keyboard_page.keyboard.press("Tab")
                if keyboard_page.locator(":focus").inner_text() != "본문으로 바로 이동":
                    raise AssertionError("첫 키보드 이동이 본문 바로가기여야 합니다.")
                if keyboard_page.locator(":focus").bounding_box()["y"] < 0:
                    raise AssertionError("본문 바로가기 초점이 화면 밖입니다.")
                keyboard_page.keyboard.press("Tab")  # 사이트 이름
                for name in NAV:
                    keyboard_page.keyboard.press("Tab")
                    focused = keyboard_page.locator(":focus")
                    if focused.inner_text() != name:
                        raise AssertionError(f"키보드 메뉴 순서 불일치: {name}")
                    if focused.evaluate("e => getComputedStyle(e).outlineStyle") == "none":
                        raise AssertionError("키보드 초점 표시가 없습니다.")
                for _ in range(6):
                    keyboard_page.keyboard.press("Shift+Tab")
                keyboard_page.keyboard.press("Enter")
                if keyboard_page.locator(":focus").get_attribute("id") != "main-content":
                    raise AssertionError("본문 바로가기가 초점을 옮기지 않습니다.")
                keyboard_page.close()
                print(f"PASS {width}px: 키보드 본문 이동·메뉴 5개·초점 표시", flush=True)
            context.close()
        finally:
            browser.close()
    return len(checked_links)


def report(base, count):
    text = (f"## 문서 화면 검사\n\n- 대상: {base}\n"
            f"- 주요 문서 {len(PAGES)}개 × 화면 1280px·390px 통과\n"
            f"- 내부 링크·앵커·자료 {count}개 통과\n"
            "- 한국어 메타데이터·H1·제목 단계·공통 메뉴·키보드 초점 확인\n"
            "- 문서 검사이며 제품·GPU·DB 실행을 대신하지 않습니다.\n")
    print(text, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("_site"))
    parser.add_argument("--base-url")
    parser.add_argument("--expected-revision")
    args = parser.parse_args()
    if args.base_url:
        base = args.base_url.rstrip("/") + "/"
        if args.expected_revision:
            wait_for_revision(base, args.expected_revision)
        report(base, check_site(base, args.expected_revision))
    else:
        with serve(args.site_dir.resolve()) as base:
            report(base, check_site(base, args.expected_revision))


if __name__ == "__main__":
    main()
