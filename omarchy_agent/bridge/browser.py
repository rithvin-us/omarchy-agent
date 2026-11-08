"""Playwright-backed browser control for the local agent.

One lazily started headless Chromium page is shared per ``BrowserController``.
Only http(s) URLs are allowed; file:, chrome:, and other schemes are refused.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MAX_TEXT_CHARS = 6_000
MAX_LINKS = 25
NAV_TIMEOUT_MS = 20_000
ACTION_TIMEOUT_MS = 5_000


class BrowserError(RuntimeError):
    """Raised when a browser action is refused or fails."""


def validate_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BrowserError("only absolute http(s) URLs are allowed")
    return parsed.geturl()


def _find_chromium() -> str | None:
    """Prefer a system Chromium so no matching Playwright download is required."""
    for name in ("chromium", "chromium-browser", "google-chrome-stable", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


class BrowserController:
    def __init__(self, *, headless: bool = True, screenshot_dir: Path | None = None) -> None:
        self.headless = headless
        self.screenshot_dir = screenshot_dir
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None

    # -- lifecycle -------------------------------------------------------
    def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserError("playwright is not installed (uv add playwright)") from exc
        try:
            self._pw = sync_playwright().start()
            launch: dict[str, Any] = {"headless": self.headless}
            executable = _find_chromium()
            if executable:
                launch["executable_path"] = executable
            self._browser = self._pw.chromium.launch(**launch)
            self._page = self._browser.new_page()
            self._page.set_default_timeout(ACTION_TIMEOUT_MS)
        except Exception as exc:  # noqa: BLE001 - playwright raises its own error types
            self.close()
            raise BrowserError(f"could not start browser: {exc}") from exc
        return self._page

    def close(self) -> None:
        for obj, method in ((self._browser, "close"), (self._pw, "stop")):
            if obj is not None:
                with contextlib.suppress(Exception):
                    getattr(obj, method)()
        self._pw = self._browser = self._page = None

    # -- actions ---------------------------------------------------------
    def goto(self, url: str) -> dict[str, Any]:
        page = self._ensure_page()
        try:
            page.goto(validate_url(url), timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"navigation failed: {exc}") from exc
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        page = self._ensure_page()
        try:
            text = page.inner_text("body")
            links = page.eval_on_selector_all(
                "a[href]",
                "els => els.slice(0, 200).map(e => ({text: e.innerText.trim().slice(0, 80),"
                " href: e.href})).filter(l => l.text)",
            )
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"snapshot failed: {exc}") from exc
        return {
            "url": page.url,
            "title": page.title(),
            "text": text[:MAX_TEXT_CHARS],
            "truncated": len(text) > MAX_TEXT_CHARS,
            "links": links[:MAX_LINKS],
        }

    def click(self, target: str) -> dict[str, Any]:
        """Click by CSS selector, or by visible text when prefixed with ``text=``."""
        page = self._ensure_page()
        try:
            page.click(target)
            page.wait_for_load_state("domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"click failed: {exc}") from exc
        return self.snapshot()

    def type_text(self, selector: str, text: str, *, submit: bool = False) -> dict[str, Any]:
        page = self._ensure_page()
        try:
            page.fill(selector, text)
            if submit:
                page.press(selector, "Enter")
                page.wait_for_load_state("domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"type failed: {exc}") from exc
        return self.snapshot()

    def screenshot(self) -> dict[str, Any]:
        page = self._ensure_page()
        directory = self.screenshot_dir or Path(tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix="omarchy-shot-", suffix=".png", delete=False
        ) as handle:
            path = Path(handle.name)
        try:
            page.screenshot(path=str(path))
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"screenshot failed: {exc}") from exc
        return {"path": str(path), "url": page.url}
