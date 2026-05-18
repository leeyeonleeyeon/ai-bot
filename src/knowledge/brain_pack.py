"""Knowledge / brain pack utilities — fetch URLs and inject as agent context."""
from __future__ import annotations

from html.parser import HTMLParser

import httpx


def is_url(s: str) -> bool:
    s = s.lower().strip()
    return s.startswith("http://") or s.startswith("https://")


class _HtmlStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: D401
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip:
            return
        d = data.strip()
        if d:
            self._parts.append(d)

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


async def fetch_url(url: str, max_chars: int = 20000) -> str:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "ai-bot/0.1"})
        resp.raise_for_status()
        text = resp.text

    # 간단한 HTML 본문 추출
    if "<html" in text.lower() or "<body" in text.lower():
        s = _HtmlStripper()
        try:
            s.feed(text)
            text = s.text
        except Exception:
            pass  # 파싱 실패 시 원문 그대로

    return text[:max_chars]
