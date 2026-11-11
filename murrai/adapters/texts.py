import urllib.request
from pathlib import Path


def fetch_text(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8-sig")


def strip_between(text, start, end):
    lines = text.splitlines()
    try:
        first = next(i for i, line in enumerate(lines) if start in line)
        last = next(i for i, line in enumerate(lines) if end in line and i > first)
    except StopIteration:
        raise ValueError(f"marker not found: {start!r} or {end!r}") from None
    return "\n".join(lines[first + 1 : last]).strip() + "\n"


class UrlText:
    def __init__(self, url, cache, start=None, end=None, fetch=fetch_text):
        self._url = url
        self._cache = Path(cache)
        self._start = start
        self._end = end
        self._fetch = fetch

    def text(self):
        if self._cache.exists():
            return self._cache.read_text()
        if (self._start is None) != (self._end is None):
            raise ValueError("start and end markers come together")
        text = self._fetch(self._url)
        if self._start is not None:
            text = strip_between(text, self._start, self._end)
        self._cache.parent.mkdir(parents=True, exist_ok=True)
        self._cache.write_text(text)
        return text
