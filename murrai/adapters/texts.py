import urllib.request


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
