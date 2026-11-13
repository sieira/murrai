import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "fetch_gutenberg.py"
spec = importlib.util.spec_from_file_location("fetch_gutenberg", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def _recorder(calls):
    def record(url, cache, start, end):
        calls.append((url, Path(cache), start, end))
        return SimpleNamespace(text=lambda: "x" * 10)

    return record


def test_the_themes_partition_the_books():
    themed = [book for books in fetch.THEMES.values() for book in books.items()]

    assert sorted(themed) == sorted(fetch.BOOKS.items())


def test_main_without_a_theme_fetches_every_book(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(fetch, "UrlText", _recorder(calls))

    fetch.main()

    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines] == [*fetch.BOOKS.values(), "total"]
    assert lines[-1] == "total: 370 chars"
    assert [url for url, _, _, _ in calls] == [
        f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt"
        for book in fetch.BOOKS
    ]
    assert [cache for _, cache, _, _ in calls] == [
        Path("data/corpus") / f"{slug}.txt" for slug in fetch.BOOKS.values()
    ]
    assert all(start == fetch.START and end == fetch.END for *_, start, end in calls)


def test_main_with_a_theme_fetches_it_into_its_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(fetch, "UrlText", _recorder(calls))

    fetch.main("philosophy")

    assert [cache for _, cache, _, _ in calls] == [
        Path("data") / "philosophy" / f"{slug}.txt"
        for slug in fetch.THEMES["philosophy"].values()
    ]


def test_an_unknown_theme_names_the_available_themes():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "cookbooks"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cookbooks" in result.stderr
    assert all(theme in result.stderr for theme in fetch.THEMES)
