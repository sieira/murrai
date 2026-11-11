import pytest

from murrai.adapters.texts import UrlText, strip_between


def test_strip_between_cuts_around_markers():
    text = (
        "license header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "the actual text\n"
        "more of the actual text\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "long license footer\n"
    )
    assert (
        strip_between(
            text,
            "START OF THE PROJECT GUTENBERG EBOOK",
            "END OF THE PROJECT GUTENBERG EBOOK",
        )
        == "the actual text\nmore of the actual text\n"
    )


def test_strip_between_fails_when_a_marker_is_missing():
    with pytest.raises(ValueError):
        strip_between("no markers here", "START", "END")


def test_url_text_fetches_strips_and_caches(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return (
            "license header\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
            "body text\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
            "license footer\n"
        )

    cache = tmp_path / "body.txt"
    source = UrlText(
        "http://example.com/book",
        cache=cache,
        start="START OF THE PROJECT GUTENBERG EBOOK",
        end="END OF THE PROJECT GUTENBERG EBOOK",
        fetch=fake_fetch,
    )
    assert source.text() == "body text\n"
    assert cache.read_text() == "body text\n"
    assert calls == ["http://example.com/book"]
    assert source.text() == "body text\n"
    assert calls == ["http://example.com/book"]


def test_url_text_fails_when_markers_are_incomplete(tmp_path):
    with pytest.raises(ValueError):
        UrlText("http://example.com/book", cache=tmp_path / "x.txt", start="S").text()
