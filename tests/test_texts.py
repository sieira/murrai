import pytest

from murrai.adapters.texts import strip_between


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
