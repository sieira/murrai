import pytest

from murrai.domain.vocabulary import Vocabulary


def test_round_trip():
    text = "abracadabra"
    v = Vocabulary.from_text(text)
    assert v.decode(v.encode(text)) == text


def test_unique_sorted_ids():
    v = Vocabulary.from_text("ba")
    assert v.chars == ("a", "b")


def test_unknown_char_raises():
    v = Vocabulary.from_text("ab")
    with pytest.raises(ValueError):
        v.encode("abc")
