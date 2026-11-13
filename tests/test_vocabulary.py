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


def test_grown_with_keeps_every_existing_id():
    v = Vocabulary.from_text("ba")
    grown = v.grown_with("cd")
    assert v.encode("ba") == [1, 0]
    assert grown.encode("ba") == [1, 0]
    assert grown.chars == ("a", "b", "c", "d")


def test_grown_with_sorts_new_chars_among_themselves():
    v = Vocabulary.from_text("b")
    grown = v.grown_with("dca")
    assert grown.chars == ("b", "a", "c", "d")


def test_grown_with_ignores_known_chars():
    v = Vocabulary.from_text("ab")
    assert v.grown_with("bab").chars == ("a", "b")
