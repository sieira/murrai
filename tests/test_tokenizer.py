import pytest

from murrai.domain.tokenizer import TokenizerEncoderDecoder

CORPUS = "aaabdaaabac"


def test_fit_learns_merge_order_with_stable_tie_break():
    t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=3)
    a, b = (ord(x) for x in "ab")
    assert t.merges == ((a, a), (a, b), (256, 257))


def test_fit_tie_breaks_to_lowest_pair():
    counts_tie = "abab"
    t = TokenizerEncoderDecoder().fit(counts_tie, num_merges=1)
    assert t.merges == ((ord("a"), ord("b")),)


def test_encode_replays_merges_in_learned_order():
    t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=3)
    a, d, c = (ord(x) for x in "adc")
    assert t.encode(CORPUS) == [258, d, 258, a, c]
    assert t.decode([258, d, 258, a, c]) == "aaabdaaabac"


def test_encode_resolves_overlaps_left_to_right():
    t = TokenizerEncoderDecoder().fit("aaaa", num_merges=1)
    assert t.encode("aaaa") == [256, 256]
    assert t.encode("aaa") == [256, ord("a")]


def test_encode_rule_fires_only_after_earlier_rules_ran():
    t = TokenizerEncoderDecoder().fit("aab", num_merges=2)
    a, b = ord("a"), ord("b")
    assert t.merges == ((a, a), (256, b))
    assert t.encode("aab") == [257]
    assert t.encode("ab") == [a, b]


def test_fresh_tokenizer_encodes_raw_bytes_without_fit():
    t = TokenizerEncoderDecoder()
    assert t.encode("ab") == [ord("a"), ord("b")]
    assert t.decode([ord("a"), ord("b")]) == "ab"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "aaabdaaabac",
        "line one\nline two\r\n\ttabbed",
        "naïve café — résumé",
        "中文测试 and Ελληνικά",
        "emoji 🚀🔥 family 👨‍👩‍👧‍👦 done",
        "a" * 500,
        "mixed 🚀 aaabdaaabac 中文\nnewline",
    ],
)
def test_round_trip_on_varied_text(text):
    t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=8)
    assert t.decode(t.encode(text)) == text


def test_round_trip_on_text_absent_from_training_corpus():
    t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=8)
    unseen = "Zzz Qq! @#$% 100% pure 🦄 unicode ♜☲Ж"
    assert t.decode(t.encode(unseen)) == unseen


def test_vocab_size_is_256_plus_merges():
    t = TokenizerEncoderDecoder()
    assert t.vocab_size == 256
    t.fit(CORPUS, num_merges=5)
    assert t.vocab_size == 261


def test_a_tokenizer_can_be_rebuilt_from_prior_merges():
    fitted = TokenizerEncoderDecoder().fit(CORPUS, num_merges=2)
    rebuilt = TokenizerEncoderDecoder(fitted.merges)
    assert rebuilt.merges == fitted.merges
    assert rebuilt.vocab_size == fitted.vocab_size
    assert rebuilt.encode(CORPUS) == fitted.encode(CORPUS)
    assert rebuilt.decode(rebuilt.encode(CORPUS)) == CORPUS


def test_decode_replaces_invalid_bytes_instead_of_raising():
    t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=2)
    assert t.decode([0x80]) == "�"
    assert t.decode([0xFF, 0x80, ord("a")]) == "��a"


def test_pieces_render_every_token_with_replacement_for_partial_utf8():
    t = TokenizerEncoderDecoder(merges=((0xC3, 0x41),))
    pieces = t.pieces
    assert len(pieces) == t.vocab_size
    assert pieces[ord("a")] == "a"
    assert pieces[256] == "�A"


def test_fit_encode_decode_are_separate_operations():
    t = TokenizerEncoderDecoder()
    assert t.merges == ()
    t2 = t.fit(CORPUS, num_merges=2)
    assert t2 is t
    assert t.merges == ((ord("a"), ord("a")), (ord("a"), ord("b")))
    assert t.encode("azb") == [ord("a"), ord("z"), ord("b")]
    assert t.decode(t.encode("azb")) == "azb"
