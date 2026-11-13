import itertools
import random
import time

import pytest

from murrai.domain.ports import Progress
from murrai.domain.tokenizer import TokenizerEncoderDecoder, _replace

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


def _naive_fit(text, num_merges):
    ids = list(text.encode("utf-8"))
    merges = []
    for _ in range(num_merges):
        counts = {}
        for pair in itertools.pairwise(ids):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)
        ids = _replace(ids, best, 256 + len(merges) - 1)
    return tuple(merges)


ADVERSARIAL_CORPORA = [
    "aaaa",
    "aaaaa",
    "aaabdaaabac",
    "abab",
    "abababab",
    "aaaaaaaab",
    "abcabcabca",
    "aabb" * 50,
]


@pytest.mark.parametrize("text", ADVERSARIAL_CORPORA)
def test_fit_matches_a_full_recount_on_adversarial_corpora(text):
    fast = TokenizerEncoderDecoder().fit(text, num_merges=12)
    assert fast.merges == _naive_fit(text, num_merges=12)


def test_fit_on_a_large_corpus_does_not_recount_every_pair():
    words = random.Random(0).choices(
        "abcdefghij",
        k=300000,
    )
    text = "".join(words)
    start = time.perf_counter()
    TokenizerEncoderDecoder().fit(text, num_merges=300)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize("text", ADVERSARIAL_CORPORA)
def test_encode_matches_the_sequential_replay(text):
    t = TokenizerEncoderDecoder().fit(text, num_merges=12)
    expected = list(text.encode("utf-8"))
    for rank, pair in enumerate(t.merges):
        expected = _replace(expected, pair, 256 + rank)
    assert t.encode(text) == expected


def test_encode_on_a_large_corpus_does_not_replay_merge_by_merge():
    text = "".join(random.Random(0).choices("abcdefghij", k=300000))
    t = TokenizerEncoderDecoder().fit(text, num_merges=300)
    start = time.perf_counter()
    t.encode(text)
    assert time.perf_counter() - start < 1.5


def test_encode_reports_progress_per_merge_replay():
    t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=3)
    seen = []

    class FakeBus:
        def emit(self, event):
            seen.append(event)

    t.encode(CORPUS, sink=FakeBus())
    assert [(event.step, event.total_steps) for event in seen] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert all(isinstance(event, Progress) for event in seen)


def test_fit_publishes_progress_through_a_bus():
    class FakeBus:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

    bus = FakeBus()
    t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=3, sink=bus)
    assert [(event.step, event.total_steps) for event in bus.events] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert len(t.merges) == 3


def test_fit_stays_silent_without_a_bus():
    t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=2)
    assert len(t.merges) == 2
