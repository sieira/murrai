import pytest

from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.presentation.chat import ChatSession


@pytest.fixture
def model():
    return Bigram.fit("abab", TokenizerEncoderDecoder())


def test_user_and_generated_chars_build_the_story(model):
    session = ChatSession(model, seed=0)
    session.extend_user("ab")
    session.extend_generated("a")
    assert session.text == "aba"
    assert session.segments() == [("ab", "entered"), ("a", "generated")]


def test_truncate_rolls_the_story_back(model):
    session = ChatSession(model, seed=0)
    session.extend_user("ab")
    session.extend_generated("b")
    session.truncate(2)
    assert session.text == "ab"
    assert session.segments() == [("ab", "entered")]


def test_text_absent_from_the_corpus_extends_without_corrupting(model):
    session = ChatSession(model, seed=0)
    session.extend_user("a")
    session.extend_user("z🚀中")
    assert session.text == "az🚀中"


def test_rollout_samples_the_next_n_tokens(model):
    session = ChatSession(model, seed=0)
    ghost = session.rollout("a", 5)
    assert len(ghost) >= 5


def test_rollout_is_deterministic_per_seed(model):
    first = ChatSession(model, seed=7)
    second = ChatSession(model, seed=7)
    assert first.rollout("ab", 12) == second.rollout("ab", 12)


def test_rollout_does_not_change_the_story(model):
    session = ChatSession(model, seed=0)
    session.extend_user("a")
    session.rollout("a", 5)
    assert session.text == "a"


def test_the_die_is_sorted_and_sums_to_one(model):
    session = ChatSession(model, seed=0)
    session.extend_user("a")
    dice = session.dice()
    probabilities = [p for _, p in dice]
    assert probabilities == sorted(probabilities, reverse=True)
    assert pytest.approx(sum(probabilities)) == 1.0


def test_the_die_reflects_the_whole_story(model):
    session = ChatSession(model, seed=0)
    session.extend_user("a")
    dice = session.dice()
    session.extend_user("ab")
    dice_after = session.dice()
    assert dice != dice_after


class _Forcing:
    kind = "forced"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def probs(self, context_ids):
        row = [0.0] * self.tokenizer.vocab_size
        row[256] = 1.0
        return row


def test_the_die_renders_a_partial_utf8_token_as_replacement_characters():
    tokenizer = TokenizerEncoderDecoder(merges=((0xC3, 0x41),))
    model = Bigram.fit("ab", tokenizer)
    session = ChatSession(model, seed=0)
    session.extend_user("a")
    dice = session.dice()
    assert "�A" in [piece for piece, _ in dice]


def test_rollout_renders_a_partial_utf8_token_without_crashing():
    session = ChatSession(_Forcing(TokenizerEncoderDecoder(merges=((0xC3, 0x41),))), seed=0)
    assert session.rollout("", 3) == "�A�A�A"


class Recording:
    tokenizer = TokenizerEncoderDecoder()

    def __init__(self):
        self.contexts = []

    def probs(self, context_ids):
        self.contexts.append(list(context_ids))
        return [0.5, 0.5]


def test_the_die_feeds_the_whole_story_to_the_port():
    recorder = Recording()
    session = ChatSession(recorder, seed=0)
    session.extend_user("a")
    session.extend_generated("b")
    session.dice()
    assert recorder.contexts == [[ord("a"), ord("b")]]


@pytest.fixture
def neural_model():
    return train_neural_bigram("abab", steps=3, batch=4, seed=0).model


def test_session_serves_the_neural_bigram_the_same_way(neural_model):
    session = ChatSession(neural_model, seed=0)
    session.extend_user("ab")
    dice = session.dice()
    probabilities = [p for _, p in dice]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("ab", 3)
