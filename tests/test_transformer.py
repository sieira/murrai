import math

import pytest
import torch

from murrai.application.runs import train_transformer_lm
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.domain.transformer import TransformerLM


@pytest.fixture
def tokenizer():
    return TokenizerEncoderDecoder()


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(0)
    return TransformerLM(tokenizer, depth=2, dimension=16, max_time=8)


def test_logits_answer_the_vocabulary(model, tokenizer):
    logits = model.logits([0, 1])
    assert logits.shape == (tokenizer.vocab_size,)


def test_different_order_of_the_same_chars_different_logits(model):
    assert not torch.allclose(model.logits([0, 1]), model.logits([1, 0]))


def test_an_empty_history_answers_through_the_start(model, tokenizer):
    row = model.probs([])
    assert row.shape == (tokenizer.vocab_size,)
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-5)


def test_generation_wanders_past_the_window(model):
    long = model.generate(3 * model.max_time, seed=3)
    assert long == model.generate(3 * model.max_time, seed=3)
    assert len(long) >= 3 * model.max_time


def test_the_state_round_trips(model):
    rebuilt = TransformerLM.from_state(
        TokenizerEncoderDecoder(model.tokenizer.merges),
        model.state(),
    )
    torch.testing.assert_close(rebuilt.logits([0, 1, 0]), model.logits([0, 1, 0]))


def test_generation_is_deterministic_and_decodes(model):
    first = model.generate(12, seed=3)
    assert first == model.generate(12, seed=3)
    assert len(first) >= 12


def test_nll_matches_a_hand_computed_case(model, tokenizer):
    model.head.weight.data.zero_()
    bias = torch.zeros(tokenizer.vocab_size)
    bias[ord("b")] = math.log(3.0)
    model.head.bias.data.copy_(bias)
    total = 255 + 3.0
    expected = -(math.log(1 / total) + math.log(3.0 / total)) / 2
    assert model.nll("ab") == pytest.approx(expected)


def test_nll_walks_long_text_in_batched_rows(model):
    calls = []
    original = model.forward

    def counting(rows):
        calls.append(rows.size(0))
        return original(rows)

    model.forward = counting
    model.nll("ab" * 301)
    assert len(calls) == 1


def test_nll_matches_the_one_row_at_a_time_value(model):
    text = "ab" * 301
    ids = [model.start, *model.vocabulary.encode(text)]
    stream = torch.tensor(ids, dtype=torch.long)
    losses = []
    for start in range(0, len(stream) - 1, model.max_time):
        row = stream[start : start + model.max_time]
        with torch.no_grad():
            logits = model.forward(row[None])
        losses.append(
            torch.nn.functional.cross_entropy(
                logits[0, :-1],
                stream[start + 1 : start + model.max_time],
            ),
        )
    expected = float(torch.stack(losses).mean())
    assert model.nll(text) == pytest.approx(expected)


def test_training_lowers_loss_on_a_toy_corpus():
    result = train_transformer_lm(
        "ab" * 200,
        steps=40,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
    )
    assert result.loss_curve[-1] < result.loss_curve[0]


def test_training_survives_a_corpus_that_compresses_below_the_window():
    result = train_transformer_lm(
        "aabbaabb" * 20,
        steps=5,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=64,
    )
    assert math.isfinite(result.loss_curve[-1])


def test_training_reports_held_out_perplexity():
    result = train_transformer_lm(
        "ab" * 400,
        steps=5,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0.2,
    )
    assert math.isfinite(result.held_out_perplexity)
    assert result.held_out_perplexity >= 1.0


def test_the_plateau_schedule_shrinks_the_rate_during_a_stalled_run():
    result = train_transformer_lm(
        "abab" * 50,
        steps=60,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        learning_rate=0.1,
        patience=3,
    )
    assert result.rate_curve[-1] < result.rate_curve[0]


def test_a_four_head_state_round_trips_to_identical_outputs(tokenizer):
    torch.manual_seed(0)
    model = TransformerLM(tokenizer, depth=2, dimension=16, max_time=8, heads=4)
    rebuilt = TransformerLM.from_state(
        TokenizerEncoderDecoder(model.tokenizer.merges),
        model.state(),
    )
    assert rebuilt.blocks[0].attend.heads == 4
    assert torch.equal(rebuilt.logits([0, 1, 0]), model.logits([0, 1, 0]))


def test_a_state_saved_before_heads_existed_loads_as_one_head(model, tokenizer):
    legacy = model.state()
    del legacy["heads"]
    rebuilt = TransformerLM.from_state(
        TokenizerEncoderDecoder(model.tokenizer.merges),
        legacy,
    )
    assert rebuilt.blocks[0].attend.heads == 1
    torch.testing.assert_close(rebuilt.logits([0, 1, 0]), model.logits([0, 1, 0]))


def test_training_accepts_a_head_count():
    result = train_transformer_lm(
        "ab" * 200,
        steps=10,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        heads=2,
    )
    assert result.model.blocks[0].attend.heads == 2


def test_training_honors_a_requested_device():
    result = train_transformer_lm(
        "ab" * 200,
        steps=5,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        device=torch.device("cpu"),
    )
    assert next(result.model.parameters()).device.type == "cpu"


def test_the_requested_device_trains_the_same_curve():
    text = "ab" * 500
    default = train_transformer_lm(
        text,
        steps=10,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
    )
    explicit = train_transformer_lm(
        text,
        steps=10,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        device=torch.device("cpu"),
    )
    assert explicit.loss_curve == default.loss_curve
