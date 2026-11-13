import math

import pytest
import torch

from murrai.application.checkpoints import snapshot_of
from murrai.application.runs import plateau_rate, train_mlp_lm
from murrai.domain.mlp import MlpLM
from murrai.domain.vocabulary import Vocabulary


@pytest.fixture
def vocabulary():
    return Vocabulary.from_text("ab")


@pytest.fixture
def model(vocabulary):
    torch.manual_seed(0)
    return MlpLM(vocabulary, window=2, dimension=8, hidden=16)


def test_fresh_model_answers_the_port(model, vocabulary):
    row = model.probs([0])
    assert row.shape == (vocabulary.size,)
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-5)


def test_context_enters_the_model(model):
    assert not torch.allclose(model.probs([0]), model.probs([1, 0]))


def test_window_trims_to_the_last_k(model):
    assert torch.allclose(model.probs([1, 1, 0]), model.probs([1, 0]))


def test_empty_history_uses_the_start(model):
    row = model.probs([])
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-5)


def test_generation_is_deterministic_and_in_vocabulary(model):
    first = model.generate(12, seed=3)
    assert first == model.generate(12, seed=3)
    assert set(first) <= set(model.vocabulary.chars)


def test_held_out_perplexity_is_finite_and_above_one(model):
    value = model.perplexity("abba")
    assert math.isfinite(value)
    assert value >= 1.0


def test_plateau_rate_holds_until_patience_runs_out():
    assert plateau_rate(2, 1.0, patience=3, shrink=0.5) == 1.0


def test_plateau_rate_shrinks_once_patience_runs_out():
    assert plateau_rate(3, 1.0, patience=3, shrink=0.5) == 0.5
    assert plateau_rate(5, 2.0, patience=3, shrink=0.5) == 1.0


def test_training_lowers_loss_on_a_toy_corpus():
    result = train_mlp_lm("ab" * 200, steps=30, batch=16, seed=0)
    assert result.loss_curve[-1] < result.loss_curve[0]


def test_training_reports_held_out_perplexity():
    result = train_mlp_lm("ab" * 200, steps=5, batch=16, seed=0, holdout=0.2)
    assert math.isfinite(result.held_out_perplexity)
    assert result.held_out_perplexity >= 1.0


def test_the_plateau_schedule_shrinks_the_rate_during_a_stalled_run():
    result = train_mlp_lm(
        "abab" * 50,
        steps=40,
        batch=16,
        seed=0,
        learning_rate=1.0,
        patience=3,
    )
    assert result.rate_curve[-1] < result.rate_curve[0]


def test_training_honors_a_requested_device():
    result = train_mlp_lm(
        "abab" * 50,
        steps=10,
        batch=16,
        seed=0,
        device=torch.device("cpu"),
    )
    assert next(result.model.parameters()).device.type == "cpu"


def test_the_requested_device_trains_the_same_curve():
    default = train_mlp_lm("abab" * 200, steps=15, batch=16, seed=0)
    explicit = train_mlp_lm(
        "abab" * 200,
        steps=15,
        batch=16,
        seed=0,
        device=torch.device("cpu"),
    )
    assert explicit.loss_curve == default.loss_curve


def test_growth_fits_the_new_vocabulary(model):
    grown = model.grown_to(Vocabulary.from_text("abc"))
    assert grown.vocabulary.chars == ("a", "b", "c")
    assert grown.start == 3
    assert grown.embed.weight.shape == (4, 8)
    assert grown.output.weight.shape == (3, 16)


def test_growth_moves_the_learned_start_row_to_the_new_last_index(model):
    grown = model.grown_to(Vocabulary.from_text("abc"))
    assert torch.equal(grown.embed.weight[3], model.embed.weight[2])


def test_growth_keeps_old_raw_scores_for_every_old_context(model):
    grown = model.grown_to(Vocabulary.from_text("abc"))
    for context in ([], [0], [1], [0, 1], [1, 0, 1, 0]):
        old_row = model.logits(context)[: model.vocabulary.size]
        new_row = grown.logits(context)[: model.vocabulary.size]
        assert torch.equal(old_row, new_row)


def test_training_continues_from_a_snapshot():
    first = train_mlp_lm("ab" * 200, steps=15, batch=16, seed=0, learning_rate=0.1)
    second = train_mlp_lm(
        "ab" * 200,
        steps=15,
        batch=16,
        seed=0,
        learning_rate=0.1,
        start=snapshot_of(first.model),
    )
    assert second.loss_curve[0] < first.loss_curve[0]
    assert second.loss_curve[0] == pytest.approx(first.loss_curve[-1], abs=0.01)
    assert second.loss_curve[-1] < second.loss_curve[0]


def test_training_continuation_grows_the_vocabulary():
    first = train_mlp_lm("ab" * 200, steps=30, batch=16, seed=0)
    second = train_mlp_lm(
        "ab" * 150 + "c" * 100,
        steps=30,
        batch=16,
        seed=0,
        start=snapshot_of(first.model),
    )
    assert second.model.vocabulary.chars == ("a", "b", "c")
    assert second.loss_curve[0] < math.log(3)


def test_continuing_from_an_untrained_snapshot_repeats_the_fresh_run():
    torch.manual_seed(0)
    fresh_model = MlpLM(
        Vocabulary.from_text("ab" * 200),
        window=2,
        dimension=8,
        hidden=16,
    )
    fresh = train_mlp_lm(
        "ab" * 200,
        steps=10,
        batch=16,
        seed=0,
        window=2,
        dimension=8,
        hidden=16,
        start=snapshot_of(fresh_model),
    )
    plain = train_mlp_lm(
        "ab" * 200,
        steps=10,
        batch=16,
        seed=0,
        window=2,
        dimension=8,
        hidden=16,
    )
    assert fresh.loss_curve == plain.loss_curve
