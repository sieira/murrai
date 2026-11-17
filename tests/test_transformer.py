import math

import pytest
import torch

import murrai.application.runs as runs
from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.application.checkpoints import snapshot_of
from murrai.application.runs import _draw_batch, train_transformer_lm
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
    ids = [model.start, *model.tokenizer.encode(text)]
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


def test_training_continues_from_a_snapshot():
    first = train_transformer_lm(
        "ab" * 200,
        steps=40,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
    )
    second = train_transformer_lm(
        "ab" * 200,
        steps=40,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        checkpoint=snapshot_of(first.model),
    )
    assert second.loss_curve[0] < first.loss_curve[0]
    assert second.loss_curve[0] == pytest.approx(first.loss_curve[-1], abs=0.05)
    assert second.loss_curve[-1] < second.loss_curve[0]


def test_continuing_from_an_untrained_snapshot_repeats_the_fresh_run():
    torch.manual_seed(0)
    tokenizer = TokenizerEncoderDecoder().fit("ab" * 200, 300)
    fresh_model = TransformerLM(
        tokenizer,
        depth=1,
        dimension=8,
        max_time=8,
    )
    fresh = train_transformer_lm(
        "ab" * 200,
        steps=10,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
        checkpoint=snapshot_of(fresh_model),
    )
    plain = train_transformer_lm(
        "ab" * 200,
        steps=10,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
    )
    assert fresh.loss_curve == plain.loss_curve


def test_continuing_rejects_a_contradictory_shape():
    first = train_transformer_lm(
        "ab" * 200,
        steps=5,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
    )
    with pytest.raises(ValueError):
        train_transformer_lm(
            "ab" * 200,
            steps=5,
            batch=8,
            seed=0,
            depth=2,
            dimension=8,
            max_time=8,
            checkpoint=snapshot_of(first.model),
        )


def test_batches_are_drawn_from_the_encoded_stream():
    tokenizer = TokenizerEncoderDecoder()
    stream = torch.tensor([tokenizer.vocab_size, *tokenizer.encode("aabbaabb" * 3)])
    generator = torch.Generator().manual_seed(0)

    rows, targets = _draw_batch(stream, window=4, batch=3, generator=generator)

    starts = torch.randint(
        0,
        len(stream) - 4,
        (3,),
        generator=torch.Generator().manual_seed(0),
    )
    assert rows.shape == (3, 4)
    assert targets.shape == (3, 4)
    for index, start in enumerate(starts.tolist()):
        assert rows[index].tolist() == stream[start : start + 4].tolist()
        assert targets[index].tolist() == stream[start + 1 : start + 5].tolist()


def test_the_run_validates_on_a_separate_held_out_stream():
    result = train_transformer_lm(
        "ab" * 400,
        steps=10,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
        held_out_text="ab" * 80,
        validate_every=2,
    )
    assert [step for step, _ in result.validation_curve] == [0, 2, 4, 6, 8, 10]
    assert all(math.isfinite(loss) for _, loss in result.validation_curve)
    assert result.held_out_perplexity == pytest.approx(
        math.exp(result.validation_curve[-1][1]),
    )


def test_the_validation_stream_is_never_trained_on():
    common = {
        "steps": 150,
        "batch": 8,
        "seed": 0,
        "depth": 1,
        "dimension": 8,
        "max_time": 8,
        "holdout": 0,
        "merges": 0,
    }
    familiar = train_transformer_lm("ab" * 400, held_out_text="ab" * 200, **common)
    foreign = train_transformer_lm("ab" * 400, held_out_text="cd" * 200, **common)

    assert familiar.validation_curve[-1][1] < familiar.validation_curve[0][1] - 3.0
    assert foreign.validation_curve[-1][1] > familiar.validation_curve[-1][1] + 3.0


def test_the_validation_curve_appears_for_a_holdout_fraction():
    result = train_transformer_lm(
        "ab" * 400,
        steps=6,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0.2,
        validate_every=2,
    )
    assert [step for step, _ in result.validation_curve] == [0, 2, 4, 6]
    assert result.held_out_perplexity == pytest.approx(
        math.exp(result.validation_curve[-1][1]),
    )


def test_a_run_without_a_held_out_stream_reports_no_validation():
    result = train_transformer_lm(
        "ab" * 400,
        steps=4,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
    )
    assert result.validation_curve == []
    assert result.held_out_perplexity is None


def test_the_run_reports_its_steps_and_token_throughput():
    result = train_transformer_lm(
        "ab" * 4000,
        steps=10,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
    )
    assert result.steps == 10
    assert result.tokens_per_second > 0


def test_a_run_resumes_from_a_periodic_snapshot_and_continues_its_curve(tmp_path):
    store = FileCheckpointStore(tmp_path / "checkpoints")
    kept = []
    common = {
        "steps": 8,
        "batch": 8,
        "seed": 0,
        "depth": 1,
        "dimension": 8,
        "max_time": 8,
        "holdout": 0,
        "held_out_text": "ab" * 80,
        "validate_every": 2,
    }

    def snapshots(snapshot):
        step = snapshot.state["run"]["step"]
        kept.append(step)
        store.save(f"midrun-{step}", snapshot)

    first = train_transformer_lm(
        "ab" * 400,
        checkpoint_every=3,
        snapshots=snapshots,
        **common,
    )
    assert kept == [3, 6, 8]

    later = {**common, "steps": 5}
    resumed = train_transformer_lm(
        "ab" * 400,
        checkpoint=store.load("midrun-3"),
        **later,
    )

    assert resumed.loss_curve == first.loss_curve[3:]
    assert resumed.rate_curve == first.rate_curve[3:]
    assert resumed.validation_curve == first.validation_curve[2:]
    assert resumed.steps == 8
    assert resumed.tokens_per_second > 0


def test_training_continues_a_finished_run_by_more_steps():
    kept = []
    first = train_transformer_lm(
        "ab" * 200,
        steps=8,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
        checkpoint_every=8,
        snapshots=kept.append,
    )
    later = train_transformer_lm(
        "ab" * 200,
        steps=3,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
        checkpoint=kept[-1],
    )
    assert later.steps == 11
    assert len(later.loss_curve) == 3
    assert later.loss_curve[-1] < first.loss_curve[-1]


def test_a_run_rejects_a_nonpositive_step_count():
    with pytest.raises(ValueError):
        train_transformer_lm(
            "ab" * 200,
            steps=0,
            batch=8,
            seed=0,
            depth=1,
            dimension=8,
            max_time=8,
        )


def test_the_run_snapshots_once_after_its_last_step():
    kept = []
    train_transformer_lm(
        "ab" * 200,
        steps=2,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
        checkpoint_every=5,
        snapshots=kept.append,
    )
    assert [snapshot.state["run"]["step"] for snapshot in kept] == [2]


def test_the_final_snapshot_does_not_duplicate_the_cadence():
    kept = []
    train_transformer_lm(
        "ab" * 200,
        steps=4,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
        checkpoint_every=2,
        snapshots=kept.append,
    )
    assert [snapshot.state["run"]["step"] for snapshot in kept] == [2, 4]


def test_the_run_snapshots_before_its_final_metrics(monkeypatch):
    seen = []

    def broken_metrics(model, stream):
        seen.append(stream)
        if len(seen) > 2:
            raise RuntimeError("metrics blew up")

    monkeypatch.setattr(runs, "_stream_nll", broken_metrics)
    kept = []
    with pytest.raises(RuntimeError):
        train_transformer_lm(
            "ab" * 200,
            steps=3,
            batch=8,
            seed=0,
            depth=1,
            dimension=8,
            max_time=8,
            holdout=0,
            held_out_text="ab" * 40,
            validate_every=2,
            checkpoint_every=5,
            snapshots=kept.append,
        )
    assert [snapshot.state["run"]["step"] for snapshot in kept] == [3]


def test_a_fresh_run_never_reencodes_its_train_text(monkeypatch):
    calls = []
    encode = TokenizerEncoderDecoder.encode

    def counting_encode(self, text, bus=None):
        calls.append(text)
        return encode(self, text, bus)

    monkeypatch.setattr(TokenizerEncoderDecoder, "encode", counting_encode)
    train_transformer_lm(
        "ab" * 200,
        steps=1,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
    )
    assert calls == []


def test_a_resumed_run_encodes_its_train_text_once(monkeypatch):
    calls = []
    encode = TokenizerEncoderDecoder.encode

    def counting_encode(self, text, bus=None):
        calls.append(text)
        return encode(self, text, bus)

    monkeypatch.setattr(TokenizerEncoderDecoder, "encode", counting_encode)
    first = train_transformer_lm(
        "ab" * 200,
        steps=2,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
    )
    calls.clear()
    train_transformer_lm(
        "ab" * 200,
        steps=1,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
        checkpoint=snapshot_of(first.model),
    )
    assert calls == ["ab" * 200]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="only a device copy abandons the shared buffer",
)
def test_the_token_array_is_released_once_its_stream_reaches_the_device(monkeypatch):
    import gc
    import weakref

    created = []
    real_array = runs.array

    def spy_array(*args, **kwargs):
        ids = real_array(*args, **kwargs)
        created.append(weakref.ref(ids))
        return ids

    monkeypatch.setattr(runs, "array", spy_array)
    draw_batch = runs._draw_batch

    def spy_draw_batch(*args, **kwargs):
        assert all(ref() is None for ref in created), "the token array is still alive"
        return draw_batch(*args, **kwargs)

    monkeypatch.setattr(runs, "_draw_batch", spy_draw_batch)
    train_transformer_lm(
        "ab" * 200,
        steps=1,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0,
        device="mps",
    )
    gc.collect()
    assert all(ref() is None for ref in created)
