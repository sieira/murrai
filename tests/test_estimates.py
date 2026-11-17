import threading
import time

from murrai.application.checkpoints import (
    CheckpointRecord,
    GradientRun,
    RunSettings,
)
from murrai.application.estimates import (
    Calibrator,
    StepClock,
    estimate_perplexity,
    estimate_seconds,
    format_seconds,
    gpt_cost,
    mlp_cost,
    neural_cost,
    uniform_perplexity,
)


def _run(
    model="gpt",
    corpus="data/corpus",
    steps=4000,
    batch=128,
    seconds=600.0,
    shape=None,
    perplexity=11.7,
):
    return CheckpointRecord(
        settings=RunSettings(
            saved_at="2026-11-16T10:16:59+00:00",
            corpus=corpus,
            seed=0,
            model=model,
        ),
        gradient=GradientRun(
            steps=steps,
            batch=batch,
            learning_rate=0.001,
            seconds=seconds,
            shape={} if shape is None else shape,
        ),
        held_out_perplexity=perplexity,
    )


def _values(**over):
    values = {
        "steps": 4000,
        "batch": 128,
        "merges": 300,
        "depth": 6,
        "dimension": 128,
        "heads": 1,
        "max_time": 128,
    }
    values.update(over)
    return values


class _Event:
    def __init__(self, phase="training"):
        self.phase = phase


def test_a_uniform_guess_lands_at_the_piece_count():
    assert uniform_perplexity(300) == 556


def test_seconds_read_in_plain_units():
    assert format_seconds(90) == "1 min 30 s"
    assert format_seconds(2400) == "40 min"
    assert format_seconds(5400) == "1 h 30 min"


def test_time_scales_from_a_past_run_by_the_steps():
    past = _run(seconds=600.0, steps=4000, shape=_values())
    estimate = estimate_seconds([past], _values(steps=2000), "gpt", gpt_cost)

    assert estimate is not None
    assert "about 5 min" in estimate.text


def test_time_scales_by_the_compute_of_the_shape():
    past = _run(seconds=600.0, steps=4000, shape=_values())
    estimate = estimate_seconds([past], _values(dimension=64), "gpt", gpt_cost)

    assert estimate is not None
    # dimension halves, so a step costs a quarter
    assert "about 2 min 30 s" in estimate.text


def test_time_uses_the_closest_past_run():
    far = _run(seconds=600.0, steps=4000, shape=_values(dimension=32))
    close = _run(seconds=600.0, steps=4000, shape=_values(dimension=64))
    estimate = estimate_seconds([far, close], _values(), "gpt", gpt_cost)

    assert estimate is not None
    assert estimate.runs == 2
    assert "about 40 min" in estimate.text


def test_time_ignores_runs_that_carry_no_measurement():
    past = CheckpointRecord(
        settings=RunSettings(
            saved_at="x",
            corpus="data/corpus",
            seed=0,
            model="gpt",
        ),
        gradient=GradientRun(steps=4000, batch=128, learning_rate=0.001),
    )

    assert estimate_seconds([past], _values(), "gpt", gpt_cost) is None


def test_time_ignores_runs_of_another_kind():
    past = _run(model="windowed-mlp", shape=_values())

    assert estimate_seconds([past], _values(), "gpt", gpt_cost) is None


def test_the_mlp_costs_its_own_shape():
    past = _run(
        model="windowed-mlp",
        seconds=600.0,
        steps=10000,
        shape={"window": 8, "dimension": 128, "hidden": 256, "merges": 300},
    )
    values = {
        "steps": 10000,
        "batch": 128,
        "merges": 300,
        "window": 8,
        "dimension": 64,
        "hidden": 256,
    }
    estimate = estimate_seconds([past], values, "windowed-mlp", mlp_cost)

    assert estimate is not None
    assert "about 5 min" in estimate.text


def test_the_neural_table_costs_its_piece_count():
    past = _run(
        model="neural-bigram",
        seconds=600.0,
        steps=2000,
        batch=64,
        shape={"merges": 300},
    )
    values = {"steps": 2000, "batch": 64, "merges": 300}
    estimate = estimate_seconds([past], values, "neural-bigram", neural_cost)

    assert estimate is not None
    assert "about 10 min" in estimate.text


def test_ppl_of_one_past_run_names_it():
    past = _run()

    estimate = estimate_perplexity([past], "gpt", past.settings.corpus)

    assert estimate is not None
    assert estimate.text == "the last run like this landed at ppl 11.7"


def test_ppl_of_several_past_runs_spans_a_band():
    runs = [_run(perplexity=8.0), _run(perplexity=14.0), _run(perplexity=11.0)]

    estimate = estimate_perplexity(runs, "gpt", runs[0].settings.corpus)

    assert estimate is not None
    assert estimate.text == (
        "runs like this landed between ppl 8.0 and 14.0 (from 3 runs)"
    )


def test_ppl_ignores_other_kinds_and_other_corpora():
    elsewhere = _run(corpus="data/other")
    other_kind = _run(model="windowed-mlp")

    assert estimate_perplexity([elsewhere], "gpt", "data/corpus") is None
    assert estimate_perplexity([other_kind], "gpt", "data/corpus") is None


def test_ppl_ignores_runs_without_a_held_out_score():
    past = _run(perplexity=None)

    assert estimate_perplexity([past], "gpt", "data/corpus") is None


def test_ppl_ignores_records_older_than_the_record_schema():
    past = CheckpointRecord(
        settings=RunSettings(saved_at="x", corpus="data/corpus", seed=0),
        gradient=GradientRun(steps=1, batch=1, learning_rate=0.001),
        held_out_perplexity=11.7,
    )

    assert estimate_perplexity([past], "gpt", "data/corpus") is None


def test_the_clock_times_only_the_training_steps():
    stamps = iter([1.0, 2.0, 3.0, 4.0, 5.0])
    clock = StepClock(now=lambda: next(stamps))

    for phase in (
        "tokenising",
        "tokenising",
        "encoding",
        "training",
        "training",
        "training",
        "training",
        "training",
    ):
        clock.emit(_Event(phase))

    # the non-training phases never stamp; five training stamps 1..5
    # → deltas 1 s; the two warmup steps drop out
    assert clock.per_step() == 1.0


def test_the_clock_drops_the_first_two_steps_as_warmup():
    stamps = iter([0.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    clock = StepClock(now=lambda: next(stamps))

    for _ in range(8):
        clock.emit(_Event())

    # deltas 10,1,1,1,1,1,1 → the warmup drop removes the 10 s step
    assert clock.per_step() == 1.0


def test_an_untimed_chip_projects_nothing():
    calibrator = Calibrator(lambda: None, gpt_cost)

    assert calibrator.project(_values()) is None
    assert not calibrator.pending


def test_a_finished_calibration_projects_the_requested_shape():
    calibrator = Calibrator(
        lambda: (0.1, {**_values(dimension=64), "batch": 128, "merges": 300}),
        gpt_cost,
    )
    calibrator.refresh()

    seconds = calibrator.project(_values())

    # past: 0.1 s per step at dimension 64;
    # now: dimension 128 → 4x the cost per step
    assert "26 min" in format_seconds(seconds)


def test_a_running_calibration_is_pending():
    gate = threading.Event()
    calibrator = Calibrator(lambda: (gate.wait(1), (0.1, _values()))[1], gpt_cost)

    calibrator.kick()
    assert calibrator.pending
    gate.set()
    for _ in range(200):
        if not calibrator.pending:
            break
        time.sleep(0.01)

    assert not calibrator.pending
    assert calibrator.project(_values()) is not None
