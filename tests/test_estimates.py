from dataclasses import FrozenInstanceError

import pytest

from murrai.application.checkpoints import (
    CheckpointRecord,
    GptShape,
    GradientRun,
    MlpShape,
    RunSettings,
    Shape,
)
from murrai.application.estimates import (
    GPT_COST,
    MLP_COST,
    NEURAL_COST,
    Calibrator,
    StepClock,
    estimate_seconds,
    format_seconds,
)


def _gpt_shape(depth=4, dimension=128, heads=1, max_time=128, merges=300):
    return GptShape(
        merges=merges,
        depth=depth,
        dimension=dimension,
        heads=heads,
        max_time=max_time,
    )


def _run(
    model="gpt",
    corpus="data/corpus",
    seconds=None,
    steps=4000,
    batch=128,
    perplexity=11.7,
    shape=None,
):
    if shape is None:
        shape = {
            "gpt": _gpt_shape(),
            "windowed-mlp": MlpShape(
                merges=300,
                window=8,
                dimension=128,
                hidden=256,
            ),
            "neural-bigram": Shape(merges=300),
        }[model]
    return CheckpointRecord(
        settings=RunSettings(
            saved_at="x",
            corpus=corpus,
            seed=0,
            model=model,
        ),
        gradient=GradientRun(
            steps=steps,
            batch=batch,
            learning_rate=0.001,
            seconds=seconds,
            shape=shape,
        ),
        held_out_perplexity=perplexity,
    )


class _Event:
    def __init__(self, phase="training"):
        self.phase = phase


def test_the_recorded_shape_is_frozen():
    shape = _gpt_shape()

    with pytest.raises(FrozenInstanceError):
        shape.dimension = 64


def test_seconds_read_in_plain_units():
    assert format_seconds(90) == "1 min 30 s"
    assert format_seconds(2400) == "40 min"
    assert format_seconds(5400) == "1 h 30 min"


def test_time_scales_from_a_past_run_by_the_steps():
    past = _run(seconds=600.0, steps=4000, shape=_gpt_shape())
    seconds = estimate_seconds([past], _gpt_shape(), 128, 2000, GPT_COST)

    assert seconds is not None
    assert format_seconds(seconds) == "5 min"


def test_time_scales_by_the_compute_of_the_shape():
    past = _run(seconds=600.0, steps=4000, shape=_gpt_shape())
    seconds = estimate_seconds(
        [past],
        _gpt_shape(dimension=64),
        128,
        4000,
        GPT_COST,
    )

    assert seconds is not None
    # dimension halves, so a step costs a quarter
    assert format_seconds(seconds) == "2 min 30 s"


def test_time_uses_the_closest_past_run():
    far = _run(seconds=600.0, steps=4000, shape=_gpt_shape(dimension=32))
    close = _run(seconds=600.0, steps=4000, shape=_gpt_shape(dimension=64))
    seconds = estimate_seconds([far, close], _gpt_shape(), 128, 4000, GPT_COST)

    assert seconds is not None
    assert format_seconds(seconds) == "40 min"


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

    assert estimate_seconds([past], _gpt_shape(), 128, 4000, GPT_COST) is None


def test_time_ignores_runs_of_another_kind():
    past = _run(
        model="windowed-mlp",
        shape=MlpShape(
            merges=300,
            window=8,
            dimension=64,
            hidden=256,
        ),
    )

    assert estimate_seconds([past], _gpt_shape(), 128, 4000, GPT_COST) is None


def test_the_mlp_costs_its_own_shape():
    past = _run(
        model="windowed-mlp",
        seconds=600.0,
        steps=10000,
        shape=MlpShape(merges=300, window=8, dimension=128, hidden=256),
    )
    seconds = estimate_seconds(
        [past],
        MlpShape(merges=300, window=8, dimension=64, hidden=256),
        128,
        10000,
        MLP_COST,
    )

    assert seconds is not None
    assert format_seconds(seconds) == "5 min"


def test_the_neural_table_costs_its_piece_count():
    past = _run(
        model="neural-bigram",
        seconds=600.0,
        steps=2000,
        batch=64,
        shape=Shape(merges=300),
    )
    seconds = estimate_seconds([past], Shape(merges=300), 64, 2000, NEURAL_COST)

    assert seconds is not None
    assert format_seconds(seconds) == "10 min"


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


def test_an_untimed_chip_projects_nothing_and_is_not_pending():
    calibrator = Calibrator(lambda: None, GPT_COST)

    assert calibrator.project(_gpt_shape(), 128, 4000) is None
    assert not calibrator.pending


def test_a_finished_calibration_projects_the_requested_shape():
    calibrator = Calibrator(
        lambda: (
            0.1,
            _gpt_shape(dimension=64),
            128,
        ),
        GPT_COST,
    )
    calibrator.begin()
    calibrator.refresh()

    seconds = calibrator.project(_gpt_shape(), 128, 4000)

    # past: 0.1 s per step at dimension 64;
    # now: dimension 128 → 4x the cost per step
    assert "26 min" in format_seconds(seconds)


def test_a_calibration_in_progress_is_pending_until_it_lands():
    calibrator = Calibrator(lambda: (0.1, _gpt_shape(), 128), GPT_COST)

    calibrator.begin()
    assert calibrator.pending
    assert calibrator.project(_gpt_shape(), 128, 4000) is None

    calibrator.refresh()
    assert not calibrator.pending
    assert calibrator.project(_gpt_shape(), 128, 4000) is not None


def test_a_failed_calibration_is_not_pending_and_projects_nothing():
    calibrator = Calibrator(
        lambda: (_ for _ in ()).throw(RuntimeError("corpus is empty")),
        GPT_COST,
    )

    calibrator.begin()
    calibrator.refresh()

    assert not calibrator.pending
    assert calibrator.project(_gpt_shape(), 128, 4000) is None


def test_a_calibration_without_a_usable_step_is_not_pending():
    calibrator = Calibrator(lambda: (None, _gpt_shape(), 128), GPT_COST)

    calibrator.begin()
    calibrator.refresh()

    assert not calibrator.pending
    assert calibrator.project(_gpt_shape(), 128, 4000) is None


def test_a_finished_calibration_does_not_measure_again():
    measures = []

    def measure():
        measures.append(1)
        return (0.1, _gpt_shape(), 128)

    calibrator = Calibrator(measure, GPT_COST)
    calibrator.begin()
    calibrator.refresh()
    calibrator.refresh()

    assert len(measures) == 1
