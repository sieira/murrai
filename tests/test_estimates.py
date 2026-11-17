from murrai.application.checkpoints import (
    CheckpointRecord,
    GradientRun,
    RunSettings,
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


def test_seconds_read_in_plain_units():
    assert format_seconds(90) == "1 min 30 s"
    assert format_seconds(2400) == "40 min"
    assert format_seconds(5400) == "1 h 30 min"


def test_time_scales_from_a_past_run_by_the_steps():
    past = _run(seconds=600.0, steps=4000, shape=_values())
    seconds = estimate_seconds([past], _values(steps=2000), "gpt", GPT_COST)

    assert seconds is not None
    assert format_seconds(seconds) == "5 min"


def test_time_scales_by_the_compute_of_the_shape():
    past = _run(seconds=600.0, steps=4000, shape=_values())
    seconds = estimate_seconds([past], _values(dimension=64), "gpt", GPT_COST)

    assert seconds is not None
    # dimension halves, so a step costs a quarter
    assert format_seconds(seconds) == "2 min 30 s"


def test_time_uses_the_closest_past_run():
    far = _run(seconds=600.0, steps=4000, shape=_values(dimension=32))
    close = _run(seconds=600.0, steps=4000, shape=_values(dimension=64))
    seconds = estimate_seconds([far, close], _values(), "gpt", GPT_COST)

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

    assert estimate_seconds([past], _values(), "gpt", GPT_COST) is None


def test_time_ignores_a_record_whose_shape_misses_the_cost_keys():
    past = _run(seconds=600.0, steps=4000, shape={"depth": 4})

    assert estimate_seconds([past], _values(), "gpt", GPT_COST) is None


def test_time_ignores_runs_of_another_kind():
    past = _run(model="windowed-mlp", shape=_values())

    assert estimate_seconds([past], _values(), "gpt", GPT_COST) is None


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
    seconds = estimate_seconds([past], values, "windowed-mlp", MLP_COST)

    assert seconds is not None
    assert format_seconds(seconds) == "5 min"


def test_the_neural_table_costs_its_piece_count():
    past = _run(
        model="neural-bigram",
        seconds=600.0,
        steps=2000,
        batch=64,
        shape={"merges": 300},
    )
    values = {"steps": 2000, "batch": 64, "merges": 300}
    seconds = estimate_seconds([past], values, "neural-bigram", NEURAL_COST)

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

    assert calibrator.project(_values()) is None
    assert not calibrator.pending


def test_a_finished_calibration_projects_the_requested_shape():
    calibrator = Calibrator(
        lambda: (0.1, {**_values(dimension=64), "batch": 128, "merges": 300}),
        GPT_COST,
    )
    calibrator.begin()
    calibrator.refresh()

    seconds = calibrator.project(_values())

    # past: 0.1 s per step at dimension 64;
    # now: dimension 128 → 4x the cost per step
    assert "26 min" in format_seconds(seconds)


def test_a_calibration_in_progress_is_pending_until_it_lands():
    calibrator = Calibrator(lambda: (0.1, _values()), GPT_COST)

    calibrator.begin()
    assert calibrator.pending
    assert calibrator.project(_values()) is None

    calibrator.refresh()
    assert not calibrator.pending
    assert calibrator.project(_values()) is not None


def test_a_failed_calibration_is_not_pending_and_projects_nothing():
    calibrator = Calibrator(
        lambda: (_ for _ in ()).throw(RuntimeError("corpus is empty")),
        GPT_COST,
    )

    calibrator.begin()
    calibrator.refresh()

    assert not calibrator.pending
    assert calibrator.project(_values()) is None


def test_a_calibration_without_a_usable_step_is_not_pending():
    calibrator = Calibrator(lambda: (None, _values()), GPT_COST)

    calibrator.begin()
    calibrator.refresh()

    assert not calibrator.pending
    assert calibrator.project(_values()) is None


def test_a_finished_calibration_does_not_measure_again():
    measures = []

    def measure():
        measures.append(1)
        return (0.1, _values())

    calibrator = Calibrator(measure, GPT_COST)
    calibrator.begin()
    calibrator.refresh()
    calibrator.refresh()

    assert len(measures) == 1
