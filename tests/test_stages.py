from murrai.application.ports import TrainingEvent


def test_a_training_event_defaults_to_the_training_stage():
    event = TrainingEvent(step=1, total_steps=10)
    assert event.phase == "training"
    assert event.phase != "tokenising"


def test_a_stage_that_is_not_a_training_stage_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        TrainingEvent(step=1, total_steps=10, phase="swimming")
