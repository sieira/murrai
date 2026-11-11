import pytest

from murrai.adapters.events import ListenerBus
from murrai.application.ports import TrainingEvent
from murrai.application.runs import train_neural_bigram


def test_the_bus_delivers_events_to_subscribers():
    bus = ListenerBus()
    first, second = [], []
    bus.subscribe(first.append)
    bus.subscribe(second.append)

    bus.emit(TrainingEvent(step=1, total_steps=10, elapsed_ms=5))

    assert first == [TrainingEvent(step=1, total_steps=10, elapsed_ms=5)]
    assert second == first

    bus.unsubscribe(first.append)
    bus.emit(TrainingEvent(step=2, total_steps=10, elapsed_ms=9))

    assert first == [TrainingEvent(step=1, total_steps=10, elapsed_ms=5)]
    assert second[-1].step == 2


def test_subscribing_twice_delivers_once():
    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)
    bus.subscribe(seen.append)

    bus.emit(TrainingEvent(step=1, total_steps=1, elapsed_ms=0))

    assert len(seen) == 1


def test_unsubscribing_an_unknown_listener_changes_nothing():
    bus = ListenerBus()

    bus.unsubscribe(lambda event: None)


def test_the_trainer_emits_one_event_per_step():
    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    train_neural_bigram("ab" * 50, steps=20, batch=8, seed=0, bus=bus)

    assert len(seen) == 20
    last = seen[-1]
    assert last.step == 20
    assert last.total_steps == 20
    assert last.elapsed_ms >= 0
    assert [event.step for event in seen] == list(range(1, 21))


def test_training_is_identical_with_a_listening_bus():
    quiet = train_neural_bigram("ab" * 50, steps=30, batch=8, seed=7)
    loud_bus = ListenerBus()
    loud = train_neural_bigram(
        "ab" * 50, steps=30, batch=8, seed=7, bus=loud_bus
    )

    assert loud.loss_curve == quiet.loss_curve


def test_mlp_and_transformer_trainers_emit_too():
    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    from murrai.application.runs import train_mlp_lm, train_transformer_lm

    train_mlp_lm("ab" * 50, steps=5, batch=8, seed=0, bus=bus)
    assert [event.step for event in seen] == [1, 2, 3, 4, 5]

    seen.clear()
    train_transformer_lm("ab" * 50, steps=5, batch=8, seed=0, bus=bus)
    assert [event.step for event in seen] == [1, 2, 3, 4, 5]
