from murrai.adapters.events import ListenerBus
from murrai.application.ports import TrainingEvent
from murrai.application.runs import train_neural_bigram


def test_the_bus_delivers_events_to_subscribers():
    bus = ListenerBus()
    first, second = [], []
    bus.subscribe(first.append)
    bus.subscribe(second.append)

    bus.emit(TrainingEvent(step=1, total_steps=10))
    bus.flush()

    assert first == [TrainingEvent(step=1, total_steps=10)]
    assert second == first

    bus.unsubscribe(first.append)
    bus.emit(TrainingEvent(step=2, total_steps=10))
    bus.flush()

    assert first == [TrainingEvent(step=1, total_steps=10)]
    assert second[-1].step == 2


def test_subscribing_twice_delivers_once():
    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)
    bus.subscribe(seen.append)

    bus.emit(TrainingEvent(step=1, total_steps=1))
    bus.flush()

    assert len(seen) == 1


def test_unsubscribing_an_unknown_listener_changes_nothing():
    bus = ListenerBus()

    bus.unsubscribe(lambda event: None)


def test_the_trainer_emits_one_event_per_step():
    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    train_neural_bigram("ab" * 50, steps=20, batch=8, seed=0, bus=bus)
    bus.flush()

    training = [event for event in seen if event.phase == "training"]
    assert len(training) == 20
    last = training[-1]
    assert last.step == 20
    assert last.total_steps == 20
    assert [event.step for event in training] == list(range(1, 21))


def test_training_is_identical_with_a_listening_bus():
    quiet = train_neural_bigram("ab" * 50, steps=30, batch=8, seed=7)
    loud_bus = ListenerBus()
    loud = train_neural_bigram(
        "ab" * 50,
        steps=30,
        batch=8,
        seed=7,
        bus=loud_bus,
    )

    assert loud.loss_curve == quiet.loss_curve


def test_mlp_and_transformer_trainers_emit_too():
    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    from murrai.application.runs import train_mlp_lm, train_transformer_lm

    train_mlp_lm("ab" * 50, steps=5, batch=8, seed=0, bus=bus)
    bus.flush()
    training = [event for event in seen if event.phase == "training"]
    assert [event.step for event in training] == [1, 2, 3, 4, 5]

    seen.clear()
    train_transformer_lm("ab" * 50, steps=5, batch=8, seed=0, bus=bus)
    bus.flush()
    training = [event for event in seen if event.phase == "training"]
    assert [event.step for event in training] == [1, 2, 3, 4, 5]


def test_emit_returns_before_the_listener_runs():
    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    bus.emit(TrainingEvent(step=1, total_steps=10))
    assert seen == []

    bus.flush()
    assert seen == [TrainingEvent(step=1, total_steps=10)]


def test_a_slow_listener_cannot_slow_the_emitter():
    import time

    bus = ListenerBus()

    def slow(event):
        time.sleep(0.05)

    bus.subscribe(slow)

    start = time.perf_counter()
    for step in range(20):
        bus.emit(TrainingEvent(step=step, total_steps=20))
    emit_seconds = time.perf_counter() - start

    assert emit_seconds < 0.3
    bus.flush()


def test_a_dying_listener_does_not_break_the_bus():
    bus = ListenerBus()
    seen = []

    def broken(event):
        raise RuntimeError("boom")

    bus.subscribe(broken)
    bus.subscribe(seen.append)

    bus.emit(TrainingEvent(step=1, total_steps=10))
    bus.flush()

    assert seen == [TrainingEvent(step=1, total_steps=10)]


def test_listener_errors_keeps_only_the_last_twenty():
    bus = ListenerBus()

    def broken(event):
        raise RuntimeError("boom")

    bus.subscribe(broken)
    for step in range(25):
        bus.emit(TrainingEvent(step=step, total_steps=25))
    bus.flush()

    assert len(bus.listener_errors) == 20
