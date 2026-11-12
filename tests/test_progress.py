import time

from murrai.application.ports import TrainingEvent
from murrai.presentation.chat.progress import EtaLine


def test_the_eta_line_renders_the_newest_event():
    writes = []
    line = EtaLine(write=writes.append)

    line(TrainingEvent(step=10, total_steps=20, elapsed_ms=1000))
    line.drain()
    line(TrainingEvent(step=11, total_steps=20, elapsed_ms=1100))
    line.drain()

    assert writes == ["step 10/20 (50%)  ETA 1s", "step 11/20 (55%)  ETA 1s"]


def test_the_eta_line_says_nothing_without_an_event():
    writes = []
    line = EtaLine(write=writes.append)

    line.drain()

    assert writes == []


def test_the_eta_line_keeps_the_last_write_when_nothing_new_arrives():
    writes = []
    line = EtaLine(write=writes.append)

    line(TrainingEvent(step=10, total_steps=20, elapsed_ms=1000))
    line.drain()
    line.drain()

    assert writes == ["step 10/20 (50%)  ETA 1s"]


def test_the_reader_thread_renders_until_stopped():
    writes = []
    line = EtaLine(write=writes.append, interval=0.01)
    line.start()
    try:
        line(TrainingEvent(step=10, total_steps=20, elapsed_ms=1000))
        for _ in range(100):
            if writes:
                break
            time.sleep(0.01)
        assert writes, "the reader thread never rendered the event"
    finally:
        line.stop()

    line.stop()
