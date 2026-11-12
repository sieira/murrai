import os
import time

import pytest

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.composition.chat import chat_commands
from murrai.presentation.chat.lanes import ChatLog
from murrai.presentation.chat.status import TrainingStatus, spawn_training


def _wait_for(log, text, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if any(text in line for line in log.lines):
            return
        time.sleep(0.05)
    raise AssertionError(f"no {text!r} in {log.lines}")


def test_a_training_runs_in_its_own_process_and_reports_over_a_pipe(tmp_path):
    (tmp_path / "story.txt").write_text("aabbaabb" * 20)
    store = FileCheckpointStore(tmp_path / "checkpoints")
    command = next(
        c
        for c in chat_commands(tmp_path, seed=0, store=store, neural_steps=20)
        if c.name == "retrain neural bigram"
    )
    log = ChatLog()
    status = TrainingStatus(observer=log)
    delivered = []

    spawn_training(command, status, delivered.append)
    _wait_for(log, "finished")

    lanes = [line.split(" ", 1)[1] for line in log.lines]
    assert lanes[0] == "[train] retrain neural bigram started"
    assert "[train] retrain neural bigram 20/20" in lanes
    assert lanes[-1] == "[train] retrain neural bigram finished"
    assert store.names() == ["neural-bigram"]
    assert len(delivered) == 1
    assert delivered[0].kind == "neural-bigram"


def test_a_training_that_cannot_leave_the_process_is_reported_failed():
    from murrai.presentation.chat import Command

    log = ChatLog()
    status = TrainingStatus(observer=log)

    spawn_training(Command("local", "x", lambda on_event=None: None, True), status)
    _wait_for(log, "failed")

    assert "[train] local failed" in log.lines[-1]


def test_the_counted_bigram_trains_in_a_process_and_swaps_in(tmp_path):
    (tmp_path / "story.txt").write_text("aabbaabb" * 20)
    store = FileCheckpointStore(tmp_path / "checkpoints")
    command = next(
        c
        for c in chat_commands(tmp_path, seed=0, store=store)
        if c.name == "counted bigram"
    )
    log = ChatLog()
    status = TrainingStatus(observer=log)
    delivered = []

    spawn_training(command, status, delivered.append)
    _wait_for(log, "finished")

    assert len(delivered) == 1
    assert store.names() == ["counted-bigram"]
    session = delivered[0]
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0


class Noisy:
    name = "noisy"

    def build(self, on_event=None):
        os.write(2, b"a warning that would land on the screen\n")
        os.write(1, b"print noise\n")


def test_a_training_process_never_writes_over_the_screen(capfd):
    log = ChatLog()
    status = TrainingStatus(observer=log)

    spawn_training(Noisy(), status)
    _wait_for(log, "finished")

    captured = capfd.readouterr()
    assert captured.err == ""
    assert captured.out == ""
