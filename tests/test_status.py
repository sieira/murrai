from murrai.application.ports import TrainingEvent
from murrai.presentation.chat.repl import plain_text
from murrai.presentation.chat.status import TrainingStatus, status_fragments


def test_a_running_training_shows_name_and_counts():
    status = TrainingStatus()

    status.track("char gpt")(TrainingEvent(step=1, total_steps=2000))

    text = plain_text(status_fragments(status))
    assert "char gpt" in text
    assert "1/2000" in text


def test_a_finished_training_leaves_the_status_bar():
    status = TrainingStatus()
    listener = status.track("char gpt")
    listener(TrainingEvent(step=1, total_steps=2000))

    status.finish("char gpt")

    assert status_fragments(status) == []


def test_a_failed_training_says_so():
    status = TrainingStatus()
    status.track("char gpt")(TrainingEvent(step=3, total_steps=2000))

    status.fail("char gpt")

    assert "char gpt failed" in plain_text(status_fragments(status))
