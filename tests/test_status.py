from murrai.application.ports import TrainingEvent
from murrai.presentation.chat.repl import plain_text, status_line
from murrai.presentation.chat.status import TrainingStatus, status_fragments


def test_the_status_bar_names_the_selected_model():
    assert plain_text(status_line(TrainingStatus(), "gpt")) == "gpt"


def test_the_selected_model_sits_beside_running_trainings():
    status = TrainingStatus()
    status.track("retrain x")(TrainingEvent(step=1, total_steps=2))

    text = plain_text(status_line(status, "gpt"))

    assert text.startswith("gpt │ retrain x")
    assert "1/2" in text


def test_running_trainings_show_without_a_selected_model():
    status = TrainingStatus()
    status.track("retrain x")(TrainingEvent(step=1, total_steps=2))

    assert plain_text(status_line(status)) == plain_text(status_fragments(status))


def test_no_model_and_no_runs_leaves_the_hint():
    assert plain_text(status_line(TrainingStatus())) == "ctrl-p commands"


def test_a_running_training_shows_name_and_counts():
    status = TrainingStatus()

    status.track("gpt")(TrainingEvent(step=1, total_steps=2000))

    text = plain_text(status_fragments(status))
    assert "gpt" in text
    assert "1/2000" in text


def test_a_finished_training_leaves_the_status_bar():
    status = TrainingStatus()
    listener = status.track("gpt")
    listener(TrainingEvent(step=1, total_steps=2000))

    status.finish("gpt")

    assert status_fragments(status) == []


def test_a_failed_training_says_so():
    status = TrainingStatus()
    status.track("gpt")(TrainingEvent(step=3, total_steps=2000))

    status.fail("gpt")

    assert "gpt failed" in plain_text(status_fragments(status))


def test_two_trainings_share_one_line():
    status = TrainingStatus()
    status.track("neural bigram")(TrainingEvent(step=1, total_steps=2))
    status.track("gpt")(TrainingEvent(step=1, total_steps=2))

    text = plain_text(status_fragments(status))

    assert "neural bigram" in text
    assert "gpt" in text
    assert "\n" not in text


def test_the_status_names_the_phase():
    status = TrainingStatus()
    status.track("retrain gpt")(
        TrainingEvent(step=3, total_steps=300, phase="tokenising")
    )

    text = plain_text(status_fragments(status))
    assert "tokenising" in text
    assert "3/300" in text


def test_the_clock_restarts_when_the_phase_changes():
    now = {"t": 0.0}
    status = TrainingStatus(clock=lambda: now["t"])
    listener = status.track("gpt")

    listener(TrainingEvent(step=1, total_steps=300, phase="tokenising"))
    now["t"] = 60.0
    listener(TrainingEvent(step=300, total_steps=300, phase="tokenising"))
    listener(TrainingEvent(step=1, total_steps=10, phase="training"))
    now["t"] = 62.0
    listener(TrainingEvent(step=2, total_steps=10, phase="training"))

    text = plain_text(status_fragments(status))
    assert "eta 4s" in text


def test_a_running_training_shows_the_time_left():
    now = {"t": 0.0}
    status = TrainingStatus(clock=lambda: now["t"])
    listener = status.track("gpt")

    listener(TrainingEvent(step=1, total_steps=100))
    now["t"] = 10.0
    listener(TrainingEvent(step=10, total_steps=100))

    text = plain_text(status_fragments(status))
    assert "eta 1:30" in text


def test_the_screen_wakes_at_most_ten_times_a_second():
    now = {"t": 0.0}
    status = TrainingStatus(clock=lambda: now["t"])
    wakes = []
    status.redraw = lambda: wakes.append(1)
    listener = status.track("gpt")

    for step in range(1, 11):
        listener(TrainingEvent(step=step, total_steps=10))
    assert len(wakes) == 1

    now["t"] = 0.2
    listener(TrainingEvent(step=11, total_steps=20))
    assert len(wakes) == 2


def test_finishing_wakes_even_in_the_same_instant():
    now = {"t": 0.0}
    status = TrainingStatus(clock=lambda: now["t"])
    wakes = []
    status.redraw = lambda: wakes.append(1)
    listener = status.track("gpt")
    listener(TrainingEvent(step=1, total_steps=10))
    assert len(wakes) == 1

    status.finish("gpt")
    assert len(wakes) == 2


def test_a_training_shows_in_the_bar_before_its_first_step():
    status = TrainingStatus()

    status.begin("gpt")

    assert plain_text(status_fragments(status)) == "gpt starting"


def test_a_run_that_is_starting_sits_beside_one_that_is_running():
    status = TrainingStatus()
    status.track("neural bigram")(TrainingEvent(step=1, total_steps=2))
    status.begin("gpt")

    text = plain_text(status_fragments(status))

    assert text.startswith("neural bigram ")
    assert text.endswith(" │ gpt starting")


def test_the_first_step_replaces_starting_with_the_bar():
    status = TrainingStatus()
    status.begin("gpt")

    status.track("gpt")(TrainingEvent(step=1, total_steps=4))

    text = plain_text(status_fragments(status))
    assert "starting" not in text
    assert "1/4" in text
