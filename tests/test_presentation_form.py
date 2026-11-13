from murrai.presentation.chat.form import Field, TrainingForm, label_fragments
from murrai.presentation.chat.lanes import ChatLog
from murrai.presentation.chat.panel import Command


def _fields():
    return (
        Field("steps", "steps", "int", 2000, check="whole above zero"),
        Field("batch", "batch", "int", 64, check="whole above zero"),
        Field(
            "start_from",
            "start from",
            "choice",
            "fresh",
            choices=("fresh", "char-gpt"),
        ),
        Field("depth", "depth", "int", 4, check="whole above zero", shape=True),
        Field("seed", "seed", "int", 0, check="whole number"),
    )


def _form(fields=None, remembered=None, observer=None):
    command = Command(
        "retrain x",
        "same memory",
        lambda: None,
        trains=True,
        fields=tuple(_fields() if fields is None else fields),
    )
    return TrainingForm(command, remembered=remembered, observer=observer)


def test_the_form_opens_on_the_baked_defaults():
    form = _form()
    form.open()

    assert form.visible
    assert form.text("steps") == "2000"
    assert form.text("batch") == "64"
    assert form.chosen("start_from") == "fresh"
    assert form.text("depth") == "4"
    assert form.text("seed") == "0"


def test_the_form_opens_on_the_first_field():
    assert _form().current.name == "steps"


def test_tab_walks_the_fields_in_order_and_wraps():
    form = _form()
    form.move(1)
    assert form.current.name == "batch"
    form.move(1)
    assert form.current.name == "start_from"
    form.move(1)
    assert form.current.name == "depth"
    form.move(1)
    assert form.current.name == "seed"
    form.move(1)
    assert form.current.name == "steps"


def test_shift_tab_walks_backwards():
    form = _form()
    form.move(-1)
    assert form.current.name == "seed"
    form.move(-1)
    assert form.current.name == "depth"


def test_picking_a_savegame_locks_the_shape_fields():
    form = _form()
    form.move_choice("start_from", 1)

    assert form.locked("depth")
    assert not form.locked("steps")
    assert not form.locked("start_from")
    assert not form.locked("seed")


def test_choosing_fresh_unlocks_the_shape_fields():
    form = _form()
    form.move_choice("start_from", 1)
    form.move_choice("start_from", -1)

    assert not form.locked("depth")


def test_tab_skips_the_locked_fields():
    form = _form()
    form.move(1)
    form.move(1)
    form.move_choice("start_from", 1)

    form.move(1)
    assert form.current.name == "seed"
    form.move(1)
    assert form.current.name == "steps"
    form.move(-1)
    assert form.current.name == "seed"


def test_the_chosen_savegame_is_what_the_form_reports():
    form = _form()
    form.move_choice("start_from", 1)

    assert form.chosen("start_from") == "char-gpt"


def test_accept_returns_the_chosen_values():
    form = _form()
    form.set_text("steps", "500")

    assert form.accept() == {
        "steps": 500,
        "batch": 64,
        "start_from": "fresh",
        "depth": 4,
        "seed": 0,
    }


def test_accept_turns_a_choice_into_its_value():
    form = _form()
    form.move_choice("start_from", 1)

    assert form.accept()["start_from"] == "char-gpt"


def test_accept_names_the_field_and_the_problem():
    form = _form()
    form.set_text("steps", "ten")

    assert form.accept() is None
    assert form.message == "steps must be a whole number above zero"
    assert form.current.name == "steps"


def test_accept_rejects_a_whole_number_below_one():
    form = _form()
    form.set_text("batch", "0")

    assert form.accept() is None
    assert form.message == "batch must be a whole number above zero"


def test_accept_rejects_an_empty_field():
    form = _form()
    form.set_text("steps", "")

    assert form.accept() is None
    assert form.message == "steps must be a whole number above zero"


def test_accept_allows_a_whole_number_zero_or_more():
    patience = Field("patience", "patience", "int", 1000, check="whole zero or more")
    form = _form((patience,))
    form.set_text("patience", "0")

    assert form.accept() == {"patience": 0}


def test_accept_allows_a_negative_seed():
    seed = Field("seed", "seed", "int", 0, check="whole number")
    form = _form((seed,))
    form.set_text("seed", "-3")

    assert form.accept() == {"seed": -3}


def test_accept_rejects_a_holdout_of_one():
    holdout = Field(
        "holdout",
        "holdout",
        "float",
        0.1,
        check="from zero below one",
    )
    form = _form((holdout,))
    form.set_text("holdout", "1")

    assert form.accept() is None
    assert form.message == (
        "holdout must be a number from zero up to, but not including, one"
    )


def test_accept_allows_a_holdout_of_zero():
    holdout = Field("holdout", "holdout", "float", 0.1, check="from zero below one")
    form = _form((holdout,))
    form.set_text("holdout", "0")

    assert form.accept() == {"holdout": 0.0}


def test_accept_rejects_a_shrink_of_zero():
    shrink = Field("shrink", "shrink", "float", 0.5, check="between zero and one")
    form = _form((shrink,))
    form.set_text("shrink", "0")

    assert form.accept() is None
    assert form.message == "shrink must be a number between zero and one"


def test_accept_rejects_a_learning_rate_of_zero():
    rate = Field(
        "learning_rate",
        "learning rate",
        "float",
        3.0,
        check="number above zero",
    )
    form = _form((rate,))
    form.set_text("learning_rate", "0")

    assert form.accept() is None
    assert form.message == "learning rate must be a number above zero"


def test_a_locked_field_is_left_out_of_the_values_and_not_checked():
    form = _form()
    form.move_choice("start_from", 1)
    form.set_text("depth", "not a number")

    values = form.accept()

    assert "depth" not in values
    assert form.message is None


def test_the_message_clears_when_the_form_accepts():
    form = _form()
    form.set_text("steps", "ten")
    form.accept()
    form.set_text("steps", "10")

    assert form.accept() is not None
    assert form.message is None


def test_the_form_prefills_the_last_accepted_values():
    form = _form()
    form.set_text("steps", "500")
    form.move_choice("start_from", 1)
    values = form.accept()

    again = _form(remembered=values)

    assert again.text("steps") == "500"
    assert again.chosen("start_from") == "char-gpt"
    assert again.text("batch") == "64"


def test_a_remembered_choice_that_no_longer_exists_falls_back_to_the_default():
    again = _form(remembered={"start_from": "gone"})

    assert again.chosen("start_from") == "fresh"


def test_opening_and_cancelling_publish_the_form_lanes():
    log = ChatLog(clock=lambda: 0.0)
    form = _form(observer=log)

    form.open()
    form.cancel()

    assert log.lines == [
        "0.000 [form] opened retrain x",
        "0.000 [form] cancelled retrain x",
    ]


def test_cancelling_closes_the_form():
    form = _form()
    form.open()
    form.cancel()

    assert not form.visible


def test_the_label_marks_the_focused_field():
    form = _form()

    assert label_fragments(form, form.current) == [("class:selected", " » steps:")]


def test_an_unfocused_label_carries_no_style():
    form = _form()

    assert label_fragments(form, form.fields[1]) == [("", "   batch:")]


def test_a_locked_label_carries_the_savegame_marker():
    form = _form()
    form.move_choice("start_from", 1)

    assert label_fragments(form, form.fields[3]) == [
        ("class:hint", "   depth: (from savegame)"),
    ]
