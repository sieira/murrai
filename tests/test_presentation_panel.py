from murrai.presentation.chat.panel import (
    Command,
    CommandPanel,
    description_fragments,
    list_fragments,
)


def commands():
    return [
        Command(
            "counted bigram",
            "odds tallied straight from the corpus",
            lambda: None,
        ),
        Command(
            "neural bigram",
            "table trained by gradient descent",
            lambda: None,
        ),
    ]


def opened(every=None):
    panel = CommandPanel(every if every is not None else commands())
    panel.open()
    return panel


def test_opening_lists_every_command_with_the_first_selected():
    every = commands()
    panel = opened(every)

    assert panel.visible
    assert panel.menu == every
    assert panel.current is every[0]


def test_down_and_up_move_the_selection_and_wrap_around():
    every = commands()
    panel = opened(every)

    panel.move(1)
    assert panel.current is every[1]
    panel.move(1)
    assert panel.current is every[0]
    panel.move(-1)
    assert panel.current is every[1]


def test_an_empty_menu_has_no_current_command_and_ignores_moves():
    panel = opened([])

    panel.move(1)

    assert panel.current is None


def test_opening_again_rebuilds_the_menu_and_resets_the_selection():
    state = {"extra": False}

    def menu():
        extra = [Command("load x", "pick it up", lambda: None)]
        return [*commands(), *extra] if state["extra"] else commands()

    panel = CommandPanel(menu)
    panel.open()
    panel.move(1)
    state["extra"] = True
    panel.open()

    assert len(panel.menu) == 3
    assert panel.current is panel.menu[0]


def test_the_list_shows_names_and_marks_the_selected_one():
    panel = opened()

    fragments = list_fragments(panel)
    text = "".join(text for _, text in fragments)

    assert "counted bigram" in text
    assert "neural bigram" in text
    assert "odds tallied" not in text
    assert ("class:selected", "counted bigram") in fragments


def test_the_description_box_shows_the_selected_command_description():
    panel = opened()
    assert "".join(t for _, t in description_fragments(panel)) == (
        "odds tallied straight from the corpus"
    )

    panel.move(1)

    assert "".join(t for _, t in description_fragments(panel)) == (
        "table trained by gradient descent"
    )


def test_the_description_box_is_empty_without_a_command():
    assert description_fragments(opened([])) == []


def test_a_long_description_wraps_between_words():
    panel = opened(
        [Command("x", "odds tallied straight from the corpus", lambda: None)],
    )

    text = "".join(t for _, t in description_fragments(panel, width=16))

    assert text.split("\n") == ["odds tallied", "straight from", "the corpus"]
