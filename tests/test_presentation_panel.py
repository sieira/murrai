from murrai.presentation.chat import Command, filter_commands


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


def test_empty_query_lists_every_command():
    every = commands()
    assert filter_commands(every, "") == every


def test_query_narrows_by_name_case_insensitively():
    every = commands()
    assert filter_commands(every, "NEUR") == [every[1]]


def test_query_narrows_by_description():
    every = commands()
    assert filter_commands(every, "gradient") == [every[1]]


def test_no_match_returns_empty():
    assert filter_commands(commands(), "zzz") == []
