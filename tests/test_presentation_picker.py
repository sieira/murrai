from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.presentation.chat import Stage, filter_stages, resolve_stage, select_stage


def stages():
    return [
        Stage("counted bigram", "odds tallied straight from the corpus", lambda: None),
        Stage("neural bigram", "table trained by gradient descent", lambda: None),
    ]


def test_empty_query_lists_every_stage():
    all_stages = stages()
    assert filter_stages(all_stages, "") == all_stages


def test_query_narrows_by_name_case_insensitively():
    all_stages = stages()
    assert filter_stages(all_stages, "NEUR") == [all_stages[1]]


def test_query_narrows_by_description():
    all_stages = stages()
    assert filter_stages(all_stages, "gradient") == [all_stages[1]]


def test_no_match_returns_empty():
    assert filter_stages(stages(), "zzz") == []


def test_resolve_stage_prefers_the_exact_name():
    all_stages = stages()
    assert resolve_stage(all_stages, "neural bigram") is all_stages[1]


def test_resolve_stage_accepts_a_unique_partial_match():
    all_stages = stages()
    assert resolve_stage(all_stages, "neu") is all_stages[1]


def test_resolve_stage_rejects_ambiguous_matches():
    ambiguous = stages() + [Stage("neural mlp", "later", lambda: None)]
    assert resolve_stage(ambiguous, "neural") is None


def test_resolve_stage_rejects_empty_queries():
    assert resolve_stage(stages(), "  ") is None


def test_select_stage_takes_the_typed_name():
    all_stages = stages()
    with create_pipe_input() as pipe:
        pipe.send_text("neural bigram\r")
        chosen = select_stage(all_stages, input=pipe, output=DummyOutput())
    assert chosen.name == "neural bigram"


def test_select_stage_eof_returns_none():
    with create_pipe_input() as pipe:
        pipe.close()
        chosen = select_stage(stages(), input=pipe, output=DummyOutput())
    assert chosen is None
