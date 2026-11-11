from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import CompleteStyle, PromptSession


@dataclass
class Stage:
    name: str
    description: str
    build: Callable
    trains: bool = False


def filter_stages(stages, query):
    query = query.strip().lower()
    if not query:
        return list(stages)
    return [
        stage
        for stage in stages
        if query in stage.name.lower() or query in stage.description.lower()
    ]


def resolve_stage(stages, query):
    named = [stage for stage in stages if stage.name == query]
    if named:
        return named[0]
    matches = filter_stages(stages, query)
    if len(matches) == 1:
        return matches[0]
    return None


class _StageCompleter(Completer):
    def __init__(self, stages):
        self._stages = stages

    def get_completions(self, document, complete_event):
        for stage in filter_stages(self._stages, document.text):
            yield Completion(stage.name, start_position=-len(document.text))


def select_stage(stages, input=None, output=None):
    session_kwargs = {}
    if input is not None:
        session_kwargs["input"] = input
    if output is not None:
        session_kwargs["output"] = output
    session = PromptSession(
        completer=_StageCompleter(stages),
        complete_while_typing=True,
        complete_style=CompleteStyle.COLUMN,
        **session_kwargs,
    )
    while True:
        try:
            query = session.prompt("stage> ")
        except KeyboardInterrupt, EOFError:
            return None
        query = query.strip()
        if not query:
            continue
        stage = resolve_stage(stages, query)
        if stage is not None:
            return stage
        matches = filter_stages(stages, query)
        if matches:
            print(f"'{query}' is ambiguous: {', '.join(s.name for s in matches)}")
        else:
            print(f"no stage matches '{query}'")
