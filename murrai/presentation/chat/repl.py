from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.shortcuts import PromptSession, print_formatted_text
from prompt_toolkit.styles import Style

BAR_WIDTH = 12
TOP = 8

STYLE = Style.from_dict(
    {
        "entered": "",
        "generated": "ansibrightcyan bold",
        "die-row": "ansibrightblack",
        "die-value": "ansicyan",
        "die-hit": "ansigreen bold",
        "rolled": "ansigreen bold",
        "more": "ansibrightblack",
    }
)


def die_fragments(dice, rolled=None, top=TOP):
    fragments = []
    for char, p in dice[:top]:
        hit = char == rolled
        marker = "»" if hit else " "
        style = "class:die-hit" if hit else "class:die-row"
        bar = "█" * max(1, round(p * BAR_WIDTH))
        fragments.append((style, f"{marker} {char} {p:6.1%} {bar}\n"))
    if len(dice) > top:
        fragments.append(("class:more", f"  ... {len(dice) - top} more\n"))
    if rolled is not None:
        fragments.append(("class:rolled", f"→ {rolled}\n"))
    return fragments


def plain_text(fragments):
    return "".join(text for _, text in fragments)


def _intro():
    return [
        (
            "class:more",
            "chat: type your part of the story; Enter hands the turn; ctrl-c quits\n",
        )
    ]


def run_chat(session, read, write):
    write(_intro())
    while True:
        try:
            buffer = read(session.text)
        except KeyboardInterrupt, EOFError:
            write([("class:more", "bye\n")])
            return
        if buffer is None:
            write([("class:more", "bye\n")])
            return
        try:
            session.commit(buffer)
            die = session.dice()
            rolled = session.send()
        except ValueError:
            write([("class:more", "unknown char for this corpus\n")])
            continue
        write(die_fragments(die, rolled))


def prompt_writer(output=None):
    def write(fragments):
        print_formatted_text(FormattedText(fragments), style=STYLE, output=output)

    return write


class _StoryLexer(Lexer):
    def __init__(self, chat):
        self._chat = chat

    def lex_document(self, document):
        text = document.text
        return lambda _: _story_fragments(self._chat, text)


def _story_fragments(chat, text):
    fragments = []
    position = 0
    for segment, origin in chat.segments():
        if position >= len(text):
            break
        piece = segment[: len(text) - position]
        style = "class:generated" if origin == "generated" else "class:entered"
        fragments.append((style, piece))
        position += len(piece)
    if position < len(text):
        fragments.append(("class:entered", text[position:]))
    return fragments


class PromptReader:
    def __init__(self, chat, input=None, output=None):
        self._chat = chat
        self._prompt = PromptSession(
            lexer=_StoryLexer(chat),
            style=STYLE,
            bottom_toolbar=lambda: self._preview(),
            complete_while_typing=False,
            input=input,
            output=output,
        )

    def read(self, story_text):
        try:
            return self._prompt.prompt(default=story_text)
        except EOFError:
            return None

    def _preview(self):
        try:
            text = self._prompt.default_buffer.text
            return FormattedText(die_fragments(self._chat.preview(text)))
        except Exception:
            return ""


def interactive_chat(session, input=None, output=None):
    run_chat(session, PromptReader(session, input, output).read, prompt_writer(output))
