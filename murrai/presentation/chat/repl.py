from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import PromptSession, print_formatted_text
from prompt_toolkit.styles import Style

GHOST_CHARS = 20
TOP = 5

STYLE = Style.from_dict(
    {
        "entered": "",
        "generated": "ansibrightcyan bold",
        "auto-suggestion": "ansibrightblack",
        "die-row": "ansibrightblack",
        "die-hit": "ansibrightcyan bold",
        "more": "ansibrightblack",
        "bottom-toolbar": "noreverse",
    },
)


class GhostChat:
    def __init__(self, session, ghost_chars=GHOST_CHARS):
        self.session = session
        self.ghost_chars = ghost_chars
        self._synced = ""

    def sync(self, buffer_text: str) -> None:
        common = 0
        for a, b in zip(self._synced, buffer_text, strict=False):
            if a != b:
                break
            common += 1
        self.session.truncate(common)
        self.session.extend_user(buffer_text[common:])
        self._synced = buffer_text

    def accept(self, buffer_text: str, char: str) -> None:
        self.sync(buffer_text)
        self.session.extend_generated(char)
        self._synced = self.session.text

    def accept_rest(self, buffer_text: str, ghost: str) -> None:
        self.sync(buffer_text)
        self.session.extend_generated(ghost)
        self._synced = self.session.text

    def suggestions(self, buffer_text: str) -> str:
        try:
            return self.session.rollout(buffer_text, self.ghost_chars)
        except ValueError:
            return ""

    def die_bar(self, buffer_text: str, ghost: str = ""):
        try:
            dice = self.session.preview(buffer_text)
        except ValueError:
            return []
        fragments = []
        hit = ghost[0] if ghost else None
        for char, p in dice[:TOP]:
            mark = "»" if char == hit else " "
            style = "class:die-hit" if char == hit else "class:die-row"
            fragments.append((style, f" {mark} {char} {p:6.1%}"))
        if len(dice) > TOP:
            fragments.append(("class:more", f"   ... {len(dice) - TOP} more"))
        return fragments


class GhostSuggestions(AutoSuggest):
    def __init__(self, chat: GhostChat):
        self._chat = chat
        self._cache: dict[str, str] = {}

    def _roll_for(self, context: str) -> str:
        if context not in self._cache:
            roll = self._chat.suggestions(context)
            if not roll:
                self._cache[context] = ""
                return ""
            for offset in range(len(roll) + 1):
                self._cache[context + roll[:offset]] = roll[offset:]
        return self._cache[context]

    def real_for(self, context: str) -> str:
        return self._roll_for(context)

    def get_suggestion(self, buffer, document):
        roll = self._roll_for(document.text)
        if not roll:
            return None
        return Suggestion(roll.replace("\n", "↵"))


def _key_bindings(chat: GhostChat, suggestions: GhostSuggestions) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("tab")
    def _(event):
        buffer = event.current_buffer
        if buffer.suggestion is None or not buffer.suggestion.text:
            return
        real = suggestions.real_for(buffer.text)
        if not real:
            return
        try:
            chat.accept(buffer.text, real[0])
            buffer.text = buffer.text + real[0]
        except ValueError:
            return

    @kb.add("enter")
    def _(event):
        buffer = event.current_buffer
        if buffer.suggestion is not None and buffer.suggestion.text:
            real = suggestions.real_for(buffer.text)
            try:
                chat.accept_rest(buffer.text, real)
                buffer.text = buffer.text + real
            except ValueError:
                pass
        buffer.validate_and_handle()

    return kb


def _intro():
    return [
        (
            "class:more",
            "murrAI: write; Tab accepts a char; Enter lets it finish; ctrl-c quits\n",
        ),
    ]


def run_chat(session, input=None, output=None):
    chat = GhostChat(session)
    suggestions = GhostSuggestions(chat)
    holder = {}

    def _toolbar():
        current = holder.get("prompt")
        if current is None:
            return []
        buffer = current.default_buffer
        return chat.die_bar(buffer.text, suggestions.real_for(buffer.text))

    prompt = PromptSession(
        auto_suggest=suggestions,
        key_bindings=_key_bindings(chat, suggestions),
        style=STYLE,
        bottom_toolbar=_toolbar,
        multiline=True,
        complete_while_typing=False,
        input=input,
        output=output,
    )
    holder["prompt"] = prompt
    writer = prompt_writer(output)
    writer(_intro())
    while True:
        try:
            text = prompt.prompt(default=chat.session.text)
        except EOFError, KeyboardInterrupt:
            writer([("class:more", "bye\n")])
            return
        chat.sync(text)


def prompt_writer(output=None):
    def write(fragments):
        print_formatted_text(FormattedText(fragments), style=STYLE, output=output)

    return write


def plain_text(fragments):
    return "".join(text for _, text in fragments)
