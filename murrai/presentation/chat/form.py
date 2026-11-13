from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    Float,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Dialog, TextArea

from murrai.presentation.chat.lanes import SILENT, FormCancelled, FormOpened

FRESH = "fresh"
START_FROM = "start_from"
FORM_WIDTH = 48
LABEL_WIDTH = 24
FORM_TOP = 2


def _whole(text):
    return int(text)


def _number(text):
    return float(text)


@dataclass(frozen=True)
class Rule:
    words: str
    parse: Callable
    allows: Callable


RULES = {
    "whole above zero": Rule(
        "a whole number above zero",
        _whole,
        lambda value: value > 0,
    ),
    "whole zero or more": Rule(
        "a whole number zero or more",
        _whole,
        lambda value: value >= 0,
    ),
    "whole number": Rule(
        "a whole number",
        _whole,
        lambda value: True,
    ),
    "from zero below one": Rule(
        "a number from zero up to, but not including, one",
        _number,
        lambda value: 0 <= value < 1,
    ),
    "between zero and one": Rule(
        "a number between zero and one",
        _number,
        lambda value: 0 < value < 1,
    ),
    "number above zero": Rule(
        "a number above zero",
        _number,
        lambda value: value > 0,
    ),
}


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    kind: str
    default: object
    check: str = ""
    choices: tuple = ()
    shape: bool = False


def _rejection(field):
    return f"{field.label} must be {RULES[field.check].words}"


class TrainingForm:
    def __init__(self, command, remembered=None, observer=SILENT):
        self.command = command
        self.fields = tuple(command.fields)
        self._observer = observer or SILENT
        self._remembered = remembered or {}
        self._texts = {}
        self._positions = {}
        for field in self.fields:
            if field.kind == "choice":
                wanted = self._remembered.get(field.name, field.default)
                if wanted not in field.choices:
                    wanted = field.default
                self._positions[field.name] = field.choices.index(wanted)
            else:
                self._texts[field.name] = str(
                    self._remembered.get(field.name, field.default),
                )
        self.visible = False
        self.message = None
        self.current = self._first_open_field()

    def _field(self, name):
        return next(field for field in self.fields if field.name == name)

    def _first_open_field(self):
        for field in self.fields:
            if not self.locked(field.name):
                return field
        return self.fields[0]

    def _continuing(self):
        if not any(field.name == START_FROM for field in self.fields):
            return False
        return self.chosen(START_FROM) != FRESH

    def open(self):
        self.visible = True
        self._observer.on_event(FormOpened(self.command.name))

    def cancel(self):
        self.visible = False
        self._observer.on_event(FormCancelled(self.command.name))

    def text(self, name):
        return self._texts[name]

    def chosen(self, name):
        field = self._field(name)
        return field.choices[self._positions[name]]

    def set_text(self, name, text):
        self._texts[name] = text

    def locked(self, name):
        return self._field(name).shape and self._continuing()

    def move(self, step):
        open_fields = [field for field in self.fields if not self.locked(field.name)]
        if not open_fields:
            return
        try:
            here = open_fields.index(self.current)
        except ValueError:
            here = 0
        self.current = open_fields[(here + step) % len(open_fields)]

    def move_choice(self, name, step):
        field = self._field(name)
        self._positions[name] = (self._positions[name] + step) % len(field.choices)

    def accept(self):
        values = {}
        for field in self.fields:
            if self.locked(field.name):
                continue
            if field.kind == "choice":
                values[field.name] = self.chosen(field.name)
                continue
            value = self._value(field)
            if value is None:
                self.message = _rejection(field)
                self.current = field
                return None
            values[field.name] = value
        self.message = None
        self.visible = False
        return values

    def _value(self, field):
        rule = RULES[field.check]
        try:
            parsed = rule.parse(self._texts[field.name])
        except ValueError:
            return None
        return parsed if rule.allows(parsed) else None


def label_fragments(form, field):
    if form.locked(field.name):
        return [("class:hint", f"   {field.label}: (from savegame)")]
    if field is form.current:
        return [("class:selected", f" » {field.label}:")]
    return [("", f"   {field.label}:")]


def choice_fragments(form, field):
    fragments = list(label_fragments(form, field))
    for choice in field.choices:
        if choice == form.chosen(field.name):
            fragments.append(("class:selected", f" [{choice}]"))
        else:
            fragments.append(("", f" {choice}"))
    return fragments


def message_fragments(form):
    if not form.message:
        return []
    return [("class:failed", form.message)]


class FormOverlay:
    def __init__(self, on_accept, on_cancel, focus):
        self.form = None
        self._on_accept = on_accept
        self._on_cancel = on_cancel
        self._focus = focus
        self.controls = {}
        self.body = _closed_body()
        self.container = ConditionalContainer(
            DynamicContainer(lambda: self.body),
            Condition(lambda: self.form is not None and self.form.visible),
        )
        self.float = Float(content=self.container, top=FORM_TOP)

    @property
    def visible(self):
        return self.form is not None and self.form.visible

    def open(self, form):
        self.form = form
        form.open()
        self._build()
        self.focus(form.current.name)

    def close(self):
        self.form = None
        self.controls = {}
        self.body = _closed_body()

    def focus(self, name):
        self._focus(self.controls[name])

    def _build(self):
        rows = []
        self.controls = {}
        for field in self.form.fields:
            if field.kind == "choice":
                rows.append(self._choice_row(field))
            else:
                rows.extend(self._text_rows(field))
        rows.append(
            Window(
                FormattedTextControl(lambda: message_fragments(self.form)),
                wrap_lines=True,
            ),
        )
        self.body = Dialog(
            body=HSplit(rows),
            title=self.form.command.name,
            width=FORM_WIDTH,
        )

    def _choice_row(self, field):
        control = FormattedTextControl(
            lambda: choice_fragments(self.form, field),
            focusable=True,
            key_bindings=self._keys(field, None),
        )
        self.controls[field.name] = control
        return Window(control, height=1)

    def _text_rows(self, field):
        editor = TextArea(text=self.form.text(field.name), multiline=False)
        editor.buffer.cursor_position = len(editor.text)

        def sync():
            self.form.set_text(field.name, editor.text)

        editor.control.key_bindings = self._keys(field, sync)
        self.controls[field.name] = editor.control
        label = Window(
            FormattedTextControl(lambda: label_fragments(self.form, field)),
            height=1,
            width=LABEL_WIDTH,
        )
        marker = Window(
            FormattedTextControl(lambda: label_fragments(self.form, field)),
            height=1,
        )
        shown = ConditionalContainer(
            VSplit([label, editor]),
            Condition(lambda: not self.form.locked(field.name)),
        )
        hidden = ConditionalContainer(
            marker,
            Condition(lambda: self.form.locked(field.name)),
        )
        return shown, hidden

    def _keys(self, field, sync):
        keys = KeyBindings()

        @keys.add("tab")
        def _(event):
            if sync:
                sync()
            self.form.move(1)
            self.focus(self.form.current.name)

        @keys.add("s-tab")
        def _(event):
            if sync:
                sync()
            self.form.move(-1)
            self.focus(self.form.current.name)

        @keys.add("enter")
        def _(event):
            if sync:
                sync()
            self._enter()

        @keys.add("escape")
        def _(event):
            self.form.cancel()
            self.close()
            self._on_cancel()

        if field.kind == "choice":

            @keys.add("up")
            def _(event):
                self.form.move_choice(field.name, -1)

            @keys.add("down")
            def _(event):
                self.form.move_choice(field.name, 1)

        return keys

    def _enter(self):
        values = self.form.accept()
        if values is None:
            self.focus(self.form.current.name)
            return
        form = self.form
        self.close()
        self._on_accept(form, values)


def _closed_body():
    return Window(FormattedTextControl(lambda: []))
