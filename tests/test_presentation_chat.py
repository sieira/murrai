from types import SimpleNamespace

from murrai.presentation.chat import render_dice, run_chat


class FakeSession:
    def __init__(self):
        self.calls = []
        self.rolls = {"a": "b", "b": "a", "q": "a"}

    def turn(self, char):
        self.calls.append(("turn", char))
        if char not in ("a", "b", "q"):
            raise ValueError("unknown char")
        return SimpleNamespace(
            dice=[("a", 0.5), ("b", 0.5)], rolled=self.rolls[char], history=char
        )


def test_render_dice_exact():
    out = render_dice("a", [("b", 0.5), ("a", 0.5)], rolled="b")
    assert out == ("after 'a', the die:\n  b 50.0%\n  a 50.0%\nrolled 'b'\n")


def test_render_dice_shows_the_history_tail():
    out = render_dice("abcdefghijk", [("b", 0.5), ("a", 0.5)], rolled="b")
    assert "after '…cdefghijk', the die:" in out


def test_render_dice_shows_only_top():
    ten = [(chr(97 + i), 0.1) for i in range(10)]
    out = render_dice("a", ten, rolled="a")
    assert "  i 10.0%" not in out
    assert "  h 10.0%" in out
    assert "  ... 2 more" in out


def test_run_chat_scripted():
    session = FakeSession()
    out = []
    scripted = iter(["a", "q", "b", ""])
    run_chat(session, lambda: next(scripted, None), out.append)
    assert out[-1].endswith("bye\n")
    assert any("after 'a', the die:" in chunk for chunk in out)
    assert any("rolled 'b'" in chunk for chunk in out)
    assert ("turn", "a") in session.calls
    assert ("turn", "q") in session.calls


def test_run_chat_ctrl_c_quits():
    session = FakeSession()
    out = []

    def read():
        raise KeyboardInterrupt

    run_chat(session, read, out.append)
    assert out[-1].endswith("bye\n")


def test_run_chat_unknown_char_message():
    class Unknown:
        def turn(self, char):
            raise ValueError("unknown char")

    out = []
    scripted = iter(["z", ""])
    run_chat(Unknown(), lambda: next(scripted, None), out.append)
    assert any("unknown" in chunk.lower() for chunk in out)
