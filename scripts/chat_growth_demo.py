"""Type an unknown character into a piped chat and watch it survive.

The model is a neural bigram trained on "abab", so it knows only a and b.
The demo types "abé", presses enter, then ctrl-c to quit. Before the fix
this killed the chat; now the character list grows and the ghost answers
for the new character right away.
"""

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.application.runs import train_neural_bigram
from murrai.presentation.chat import ChatSession, run_chat
from murrai.presentation.chat.repl import GhostChat, GhostSuggestions


def main():
    model = train_neural_bigram("abab", steps=3, batch=4, seed=0).model
    session = ChatSession(model, seed=0)
    with create_pipe_input() as pipe:
        pipe.send_text("abé\r\x03")
        returned = run_chat(session, input=pipe, output=DummyOutput())
    print(f"the chat exited cleanly: {returned is session}")
    print(f"story: {session.text!r}")
    print(f"character list: {session.chars}")
    ghost = GhostSuggestions(GhostChat(session))
    print(f"ghost for the new character: {ghost.real_for('abé')!r}")


if __name__ == "__main__":
    main()
