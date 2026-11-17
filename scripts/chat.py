import sys
from pathlib import Path

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.adapters.story import FileStoryStore
from murrai.composition.chat import chat_menu, load_checkpoint_session
from murrai.presentation.chat import run_chat
from murrai.presentation.chat.lanes import ChatLog


def main():
    corpus = Path("data/corpus")
    if not any(corpus.glob("*.txt")):
        print("no corpus yet; run: make corpus", file=sys.stderr)
        sys.exit(1)

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    store = FileCheckpointStore("checkpoints")
    stories = FileStoryStore("sessions")
    session = load_checkpoint_session(store, seed) if store.names() else None
    if session is None:
        print("no trained model yet; ctrl-p opens the menu")

    live = {"session": session}
    run_chat(
        session,
        menu=chat_menu(
            corpus,
            seed=seed,
            store=store,
            stories=stories,
            session=lambda: live["session"],
        ),
        observer=ChatLog("chat.log"),
        load=lambda name: load_checkpoint_session(store, seed, name),
        on_swap=lambda next_session: live.update(session=next_session),
    )


if __name__ == "__main__":
    main()
