import sys
from pathlib import Path

from murrai.adapters.checkpoint import FileCheckpointStore
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
    session = load_checkpoint_session(store, seed) if store.names() else None
    if session is None:
        print("no trained model yet; ctrl-p opens the menu")

    run_chat(
        session,
        menu=chat_menu(corpus, seed=seed, store=store),
        observer=ChatLog("chat.log"),
        load=lambda name: load_checkpoint_session(store, seed, name),
    )


if __name__ == "__main__":
    main()
