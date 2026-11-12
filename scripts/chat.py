import sys
from pathlib import Path

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.composition.chat import chat_commands, load_checkpoint_session
from murrai.presentation.chat import run_chat

corpus = Path("data/corpus")
if not any(corpus.glob("*.txt")):
    print("no corpus yet; run: make corpus", file=sys.stderr)
    sys.exit(1)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
store = FileCheckpointStore("checkpoints")
session = load_checkpoint_session(store, seed) if store.names() else None
if session is None:
    print("no trained model yet; ctrl-p opens commands")

run_chat(session, commands=lambda: chat_commands(corpus, seed=seed, store=store))
