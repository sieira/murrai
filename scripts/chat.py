import sys
from pathlib import Path

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.composition.chat import (
    chat_commands,
    chat_stages,
    load_checkpoint_session,
)
from murrai.presentation.chat import run_chat, select_stage

corpus = Path("data/corpus")
if not any(corpus.glob("*.txt")):
    print("no corpus yet; run: make corpus", file=sys.stderr)
    sys.exit(1)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
store = FileCheckpointStore("checkpoints")
if store.names():
    name = store.names()[-1]
    print(f"loaded {name}; ctrl-p for commands")
    session = load_checkpoint_session(store, seed)
else:
    stage = select_stage(chat_stages(corpus, seed=seed, store=store))
    if stage is None:
        sys.exit(0)
    if stage.trains:
        print(f"training {stage.name}...")
    session = stage.build()

run_chat(
    session,
    commands=lambda: chat_commands(corpus, seed=seed, store=store),
)
