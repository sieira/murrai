import sys
from pathlib import Path

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.composition.chat import chat_stages
from murrai.presentation.chat import run_chat, select_stage

corpus = Path("data/corpus")
if not any(corpus.glob("*.txt")):
    print("no corpus yet; run: make corpus", file=sys.stderr)
    sys.exit(1)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
store = FileCheckpointStore("checkpoints")
stage = select_stage(chat_stages(corpus, seed=seed, store=store))
if stage is None:
    sys.exit(0)
if stage.trains:
    print(f"training {stage.name}...")
run_chat(stage.build())
