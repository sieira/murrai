import sys
from pathlib import Path

from murrai.composition.chat import chat_stages
from murrai.presentation.chat import select_stage
from murrai.presentation.chat.repl import interactive_chat

corpus = Path("data/corpus")
if not any(corpus.glob("*.txt")):
    print("no corpus yet; run: make corpus", file=sys.stderr)
    sys.exit(1)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stage = select_stage(chat_stages(corpus, seed=seed))
if stage is None:
    sys.exit(0)
if stage.trains:
    print(f"training {stage.name}...")
interactive_chat(stage.build())
