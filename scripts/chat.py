import sys
from pathlib import Path

from murrai.composition.chat import chat_session_from_directory
from murrai.presentation.chat import run_chat

corpus = Path("data/corpus")
if not any(corpus.glob("*.txt")):
    print("no corpus yet; run: make corpus", file=sys.stderr)
    sys.exit(1)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
run_chat(
    chat_session_from_directory(corpus, seed=seed),
    sys.stdin.readline,
    sys.stdout.write,
)
