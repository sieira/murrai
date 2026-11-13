import glob
import time
from pathlib import Path

from murrai.domain.tokenizer import TokenizerEncoderDecoder

CORPUS_DIR = Path("data/corpus")


def timed_fit(text, merges):
    start = time.perf_counter()
    tokenizer = TokenizerEncoderDecoder().fit(text, merges)
    seconds = time.perf_counter() - start
    return tokenizer, seconds


def report(name, text, merges):
    tokenizer, seconds = timed_fit(text, merges)
    print(
        f"{name}: {len(text):,} chars, {merges} merges"
        f" -> {len(tokenizer.merges)} learned in {seconds:.2f}s",
    )
    start = time.perf_counter()
    ids = tokenizer.encode(text)
    print(
        f"  encode: {len(ids):,} tokens in {time.perf_counter() - start:.2f}s",
    )


report(
    "synthetic",
    "".join(["the quick brown fox jumps over the lazy dog "] * 20000),
    300,
)

paths = sorted(glob.glob(str(CORPUS_DIR / "*.txt")))
if paths:
    real = "".join(Path(p).read_text(errors="ignore") for p in paths)
    report("corpus", real, 300)
else:
    print("corpus: data/corpus/ not present, skipped")
