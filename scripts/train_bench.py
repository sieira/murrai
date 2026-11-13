import platform
import sys
import tempfile
import time
from pathlib import Path

import torch

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.adapters.corpus import DirTextCorpus
from murrai.application.checkpoints import snapshot_of
from murrai.application.runs import train_mlp_lm, train_transformer_lm

CORPUS = Path("data/corpus")

GPT_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MLP_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
WARMUP_STEPS = 5
TIMING_CHARS = 4_000_000


def devices():
    if torch.cuda.is_available():
        yield torch.device("cuda")
    if torch.backends.mps.is_available():
        yield torch.device("mps")
    yield torch.device("cpu")


def device_name(device):
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name()} (cuda)"
    if device.type == "mps":
        return f"{torch.mps.get_device_name()} (mps)"
    return "cpu"


def time_run(run, steps, warmup_steps):
    run(warmup_steps)
    warm_elapsed, _ = run(warmup_steps)
    started = time.perf_counter()
    _, result = run(steps)
    elapsed = time.perf_counter() - started
    rate = (steps - warmup_steps) / (elapsed - warm_elapsed)
    return rate, result


def bench_gpt(device, text):
    text = text[:TIMING_CHARS]

    def run(steps):
        started = time.perf_counter()
        result = train_transformer_lm(
            text,
            steps=steps,
            batch=256,
            seed=0,
            depth=4,
            dimension=64,
            max_time=64,
            holdout=0.0005,
            device=device,
        )
        return time.perf_counter() - started, result

    rate, result = time_run(run, GPT_STEPS, WARMUP_STEPS)
    tokens = 256 * 64
    print(
        f"gpt : {rate:6.2f} steps/s  {rate * tokens / 1000:8.0f}k tokens/s  "
        f"(loss {result.loss_curve[-1]:.3f})",
    )


def bench_mlp(device, text):
    text = text[:TIMING_CHARS]

    def run(steps):
        started = time.perf_counter()
        result = train_mlp_lm(
            text,
            steps=steps,
            batch=256,
            seed=0,
            window=8,
            dimension=32,
            hidden=256,
            learning_rate=0.5,
            patience=1000,
            holdout=0.0005,
            device=device,
        )
        return time.perf_counter() - started, result

    rate, result = time_run(run, MLP_STEPS, WARMUP_STEPS)
    print(
        f"mlp : {rate:6.2f} steps/s  {rate * 256 / 1000:8.0f}k chars/s  "
        f"(loss {result.loss_curve[-1]:.3f})",
    )


def correctness(device, text):
    result = train_transformer_lm(
        text[:200_000],
        steps=5,
        batch=64,
        seed=0,
        depth=4,
        dimension=64,
        max_time=64,
        device=device,
    )
    sample = result.model.generate(20, seed=0)
    store = FileCheckpointStore(Path(tempfile.mkdtemp()) / "checkpoints-bench")
    store.save("bench", snapshot_of(result.model))
    loaded = store.load("bench")
    saved_devices = {
        tensor.device.type
        for tensor in loaded.state["parameters"].values()
        if isinstance(tensor, torch.Tensor)
    }
    print(
        f"    loss {result.loss_curve[-1]:.3f}, held-out perplexity "
        f"{result.held_out_perplexity:.1f}, sample {sample!r}, "
        f"checkpoint devices {sorted(saved_devices)}",
    )


def main():
    text = DirTextCorpus(CORPUS).text()
    print(
        f"{platform.platform()} | torch {torch.__version__} | "
        f"corpus {len(text):,} chars | timing on first {TIMING_CHARS:,} chars | "
        f"gpt {GPT_STEPS} steps, mlp {MLP_STEPS} steps",
    )
    for device in devices():
        print(f"\n== {device_name(device)} ==")
        bench_gpt(device, text)
        bench_mlp(device, text)
        correctness(device, text)


if __name__ == "__main__":
    main()
