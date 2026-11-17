import sys
import tempfile
from pathlib import Path

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.adapters.events import ListenerBus
from murrai.application.ports import Stage
from murrai.application.runs import (
    train_and_held_out,
    train_transformer_lm,
    training_device,
)

SMOKE = Path("data/corpus/tinystories.txt")


def report(event):
    if event.phase is not Stage.TRAINING:
        return
    if event.step % 20 == 0 or event.step == event.total_steps:
        print(f"step {event.step} / {event.total_steps}", flush=True)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    text = SMOKE.read_text()
    train_text, held_text = train_and_held_out(text, 0.05)
    device = training_device()
    print(f"smoke corpus: {len(text):,} chars on {device}")

    bus = ListenerBus()
    bus.subscribe(report)
    common = {
        "batch": 64,
        "seed": 0,
        "depth": 2,
        "dimension": 64,
        "max_time": 64,
        "heads": 2,
        "held_out_text": held_text,
        "validate_every": 60,
        "device": device,
        "bus": bus,
    }
    whole = train_transformer_lm(train_text, steps=steps, **common)
    bus.flush()
    part = steps // 3
    with tempfile.TemporaryDirectory() as folder:
        store = FileCheckpointStore(Path(folder) / "checkpoints")
        stopped = train_transformer_lm(
            train_text,
            steps=part,
            checkpoint_every=part,
            mid_run_checkpoints=lambda snapshot: store.save("midrun", snapshot),
            **common,
        )
        bus.flush()
        resumed = train_transformer_lm(
            train_text,
            steps=steps - part,
            checkpoint=store.load("midrun"),
            **common,
        )
        bus.flush()
        gap = max(
            abs(a - b)
            for a, b in zip(resumed.loss_curve, whole.loss_curve[part:], strict=True)
        )
        assert gap < 0.05, f"resume diverged (max step gap {gap:.4f})"
        params = sum(p.numel() for p in whole.model.parameters())
        print(f"model: {params:,} parameters")
        print(
            f"whole:    loss {whole.loss_curve[-1]:.3f}"
            f"  tokens/s {whole.tokens_per_second:.0f}",
        )
        print(f"whole validation: {whole.validation_curve}")
        print(f"stopped:  loss {stopped.loss_curve[-1]:.3f}  steps {stopped.steps}")
        print(f"resumed:  loss {resumed.loss_curve[-1]:.3f}  steps {resumed.steps}")
        print(f"resumed validation: {resumed.validation_curve}")
        print(f"resumed tokens/s {resumed.tokens_per_second:.0f}")
        print("sample from the resumed model:")
        print(resumed.model.generate(200, seed=7))


if __name__ == "__main__":
    main()
