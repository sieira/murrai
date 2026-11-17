import sys
from pathlib import Path

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.adapters.corpus import DirTextCorpus
from murrai.adapters.events import ListenerBus
from murrai.application.ports import Stage
from murrai.application.runs import training_device
from murrai.composition.train import train_transformer_from_corpus

TRAIN_DIR = Path("data/corpus/tinystories-train")
VALID_DIR = Path("data/corpus/tinystories-valid")


def report(event):
    if event.phase is not Stage.TRAINING:
        return
    if event.step % 100 == 0 or event.step == event.total_steps:
        print(f"step {event.step} / {event.total_steps}", flush=True)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    train_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_DIR
    valid_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else VALID_DIR
    if not any(train_dir.glob("*.txt")):
        print(
            f"no corpus in {train_dir}; run: make corpus-tinystories",
            file=sys.stderr,
        )
        sys.exit(1)

    bus = ListenerBus()
    bus.subscribe(report)
    result = train_transformer_from_corpus(
        DirTextCorpus(train_dir),
        steps=steps,
        held_out=DirTextCorpus(valid_dir),
        store=FileCheckpointStore("checkpoints"),
        device=training_device(),
        bus=bus,
    )
    bus.flush()
    print(
        f"steps {result.steps}  loss {result.loss_curve[-1]:.3f}  "
        f"tokens/s {result.tokens_per_second:.0f}",
    )
    if result.held_out_perplexity is not None:
        print(f"held-out perplexity {result.held_out_perplexity:.3f}")


if __name__ == "__main__":
    main()
