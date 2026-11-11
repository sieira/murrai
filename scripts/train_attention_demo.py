import sys
from pathlib import Path

from murrai.application.runs import (
    train_bigram,
    train_mlp_lm,
    train_transformer_lm,
)


class SliceSource:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def sample(model):
    lines = model.generate(200, seed=17).splitlines() or [""]
    return "\n".join(lines[:3])


def main():
    text = (Path(sys.argv[1]) / "tinystories.txt").read_text()[:300_000]

    bigram = train_bigram(SliceSource(text), holdout=0.1)
    mlp = train_mlp_lm(
        text,
        steps=10000,
        batch=256,
        seed=0,
        learning_rate=0.5,
        patience=1000,
        window=8,
        dimension=32,
        hidden=256,
    )
    gpt = train_transformer_lm(
        text,
        steps=2000,
        batch=256,
        seed=0,
        depth=4,
        dimension=64,
        max_time=64,
    )

    print("| model | held-out perplexity | train loss |")
    print("|---|---|---|")
    for name, result in [
        ("counted bigram", bigram),
        ("windowed mlp", mlp),
        ("char gpt", gpt),
    ]:
        loss = (
            f"{result.loss_curve[0]:.3f} -> {result.loss_curve[-1]:.3f}"
            if hasattr(result, "loss_curve")
            else "-"
        )
        print(f"| {name} | {result.held_out_perplexity:.3f} | {loss} |")

    for name, model in [
        ("counted bigram", bigram.model),
        ("windowed mlp", mlp.model),
        ("char gpt", gpt.model),
    ]:
        print(f"\n--- {name} (seed 17) ---")
        print(sample(model))


if __name__ == "__main__":
    main()
