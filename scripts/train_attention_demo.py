import sys

from murrai.adapters.corpus import DirTextCorpus
from murrai.application.runs import (
    train_and_held_out,
    train_bigram,
    train_mlp_lm,
    train_transformer_lm,
)


def sample(model):
    lines = model.generate(200, seed=17).splitlines() or [""]
    return "\n".join(lines[:3])


class TextSlice:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def main():
    text = DirTextCorpus(sys.argv[1]).text()[:300_000]
    train_text, held_text = train_and_held_out(text, 0.1)

    bigram = train_bigram(TextSlice(train_text), held_out_text=held_text)
    mlp = train_mlp_lm(
        train_text,
        steps=10000,
        batch=256,
        seed=0,
        learning_rate=0.5,
        patience=1000,
        window=8,
        dimension=32,
        hidden=256,
        held_out_text=held_text,
    )
    gpt = train_transformer_lm(
        train_text,
        steps=2000,
        batch=256,
        seed=0,
        depth=4,
        dimension=64,
        max_time=64,
        held_out_text=held_text,
    )

    print("| model | held-out perplexity | train loss |")
    print("|---|---|---|")
    for name, result in [
        ("counted bigram", bigram),
        ("windowed mlp", mlp),
        ("gpt", gpt),
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
        ("gpt", gpt.model),
    ]:
        print(f"\n--- {name} (seed 17) ---")
        print(sample(model))


if __name__ == "__main__":
    main()
