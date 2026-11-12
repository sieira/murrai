import sys

from murrai.adapters.corpus import DirTextCorpus
from murrai.application.runs import train_transformer_lm


def main():
    text = DirTextCorpus(sys.argv[1]).text()[:300_000]
    heads = int(sys.argv[2])
    result = train_transformer_lm(
        text,
        steps=2000,
        batch=256,
        seed=0,
        depth=4,
        dimension=64,
        max_time=64,
        heads=heads,
    )
    print(f"heads {heads}")
    print(f"held-out perplexity {result.held_out_perplexity:.3f}")
    print(f"train loss {result.loss_curve[0]:.3f} -> {result.loss_curve[-1]:.3f}")
    print("--- sample (seed 17) ---")
    print(result.model.generate(200, seed=17))


if __name__ == "__main__":
    main()
