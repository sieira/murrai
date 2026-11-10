import sys

from murrai.adapters.corpus import DirTextCorpus
from murrai.application.runs import train_mlp_lm

directory = sys.argv[1]
text = DirTextCorpus(directory).text()
result = train_mlp_lm(
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

print(f"train chars: {result.train_chars}")
print(f"held-out chars: {result.held_out_chars}")
print(f"held-out perplexity: {result.held_out_perplexity:.3f}")
print(f"loss: {result.loss_curve[0]:.3f} -> {result.loss_curve[-1]:.3f}")
print(f"learning rate: {result.rate_curve[0]:.3f} -> {result.rate_curve[-1]:.3f}")
print("---")
print(result.model.generate(200, seed=17))
