import sys

from murrai.composition.app import train_from_directory

directory = sys.argv[1]
result = train_from_directory(directory)
model = result.model
print(f"train chars: {result.train_chars}")
print(f"held-out chars: {result.held_out_chars}")
print(f"held-out perplexity: {result.held_out_perplexity:.3f}")
print("---")
print(model.generate(200, seed=0))
