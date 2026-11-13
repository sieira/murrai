import math
from collections import Counter

from murrai.adapters.corpus import DirTextCorpus
from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder


def main():
    text = DirTextCorpus("data/corpus").text()
    train = text[: int(len(text) * 0.8)]
    tokenizer = TokenizerEncoderDecoder().fit(train, 300)
    counted = Bigram.fit(train, tokenizer)

    print("training softmax bigram through the autograd engine...")
    result = train_neural_bigram(train, steps=2000, batch=64, seed=0, learning_rate=3.0)
    neural = result.model

    curve = result.loss_curve
    width = max(f"{m:.2f}".__len__() for m in (min(curve), max(curve)))
    print("\nloss curve (training steps, one every 100):")
    for i in range(0, len(curve), 100):
        bar = "#" * max(
            1,
            int((curve[i] - min(curve)) / (max(curve) - min(curve)) * 40 + 0.5),
        )
        print(f"  step {i:4d}  {curve[i]:{width}.2f}  {bar}")
    print(
        f"\nuniform no-knowledge loss: {math.log(tokenizer.vocab_size):.3f}"
        f"\ncounted bigram loss:       {counted.nll(train):.3f}"
        f"\ntrained bigram loss:       {curve[-1]:.3f}",
    )

    ids = tokenizer.encode(train)
    pairs = zip([tokenizer.vocab_size, *ids][:-1], ids, strict=False)
    prev_counts = Counter(p for p, _ in pairs)
    distances = []
    for prev in range(tokenizer.vocab_size):
        if prev_counts.get(prev, 0) < 50:
            continue
        learned = neural.probs([prev])
        counted_probs = counted.probs([prev]).tolist()
        distance = 0.5 * sum(
            abs(a - b) for a, b in zip(learned, counted_probs, strict=False)
        )
        distances.append(distance)
    print(
        f"\ntable distance to the counted table (rows with >= 50 uses, 0 = identical):"
        f"\n  mean {sum(distances) / len(distances):.3f}   worst {max(distances):.3f}",
    )

    print("\nsample text, counted bigram (seed 17):")
    print(" ", counted.generate(180, seed=17))
    print("sample text, trained bigram (seed 17):")
    print(" ", neural.generate(180, seed=17))


if __name__ == "__main__":
    main()
