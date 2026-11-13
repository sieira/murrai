import math
import random

from murrai.domain.autograd import Value


class NeuralBigram:
    kind = "neural-bigram"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.start = tokenizer.vocab_size
        rows = tokenizer.vocab_size + 1
        self.weights = [
            [Value(0.0) for _ in range(tokenizer.vocab_size)] for _ in range(rows)
        ]

    def logits(self, prev_id):
        return self.weights[prev_id]

    def state(self):
        return {"weights": [[value.data for value in row] for row in self.weights]}

    @classmethod
    def from_state(cls, tokenizer, state):
        model = cls(tokenizer)
        for row, values in zip(model.weights, state["weights"], strict=False):
            for value, data in zip(row, values, strict=False):
                value.data = data
        return model

    def batch_loss(self, prev_ids, next_ids):
        loss = Value(0.0)
        rows = {}
        for prev, nxt in zip(prev_ids, next_ids, strict=False):
            rows.setdefault(prev, []).append(nxt)
        for prev, targets in rows.items():
            row = self.logits(prev)
            exps = [logit.exp() for logit in row]
            normalizer = exps[0]
            for e in exps[1:]:
                normalizer = normalizer + e
            for nxt in targets:
                loss = loss + -(exps[nxt] / normalizer).log()
        return loss * (1.0 / len(prev_ids))

    def probs(self, context_ids):
        prev = context_ids[-1] if context_ids else self.start
        row = self.logits(prev)
        exps = [_exp(logit.data) for logit in row]
        total = sum(exps)
        return [e / total for e in exps]

    def update(self, learning_rate):
        for row in self.weights:
            for weight in row:
                weight.data -= learning_rate * weight.grad
                weight.grad = 0.0

    def generate(self, n_tokens, seed):
        rng = random.Random(seed)
        ids = []
        prev = self.start
        for _ in range(n_tokens):
            nxt = self.roll(prev, rng)
            ids.append(nxt)
            prev = nxt
        return self.tokenizer.decode(ids)

    def roll(self, prev_id, rng):
        r = rng.random() * sum(self.probs([prev_id]))
        so_far = 0.0
        for i, p in enumerate(self.probs([prev_id])):
            so_far += p
            if r < so_far:
                return i
        return len(self.probs([prev_id])) - 1


def _exp(x):
    return math.exp(x) if x < 300 else math.exp(300)
