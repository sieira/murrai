import random

import torch


class NeuralBigram:
    kind = "neural-bigram"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.start = tokenizer.vocab_size
        rows = tokenizer.vocab_size + 1
        self.weights = torch.zeros(
            rows,
            tokenizer.vocab_size,
            requires_grad=True,
        )

    def logits(self, prev_id):
        return self.weights[prev_id]

    def state(self):
        return {"weights": self.weights.tolist()}

    @classmethod
    def from_state(cls, tokenizer, state):
        model = cls(tokenizer)
        with torch.no_grad():
            model.weights.copy_(
                torch.tensor(state["weights"], dtype=torch.float32),
            )
        return model

    def batch_loss(self, prev_ids, next_ids):
        prevs = torch.tensor(prev_ids, dtype=torch.long)
        nexts = torch.tensor(next_ids, dtype=torch.long)
        return torch.nn.functional.cross_entropy(self.weights[prevs], nexts)

    def probs(self, context_ids):
        prev = context_ids[-1] if context_ids else self.start
        return torch.softmax(self.weights[prev], dim=-1).tolist()

    def update(self, learning_rate):
        with torch.no_grad():
            self.weights -= learning_rate * self.weights.grad
            self.weights.grad.zero_()

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
