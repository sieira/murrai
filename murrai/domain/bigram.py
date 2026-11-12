import math

import torch


class Bigram:
    kind = "counted-bigram"

    def __init__(self, vocabulary, counts):
        self.vocabulary = vocabulary
        self.start = vocabulary.size
        probs = counts.to(torch.float32)
        self._probs = probs / probs.sum(dim=1, keepdim=True)

    @classmethod
    def fit(cls, text, vocabulary, on_progress=None):
        counts = torch.ones(vocabulary.size + 1, vocabulary.size)
        chunk = max(1, len(text) // 100)
        previous = cls._start(vocabulary)
        for from_char in range(0, len(text), chunk):
            for nxt in vocabulary.encode(text[from_char : from_char + chunk]):
                counts[previous, nxt] += 1
                previous = nxt
            if on_progress is not None:
                on_progress(min(from_char + chunk, len(text)), len(text))
        return cls(vocabulary, counts)

    @staticmethod
    def _start(vocabulary):
        return vocabulary.size

    def state(self):
        return {"probs": self._probs}

    @classmethod
    def from_state(cls, vocabulary, state):
        return cls(vocabulary, state["probs"])

    def probs(self, context_ids):
        return self._probs[context_ids[-1] if context_ids else self.start]

    def roll(self, prev_id, generator):
        return int(torch.multinomial(self._probs[prev_id], 1, generator=generator))

    def nll(self, text):
        ids = self.vocabulary.encode(text)
        prev = [self.start, *ids[:-1]]
        p = self._probs[prev, ids]
        return float(-p.log().mean())

    def perplexity(self, text):
        return math.exp(self.nll(text))

    def generate(self, n_chars, seed):
        gen = torch.Generator().manual_seed(seed)
        ids = []
        prev = self.start
        for _ in range(n_chars):
            nxt = int(torch.multinomial(self._probs[prev], 1, generator=gen))
            ids.append(nxt)
            prev = nxt
        return self.vocabulary.decode(ids)
