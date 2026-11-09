import math

import torch
import torch.nn.functional as F

from murrai.domain.vocabulary import Vocabulary


class MlpLM(torch.nn.Module):
    def __init__(
        self, vocabulary: Vocabulary, window: int, dimension: int, hidden: int
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.start = vocabulary.size
        self.window = window
        self.embed = torch.nn.Embedding(vocabulary.size + 1, dimension)
        self.input = torch.nn.Linear(window * dimension, hidden)
        self.output = torch.nn.Linear(hidden, vocabulary.size)

    def _pack(self, contexts: list[list[int]]) -> torch.Tensor:
        rows = []
        for context in contexts:
            tail = context[-self.window :]
            missing = self.window - len(tail)
            rows.append([self.start] * missing + tail)
        return torch.tensor(rows, dtype=torch.long)

    def forward_batch(self, contexts: list[list[int]]) -> torch.Tensor:
        packed = self._pack(contexts)
        vectors = self.embed(packed)
        hidden = torch.tanh(self.input(vectors.flatten(start_dim=1)))
        return self.output(hidden)

    def logits(self, context_ids: list[int]) -> torch.Tensor:
        return self.forward_batch([context_ids])[0]

    def probs(self, context_ids: list[int]) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.logits(context_ids), dim=-1)

    def nll(self, text: str) -> float:
        ids = self.vocabulary.encode(text)
        contexts = [ids[max(0, i - self.window) : i] for i in range(len(ids))]
        targets = torch.tensor(ids, dtype=torch.long)
        with torch.no_grad():
            losses = F.cross_entropy(
                self.forward_batch(contexts), targets, reduction="none"
            )
        return float(losses.mean())

    def perplexity(self, text: str) -> float:
        return math.exp(self.nll(text))

    def generate(self, n_chars: int, seed: int) -> str:
        generator = torch.Generator().manual_seed(seed)
        ids: list[int] = []
        for _ in range(n_chars):
            row = self.probs(list(ids))
            nxt = int(torch.multinomial(row, 1, generator=generator))
            ids.append(nxt)
        return self.vocabulary.decode(ids)
