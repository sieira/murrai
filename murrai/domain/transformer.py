import math

import torch
import torch.nn.functional as F

from murrai.domain.block import TransformerBlock
from murrai.domain.vocabulary import Vocabulary


class TransformerLM(torch.nn.Module):
    kind = "char-gpt"

    def __init__(
        self,
        vocabulary: Vocabulary,
        depth: int,
        dimension: int,
        max_time: int,
        heads: int = 1,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.start = vocabulary.size
        self.max_time = max_time
        self.embed = torch.nn.Embedding(vocabulary.size + 1, dimension)
        self.position = torch.nn.Embedding(max_time, dimension)
        self.blocks = torch.nn.ModuleList(
            TransformerBlock(dimension, heads) for _ in range(depth)
        )
        self.norm = torch.nn.LayerNorm(dimension)
        self.head = torch.nn.Linear(dimension, vocabulary.size)

    def state(self):
        return {
            "depth": len(self.blocks),
            "dimension": self.embed.embedding_dim,
            "max_time": self.max_time,
            "heads": self.blocks[0].attend.heads,
            "parameters": self.state_dict(),
        }

    @classmethod
    def from_state(cls, vocabulary, state):
        model = cls(
            vocabulary,
            depth=state["depth"],
            dimension=state["dimension"],
            max_time=state["max_time"],
            heads=state.get("heads", 1),
        )
        model.load_state_dict(state["parameters"])
        return model

    def _rows(self, contexts: list[list[int]]) -> torch.Tensor:
        trimmed = [context[-self.max_time :] or [self.start] for context in contexts]
        widest = max(len(row) for row in trimmed)
        rows = [[self.start] * (widest - len(row)) + row for row in trimmed]
        return torch.tensor(rows, dtype=torch.long)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        x = self.embed(rows) + self.position(torch.arange(rows.size(1)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def forward_batch(self, contexts: list[list[int]]) -> torch.Tensor:
        return self(self._rows(contexts))

    def logits(self, context_ids: list[int]) -> torch.Tensor:
        return self.forward_batch([context_ids])[0, -1]

    def probs(self, context_ids: list[int]) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.logits(context_ids), dim=-1)

    def nll(self, text: str) -> float:
        ids = [self.start, *self.vocabulary.encode(text)]
        stream = torch.tensor(ids, dtype=torch.long)
        losses = []
        for start in range(0, len(stream) - 1, self.max_time):
            row = stream[start : start + self.max_time]
            with torch.no_grad():
                logits = self.forward(row[None])
            losses.append(
                F.cross_entropy(
                    logits[0, :-1],
                    stream[start + 1 : start + self.max_time],
                ),
            )
        return float(torch.stack(losses).mean())

    def perplexity(self, text: str) -> float:
        return math.exp(self.nll(text))

    def generate(self, n_chars: int, seed: int) -> str:
        generator = torch.Generator().manual_seed(seed)
        ids: list[int] = []
        for _ in range(n_chars):
            row = self.probs(ids[-self.max_time :])
            nxt = int(torch.multinomial(row, 1, generator=generator))
            ids.append(nxt)
        return self.vocabulary.decode(ids)
