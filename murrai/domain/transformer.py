import math
from itertools import batched

import torch
import torch.nn.functional as F

from murrai.domain.block import TransformerBlock
from murrai.domain.tokenizer import TokenizerEncoderDecoder

ROWS_PER_PASS = 512


class TransformerLM(torch.nn.Module):
    kind = "gpt"

    def __init__(
        self,
        tokenizer: TokenizerEncoderDecoder,
        depth: int,
        dimension: int,
        max_time: int,
        heads: int = 1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.start = tokenizer.vocab_size
        self.max_time = max_time
        self.embed = torch.nn.Embedding(tokenizer.vocab_size + 1, dimension)
        self.position = torch.nn.Embedding(max_time, dimension)
        self.blocks = torch.nn.ModuleList(
            TransformerBlock(dimension, heads) for _ in range(depth)
        )
        self.norm = torch.nn.LayerNorm(dimension)
        self.head = torch.nn.Linear(dimension, tokenizer.vocab_size)

    def state(self):
        return {
            "depth": len(self.blocks),
            "dimension": self.embed.embedding_dim,
            "max_time": self.max_time,
            "heads": self.blocks[0].attend.heads,
            "parameters": self.state_dict(),
        }

    @classmethod
    def from_state(cls, tokenizer, state):
        model = cls(
            tokenizer,
            depth=state["depth"],
            dimension=state["dimension"],
            max_time=state["max_time"],
            heads=state.get("heads", 1),
        )
        model.load_state_dict(state["parameters"])
        return model

    def _device(self):
        return self.embed.weight.device

    def _rows(self, contexts: list[list[int]]) -> torch.Tensor:
        trimmed = [context[-self.max_time :] or [self.start] for context in contexts]
        widest = max(len(row) for row in trimmed)
        rows = [[self.start] * (widest - len(row)) + row for row in trimmed]
        return torch.tensor(rows, dtype=torch.long, device=self._device())

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        x = self.embed(rows) + self.position(
            torch.arange(rows.size(1), device=rows.device),
        )
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
        ids = [self.start, *self.tokenizer.encode(text)]
        stream = torch.tensor(ids, dtype=torch.long, device=self._device())
        row_losses = []
        for starts in batched(
            range(0, len(stream) - 1, self.max_time),
            ROWS_PER_PASS,
            strict=False,
        ):
            rows = [stream[start : start + self.max_time] for start in starts]
            widest = max(len(row) for row in rows)
            batch = torch.full(
                (len(rows), widest),
                self.start,
                dtype=torch.long,
                device=self._device(),
            )
            targets = torch.full(
                (len(rows), widest),
                -100,
                dtype=torch.long,
                device=self._device(),
            )
            for index, row in enumerate(rows):
                batch[index, : len(row)] = row
                targets[index, : len(row) - 1] = row[1:]
            with torch.no_grad():
                logits = self.forward(batch)
            per_position = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(len(rows), widest)
            row_losses.extend(
                per_position[index, : len(row) - 1].mean()
                for index, row in enumerate(rows)
            )
        return float(torch.stack(row_losses).mean())

    def perplexity(self, text: str) -> float:
        return math.exp(self.nll(text))

    def generate(self, n_tokens: int, seed: int) -> str:
        generator = torch.Generator(device=self._device()).manual_seed(seed)
        ids: list[int] = []
        for _ in range(n_tokens):
            row = self.probs(ids[-self.max_time :])
            nxt = int(torch.multinomial(row, 1, generator=generator))
            ids.append(nxt)
        return self.tokenizer.decode(ids)
