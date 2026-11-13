import math

import torch
import torch.nn.functional as F

from murrai.domain.tokenizer import TokenizerEncoderDecoder


class MlpLM(torch.nn.Module):
    kind = "windowed-mlp"

    def __init__(
        self,
        tokenizer: TokenizerEncoderDecoder,
        window: int,
        dimension: int,
        hidden: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.start = tokenizer.vocab_size
        self.window = window
        self.embed = torch.nn.Embedding(tokenizer.vocab_size + 1, dimension)
        self.input = torch.nn.Linear(window * dimension, hidden)
        self.output = torch.nn.Linear(hidden, tokenizer.vocab_size)

    def state(self):
        return {
            "window": self.window,
            "dimension": self.embed.embedding_dim,
            "hidden": self.input.out_features,
            "parameters": self.state_dict(),
        }

    @classmethod
    def from_state(cls, tokenizer, state):
        model = cls(
            tokenizer,
            window=state["window"],
            dimension=state["dimension"],
            hidden=state["hidden"],
        )
        model.load_state_dict(state["parameters"])
        return model

    def _device(self):
        return self.embed.weight.device

    def _pack(self, contexts: list[list[int]]) -> torch.Tensor:
        rows = []
        for context in contexts:
            tail = context[-self.window :]
            missing = self.window - len(tail)
            rows.append([self.start] * missing + tail)
        return torch.tensor(rows, dtype=torch.long, device=self._device())

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
        ids = self.tokenizer.encode(text)
        contexts = [ids[max(0, i - self.window) : i] for i in range(len(ids))]
        targets = torch.tensor(ids, dtype=torch.long, device=self._device())
        with torch.no_grad():
            losses = F.cross_entropy(
                self.forward_batch(contexts),
                targets,
                reduction="none",
            )
        return float(losses.mean())

    def perplexity(self, text: str) -> float:
        return math.exp(self.nll(text))

    def generate(self, n_tokens: int, seed: int) -> str:
        generator = torch.Generator(device=self._device()).manual_seed(seed)
        ids: list[int] = []
        for _ in range(n_tokens):
            row = self.probs(list(ids))
            nxt = int(torch.multinomial(row, 1, generator=generator))
            ids.append(nxt)
        return self.tokenizer.decode(ids)
