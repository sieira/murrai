import torch


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, dimension, heads=1):
        super().__init__()
        if dimension % heads:
            raise ValueError(f"dimension {dimension} is not divisible by heads {heads}")
        self.heads = heads
        self.query = torch.nn.Linear(dimension, dimension, bias=False)
        self.key = torch.nn.Linear(dimension, dimension, bias=False)
        self.value = torch.nn.Linear(dimension, dimension, bias=False)
        self.out = torch.nn.Linear(dimension, dimension, bias=False)
        self.shares = None

    def _split(self, x):
        batch, time, dimension = x.shape
        return x.view(batch, time, self.heads, dimension // self.heads).transpose(1, 2)

    def forward(self, x):
        batch, time, dimension = x.shape
        q = self._split(self.query(x))
        k = self._split(self.key(x))
        v = self._split(self.value(x))
        scores = q @ k.transpose(-2, -1) / (dimension // self.heads) ** 0.5
        future = torch.triu(
            torch.ones(time, time, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future, float("-inf"))
        self.shares = torch.softmax(scores, dim=-1)
        blended = (self.shares @ v).transpose(1, 2).reshape(batch, time, dimension)
        return self.out(blended)
