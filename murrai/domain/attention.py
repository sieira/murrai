import torch


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.query = torch.nn.Linear(dimension, dimension, bias=False)
        self.key = torch.nn.Linear(dimension, dimension, bias=False)
        self.value = torch.nn.Linear(dimension, dimension, bias=False)
        self.out = torch.nn.Linear(dimension, dimension, bias=False)
        self.shares = None

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) / x.size(-1) ** 0.5
        time = x.size(-2)
        future = torch.triu(
            torch.ones(time, time, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(future, float("-inf"))
        self.shares = torch.softmax(scores, dim=-1)
        return self.out(self.shares @ v)
