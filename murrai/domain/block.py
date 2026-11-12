import torch

from murrai.domain.attention import CausalSelfAttention


class TransformerBlock(torch.nn.Module):
    def __init__(self, dimension, heads=1):
        super().__init__()
        self.attend_norm = torch.nn.LayerNorm(dimension)
        self.attend = CausalSelfAttention(dimension, heads)
        self.mlp_norm = torch.nn.LayerNorm(dimension)
        self.mlp_up = torch.nn.Linear(dimension, 4 * dimension)
        self.mlp_act = torch.nn.GELU()
        self.mlp_down = torch.nn.Linear(4 * dimension, dimension)

    def forward(self, x):
        x = x + self.attend(self.attend_norm(x))
        x = x + self.mlp_down(self.mlp_act(self.mlp_up(self.mlp_norm(x))))
        return x
