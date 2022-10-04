import torch
import torch.nn as nn

from typing import TypeVar

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims, n_heads):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embedding_dims)
        self.ln_2 = nn.LayerNorm(embedding_dims)

        self.attention = nn.MultiheadAttention(embedding_dims, n_heads)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dims, 4 * embedding_dims),
            nn.GELU(),
            nn.Linear(4 * embedding_dims, embedding_dims),
        )
        
        self.mask = torch.tril(torch.ones(256, 256))

    def forward(self, x: Tensor) -> Tensor:
        x, _ = x + self.attention(x, x, x, attn_mask=self.mask)
        x = self.ln_1(x)
        x = x + self.fc(x)
        x = self.ln_2(x)
        return x
