import math

from typing import TypeVar

import torch
import torch.nn as nn

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class PositionalEncoder(nn.Module):
    def __init__(self, max_len: int, embedding_dim: int, dropout: float = 0.1):
        """Implement the positional encoder module.

        The class implements the positional encoder part of a transformer
        architecture. The positional encoder takes in a latent vector and adds
        a positional encoding to it that describes the position of the vector
        in the sequence.

        Args:
            max_seq_len (int): The maximum sequence length.
            embedding_dim (int): The dimension of the embedding vectors.
        
        Attributes:
            position_encoding (Tensor): The positional encoding matrix.

        Example:
            >>> import torch
            >>> from pygment.modules import PositionalEncoder
            >>> pe = PositionalEncoder(10, 64)
            >>> latent = torch.randn(1, 10, 64)
            >>> embedding = pe(latent)
            >>> print(embedding.shape)
            torch.Size([1, 10, 64])
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term) 

        # We register the positional encoding as a buffer so that we can
        # include it in the `state_dict` of the model but not in the model
        # parameters. This is because the positional encoding is not updated
        # during training.
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Implement the forward pass of the positional encoder.

        Args:
            x (Tensor): The latent vector to be encoded.
        
        Returns:
            embedding (Tensor): The encoded vector.
        """     
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
