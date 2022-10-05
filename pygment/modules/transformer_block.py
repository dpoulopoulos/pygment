import torch
import torch.nn as nn

from typing import TypeVar

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class TransformerBlock(nn.Module):
    """Implement a transformer block.

    The class implements a transformer block. The transformer block is a
    self-attention block that takes in a sequence of vectors and outputs a
    sequence of vectors. The transformer block is used to encode the input
    sequence into a latent vector.

    Citation:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
          A. N., ... & Polosukhin, I. (2017). Attention is all you need.
          Advances in neural information processing systems, 30.
    
    Args:
        embedding_dims (int): The number of embedding dimensions.
        n_heads (int): The number of heads in the multi-head attention layer.
            Note that embed_dim will be split across num_heads (i.e. each head
            will have dimension embed_dim // num_heads).
        seq_len (int): The maximum sequence length.
        dropout (float): Dropout probability on ``attn_output_weights``.
    
    Attributes:
        ln_1 (nn.LayerNorm): The first layer normalization layer.
        ln_2 (nn.LayerNorm): The second layer normalization layer.
        attention (nn.MultiheadAttention): The multi-head attention layer.
        fc (nn.Sequential): The feed-forward network.
        mask (torch.Tensor): The mask to prevent the model from attending to
            future tokens.

    Example:
        >>> import torch
        >>> from pygment.modules import TransformerBlock
        >>> x = torch.randn(1, 256, 256)
        >>> model = TransformerBlock(256, 8, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 256, 256])
    """
    def __init__(self, embedding_dims: int, n_heads: int, seq_len: int,
                 device: torch.device, dropout: float = 0.):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embedding_dims)
        self.ln_2 = nn.LayerNorm(embedding_dims)

        self.attention = nn.MultiheadAttention(embedding_dims, n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dims, 4 * embedding_dims),
            nn.GELU(),
            nn.Linear(4 * embedding_dims, embedding_dims),
        )

        self.mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Implement the forward pass of the transformer block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tensor (torch.Tensor): The output tensor.
        """
        attn_output, _ = self.attention(query=x, key=x, value=x,
                                        attn_mask=self.mask)
        x = x + attn_output
        x = self.ln_1(x)
        x = x + self.fc(x)
        x = self.ln_2(x)
        return x
