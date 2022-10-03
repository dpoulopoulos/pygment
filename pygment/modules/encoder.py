import torch
import torch.nn as nn

from typing import TypeVar

from .residual_block import ResidualBlock

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class Encoder(nn.Module):
    """Define a VAE encoder module.
    
    The class implements the encoder part of a Variational Autoencoder
    architecture. The encoder takes in an input image and outputs a latent
    vector.

    The implementation is based on the sonnet VQ-VAE example by DeepMind:
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb

    Args:
        input_channels (int): The number of channels in the input image.
        n_hid (int): The number of hidden channels.
    
    Attributes:
        encoder (nn.Sequential): The encoder network architecture.

    Example:
        >>> import torch
        >>> from pygment.modules import Encoder
        >>> x = torch.randn(1, 3, 32, 32)
        >>> encoder = Encoder()
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([1, 128, 8, 8])
    """
    def __init__(self, input_channels: int = 3, n_hid: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, n_hid,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(2*n_hid, 2*n_hid//4),
            ResidualBlock(2*n_hid, 2*n_hid//4))

    def forward(self, x: Tensor) -> Tensor:
        """Implement the forward pass of the encoder.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            Tensor (torch.Tensor): The output tensor.
        """
        return self.encoder(x)
