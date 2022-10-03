import torch
import torch.nn as nn

from typing import TypeVar

from .residual_block import ResidualBlock

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class Decoder(nn.Module):
    """Define a VAE decoder module.

    The class implements the decoder part of a Variational Autoencoder
    architecture. The decoder takes in a latent vector and outputs a
    reconstructed image.

    The implementation is based on the sonnet VQ-VAE example by DeepMind:
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb

    Args:
        latent_dims (int): The dimensionality of the latent vector.
        n_hid (int): The number of hidden channels.
        output_channels (int): The number of channels in the output image.
    
    Attributes:
        decoder (nn.Sequential): The decoder network architecture.

    Example:
        >>> import torch
        >>> from pygment.modules import Decoder
        >>> x = torch.randn(1, 32, 8, 8)
        >>> decoder = Decoder()
        >>> out = decoder(x)
        >>> out.shape
        torch.Size([1, 3, 32, 32])
    """
    def __init__(self, latent_dims: int = 32, n_hid: int = 64,
                 output_channels: int = 3):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dims, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(2*n_hid, 2*n_hid//4),
            ResidualBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        """Implement the forward pass of the decoder.

        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            Tensor (torch.Tensor): The output tensor.
        """
        return self.decoder(x)
