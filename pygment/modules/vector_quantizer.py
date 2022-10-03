import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import TypeVar

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class VectorQuantizer(nn.Module):
    """Define a vector quantizer module.
    
    The class implements the latent quantizer part of a VAE. The quantizer
    takes in a latent vector and snaps it to a learnable discrete set of
    vectors (codebook).

    Args:
        n_embeding (int): The number of embedding vectors in the codebook.
        embedding_dim (int): The dimension of the embedding vectors.
    
    Attributes:
        embedding (nn.Embedding): The embedding layer that holds the codebook.
        kl_scale (float): The scale of the KL divergence loss.

    Example:
        >>> import torch
        >>> from pygment.modules import VectorQuantizer
        >>> vq = VectorQuantizer(10, 64)
        >>> latent = torch.randn(1, 64, 32, 32)
        >>> quantized_latent, vq_loss, _ = vq(latent)
        >>> print(quantized_latent.shape)
        torch.Size([1, 64, 32, 32])
    """
    def __init__(self, n_embeding, embedding_dim):
        super().__init__()

        self.n_embeding = n_embeding
        self.embedding_dim = embedding_dim

        self.kl_scale = 10.0

        self.embedding = nn.Embedding(self.n_embeding, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.n_embeding,
                                            +1 / self.n_embeding)

    def forward(self, z: Tensor) -> Tensor:
        """Implement the forward pass of the quantizer.
        
        Args:
            z (Tensor): The latent vector to be quantized.

        Returns:
            z_q (Tensor): The quantized latent vector.
            vq_loss (Tensor): The loss of the quantization.
            indices (Tensor): The indices of the embedding vectors in the
              codebook.
        """
        # [B x C x H x W] -> [B x H x W x C]
        z = z.permute(0, 2, 3, 1).contiguous()
        # [BHW x C]
        flat_z = z.view(-1, self.embedding_dim)

        # Compute the L2 distance between the latent vector and the
        # embedding weights
        dist = torch.sum(flat_z**2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight**2, dim=1) - \
               2 * torch.matmul(flat_z, self.embedding.weight.t())

        # Get the embedding that has the min distance
        embedding_indices = torch.argmin(dist, dim=1)

        # [BHW, C] -> [B x H x W x C]
        z_q = self.embedding(embedding_indices).view(z.shape)

        # Compute the vq loss  to train the embeddings
        vq_loss = torch.mean(
            ((z_q.detach() - z)**2) +
              self.kl_scale * torch.mean((z_q - z.detach())**2))

        # Noop in forward pass, straight-through gradient estimator
        # in backward pass. This way, the gradient of the quantized latent
        # gets "copied" to the gradient of the latent.
        z_q = z + (z_q - z).detach()

        # Return the channel dimension to the original position.
        # [B x H x W x C] -> [B x C x H x W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, vq_loss, embedding_indices.view(z_q.shape[0], -1)
