from argparse import ArgumentParser

from typing import TypeVar, Tuple, Dict

import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.optim import optimizer

from pygment.modules import Encoder, Decoder, VectorQuantizer

Tensor = TypeVar("Tensor", bound=torch.Tensor)
Optimizer = TypeVar("Optimizer", bound=optimizer.Optimizer)


class VQVAE(pl.LightningModule):
    """Implement a VQ-VAE model.

    The class implements the VQ-VAE model. The model takes in an input image
    and outputs a reconstructed image. This way, the model learns to compress
    the image into a latent vector which describes the inherent structure of
    the image. The latent vector is then quantized to a discrete set of
    vectors (codebook) and then decoded back to the original image.

    Since the model is trained to reconstruct the original image, it learns
    the undelying distrbution of the data. This way, the model can be used
    to generate new images that are similar to the ones it has seen in the
    training dataset.

    Citation:
        Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation
          learning. Advances in neural information processing systems, 30.

    Args:
        input_channels (int): The number of channels in the input image.
        n_hid (int): The number of hidden channels.
        latent_dims (int): The number of latent dimensions.
        n_embeding (int): The number of embedding vectors in the codebook.
        embedding_dim (int): The dimensionality of the embedding vectors in
          the codebook.
    
    Attributes:
        encoder (Encoder): The encoder network.
        decoder (Decoder): The decoder network.
        quantizer (VectorQuantizer): The vector quantizer.

    Example:
        >>> import torch
        >>> from pygment.models import VQVAE
        >>> x = torch.randn(1, 3, 32, 32)
        >>> from argparse import ArgumentParser
        >>> args = ArgumentParser()
        >>> args.input_channels = 3
        >>> args.n_hid = 32
        >>> args.latent_dims = 64
        >>> args.n_embeding = 512
        >>> args.embedding_dim = 64
        >>> model = VQVAE(args)
        >>> reconstruction, vq_loss, _ = model(x)
        >>> reconstruction.shape
        torch.Size([1, 3, 32, 32])
    """
    def __init__(self, args: ArgumentParser):
        super().__init__()
        self.encoder = Encoder(args.input_channels, args.n_hid)
        self.decoder = Decoder(args.latent_dims, args.n_hid,
                               args.input_channels)
        self.quantizer = VectorQuantizer(args.codebook_embeddings,
                                         args.codebook_dims)

    def forward(self, input: Tensor) -> Tensor:
        """Implement the forward pass of the model.
        
        Args:
            input (Tensor): The input image.
        
        Returns:
            Tensor: The reconstructed image.
            Tensor: The vector quantization loss.
        """
        latent = self.encoder(input)
        quantized_latent, vq_loss, indices = self.quantizer(latent)
        reconstruction = self.decoder(quantized_latent)

        return reconstruction, vq_loss, indices

    def training_step(self, batch: Tuple) -> Dict:
        """Implement the training step of the model.
        
        Args:
            batch (Tuple): The batch of data.
        
        Returns:
            Dict: The training loss.
        """
        img, _ = batch
        # forward pass
        reconstruction, vq_loss, _ = self.forward(img)
        # reconstruction loss
        reconstruction_loss = F.mse_loss(reconstruction, img)
        # total loss
        loss = reconstruction_loss + vq_loss

        return {"loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "vq_Loss": vq_loss}

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """Implement the validation step of the model.

        Args:
            batch (Tuple): The batch of data.
            batch_idx (int): The batch index.
        
        Returns:
            Dict: The validation loss.
        """
        img, _ = batch
        # forward pass
        reconstruction, vq_loss, _ = self.forward(img)
        # reconstruction loss
        reconstruction_loss = F.mse_loss(reconstruction, img)
        # total loss
        loss = reconstruction_loss + vq_loss
        # log the validation loss for checkpointing
        self.log("val_loss", loss, prog_bar=True)
        
        return {"val_loss": loss,
                "val_reconstruction_loss": reconstruction_loss,
                "val_vq_loss": vq_loss}

    def configure_optimizers(self) -> Optimizer:
        """Configure the optimizer for the model.
        
        Returns:
            Optimizer: The training optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=3e-3,
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=1e-4)

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> ArgumentParser:
        """Add the model arguments to the parser.

        Args:
            parser (ArgumentParser): The parser to add the arguments to.
        
        Returns:
            ArgumentParser: The parser with the added arguments.
        """
        parser = ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument("--input_channels", type=int, default=3,
                            help="The number of channels in the input image.")
        parser.add_argument("--n_hid", type=int, default=64,
                            help="The number of hidden channels.")
        parser.add_argument("--latent_dims", type=int, default=128,
                            help="The number of latent dimensions.")
        parser.add_argument("--codebook_embeddings", type=int, default=256,
                            help="The number of embedding vectors in"
                                 " the codebook.")
        parser.add_argument("--codebook_dims", type=int, default=64,
                            help="The dimensionality of the codebook.")

        return parser
