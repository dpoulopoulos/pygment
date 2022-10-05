from argparse import ArgumentParser

from typing import TypeVar, Tuple, Dict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import optimizer

from .vq_vae import VQVAE
from .min_gpt import MinGPT

Tensor = TypeVar("Tensor", bound=torch.Tensor)
Optimizer = TypeVar("Optimizer", bound=optimizer.Optimizer)


class VQTransformer(pl.LightningModule):
    """Implement a VQ-Transformer model.

    The class implements the VQ-Transformer model. The model takes in an input
    n an input sequence from a VQ-VAE codebook and outputs a probability
    distribution over the next token in the sequence.

    Args:
        pkeep (float): The probability of keeping a token in the input
            sequence.
        sos_token (int): The start of sequence token.
        vqvae_ckpt (str): The path to the VQ-VAE checkpoint.
    
    Attributes:
        transformer (MinGPT): The transformer network.
        vq_vae (VQVAE): The VQ-VAE network.
    """
    def __init__(self, args: ArgumentParser):
        super().__init__()
        self.args = args

        self.pkeep = args.pkeep
        self.sos_token = args.sos_token

        self.transformer = MinGPT(args)
        self.vq_vae = VQVAE.load_from_checkpoint(args.vqvae_ckpt, args=args)

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode the input image into a latent vector.
        
        Args:
            x (torch.Tensor): The input image.
        
        Returns:
            Tensor (torch.Tensor): The latent vector.
        """
        return self.vq_vae.encoder(x)

    @torch.no_grad()
    def quantize(self, z: Tensor) -> Tensor:
        """Quantize the latent vector into a token.

        Args:
            z (torch.Tensor): The latent vector.

        Returns:
            indices (torch.Tensor): The sequence of the codebook's indices.
        """
        _, _, indices = self.vq_vae.quantizer(z)
        return indices

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Implement the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input sequence.
        
        Returns:
            Tensor (Tuple[torch.Tensor, torch.Tensor]): The output sequence.
        """
        z = self.encode(x)
        indices = self.quantize(z)

        # Add the start of sequence token to the input sequence
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(indices.device)

        # Mask random elements in the input sequence
        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(
            indices, self.args.codebook_embeddings)
        new_indices = mask * indices + (1 - mask) * random_indices

        # Concatenate the start of sequence token and the masked input sequence
        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits = self.transformer(new_indices[:, :-1])

        return logits, target

    def training_step(self, batch: Tuple) -> Dict:
        """Implement the training step of the model.
        
        Args:
            batch (Tuple): The batch of data.
        
        Returns:
            Dict: The training loss.
        """
        imgs, _ = batch
        # forward pass
        logits, targets = self(imgs)
        # cross-entropy loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        return {"loss": loss}

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """Implement the validation step of the model.

        Args:
            batch (Tuple): The batch of data.
            batch_idx (int): The batch index.
        
        Returns:
            Dict: The validation loss.
        """
        imgs, _ = batch
        # forward pass
        logits, targets = self(imgs)
        # cross-entropy loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))

        # log the validation loss for checkpointing
        self.log("val_loss", loss, prog_bar=True)

        return {"val_loss": loss}

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

        parser.add_argument("--pkeep", type=float, default=0.9,
                            help="The probability of keeping a token in the input sequence.")
        parser.add_argument("--sos_token", type=int, default=0,
                            help="The start of sequence token.")
        parser.add_argument("--vqvae_ckpt", type=str, required=True,
                            help="The path to the VQ-VAE checkpoint.")

        return parser
