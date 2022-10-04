from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from .gpt import MinGPT
from .vq_vae import VQVAE


class VQTransformer(pl.LightningModule):
    def __init__(self, args: ArgumentParser):
        super().__init__()

        self.pkeep = args.pkeep
        self.sos_token = args.sos_token

        self.transformer = MinGPT(args)
        self.vq_vae = VQVAE.load_from_checkpoint(args.vqvae_ckpt, args=args)

    @torch.no_grad()
    def encode(self, x):
        return self.vq_vae.encoder(x)

    @torch.no_grad()
    def quantize(self, z):
        _, _, indices = self.vq_vae.quantizer(z)
        return indices

    def forward(self, x):
        z = self.encode(x)
        indices = self.quantize(z)

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(indices.device)

        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(
            indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target