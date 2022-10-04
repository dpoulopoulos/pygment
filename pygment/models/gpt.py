from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl

from pygment.modules import PositionalEncoder, TransformerBlock


class MinGPT(pl.LightningModule):
    def __init__(self, args: ArgumentParser):
        super().__init__()

        self.embeddings = nn.Embedding(args.n_embedding, args.embedding_dim)
        self.pos_encoder = PositionalEncoder(256, args.embedding_dim)
        blocks = [TransformerBlock(args.embedding_dims, args.n_heads)
                 for _ in range(args.n_layer)]
        self.transformer_blocks = nn.Sequential(*blocks)

        self.ln_f = nn.LayerNorm(args.embedding_dim)
        self.fc = nn.Linear(args.embedding_dims, args.n_embedding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.pos_encoder(x)
        x = self.transformer_blocks(x)
        x = self.ln_f(x)
        return self.fc(x)
