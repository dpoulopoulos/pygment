from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl

from pygment.modules import PositionalEncoder, TransformerBlock


class MinGPT(pl.LightningModule):
    """Implement a MinGPt model.
    
    The class implements the MinGPT model. The model takes in an input sequence
    of indices and outputs a probability distribution over the next token in
    the sequence.

    The model is based on OpenAI's GPT-2 model and Andrej Karpathy's minGPT
    implementation.

    Citation:
        Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I.
          (2019). Language models are unsupervised multitask learners. OpenAI
          blog, 1(8), 9.

        Karpathy, A. (2020). minGPT: Minimal GPT in 50 lines of PyTorch.
          GitHub. https://github.com/karpathy/minGPT

    Args:
        n_embedding (int): The number of unique tokens in the vocabulary.
        embedding_dims (int): The number of embedding dimensions.
        n_heads (int): The number of heads in the multi-head attention layer.
            Note that embed_dim will be split across num_heads (i.e. each head
            will have dimension embed_dim // num_heads).
        n_layer (int): The number of transformer blocks.
        transformer_dropout (float): Dropout probability on
            ``attn_output_weights``.
        max_seq_len (int): The maximum sequence length.

    Attributes:
        embeddings (nn.Embedding): The embedding layer.
        pos_encoder (PositionalEncoder): The positional encoder.
        transformer_blocks (nn.Sequential): The transformer blocks.
        ln_f (nn.LayerNorm): The final layer normalization layer.
        fc (nn.Linear): The final linear layer.

    Example:
        >>> from argparse import ArgumentParser
        >>> import torch
        >>> from pygment.models import MinGPT
        >>> x = torch.randint(0, 100, (1, 10))
        >>> args = ArgumentParser()
        >>> args.n_embedding = 64
        >>> args.embedding_dims = 256
        >>> args.n_heads = 8
        >>> args.n_layer = 6
        >>> args.transformer_dropout = 0.1
        >>> args.max_seq_len = 10
        >>> model = MinGPT(args)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 10, 64])
    """
    def __init__(self, args: ArgumentParser):
        super().__init__()

        self.embeddings = nn.Embedding(args.n_embedding, args.embedding_dims)
        self.pos_encoder = PositionalEncoder(args.max_seq_len,
                                             args.embedding_dims)

        device = torch.device("cuda" if args.accelerator == "gpu" else "cpu")

        blocks = [TransformerBlock(args.embedding_dims, args.n_heads,
                                   args.max_seq_len, device,
                                   args.transformer_dropout)
                 for _ in range(args.n_layer)]
        self.transformer_blocks = nn.Sequential(*blocks)

        self.ln_f = nn.LayerNorm(args.embedding_dims)
        self.fc = nn.Linear(args.embedding_dims, args.n_embedding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.pos_encoder(x)
        x = self.transformer_blocks(x)
        x = self.ln_f(x)
        return self.fc(x)

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> ArgumentParser:
        """Add the model arguments to the parser.

        Args:
            parser (ArgumentParser): The parser to add the arguments to.
        
        Returns:
            ArgumentParser: The parser with the added arguments.
        """
        parser = ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument("--n_embedding", type=int, default=256,
                            help="The number of unique tokens in the"
                                 " vocabulary.")
        parser.add_argument("--embedding_dims", type=int, default=256,
                            help="The number of embedding dimensions.")
        parser.add_argument("--n_heads", type=int, default=8,
                            help="The number of heads in the multi-head"
                                 " attention layer. Note that embed_dim will"
                                 " be split across num_heads (i.e. each head"
                                 " will have dimension embed_dim //"
                                 " num_heads).")
        parser.add_argument("--n_layer", type=int, default=6,
                            help="The number of transformer blocks.")
        parser.add_argument("--transformer_dropout", type=float, default=0.1,
                            help="Dropout probability on attn_output_weights.")
        parser.add_argument("--max_seq_len", type=int, required=True,
                            help="The maximum sequence length.")

        return parser
