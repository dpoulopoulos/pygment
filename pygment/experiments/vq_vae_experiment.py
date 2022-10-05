"""Create an experiment with a VQ-VAE model and the Stanford Cars dataset.

This script creates an experiment with a VQ-VAE model and the Stanford Cars
dataset. It trains the model on the training dataset and evaluates it
on a separate validation dataset, keeping the training logs and the model
checkpoint using the power of PyTorch Lightning.
"""

import sys

from argparse import ArgumentParser

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

from pygment.datasets import StanfordCarsData
from pygment.models import VQVAE


def main():
    pl.seed_everything(42)

    parser = ArgumentParser()
    # Add training related arguments
    parser = pl.Trainer.add_argparse_args(parser)
    # Add model related arguments
    parser = VQVAE.add_model_args(parser)
    # Add dataset related arguments
    parser = StanfordCarsData.add_data_args(parser)
    # Parse arguments
    args = parser.parse_args()

    # Instantiate the model and the dataset
    data = StanfordCarsData(args)
    model = VQVAE(args)

    # Define the checkpoint callback
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val_loss', mode='min'))

    # Define and fit the trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, data)


if __name__ == "__main__":
    sys.exit(main())
