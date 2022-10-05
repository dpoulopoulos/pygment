from argparse import ArgumentParser

import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102


class FlowersData(pl.LightningDataModule):
    """Return examples from the Oxford 102 Flower Dataset..

    Oxford 102 Flower is an image classification dataset consisting of 102
    flower categories. The flowers were chosen to be flowers commonly occurring
    in the United Kingdom. Each class consists of between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there
    are categories that have large variations within the category, and several
    very similar categories.
    
    Citation:
        Nilsback, M. E., & Zisserman, A. (2008, December). Automated flower
          classification over a large number of classes. In 2008 Sixth Indian
          Conference on Computer Vision, Graphics & Image Processing
          (pp. 722-729). IEEE.
    
    Args:
        download_path (str): The path to download the dataset to.
        img_size (int): The size of the image to resize to.
        batch_size (int): The batch size to use.
        num_workers (int): The number of workers to use in the DataLoader.
    
    Example:
        >>> from pygment.dataset import FlowersData
        >>> from argparse import ArgumentParser
        >>> args = ArgumentParser()
        >>> args.download_path = "data"
        >>> args.img_size = 224
        >>> args.batch_size = 32
        >>> args.num_workers = 4
        >>> dataset = FlowersData(args)
        >>> train_dataloader = dataset.train_dataloader()
        >>> val_dataloader = dataset.val_dataloader()
    """

    def __init__(self, args: ArgumentParser):
        super().__init__()
        self.download_path = args.download_path
        self.img_size = args.img_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def train_dataloader(self):
        """Return a DataLoader for the training set."""
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        
        train_dataset = Flowers102(root=self.download_path, split="train",
                                   transform=transform, download=True)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
            shuffle=True)

        return train_dataloader

    def val_dataloader(self):
        """Return a DataLoader for the validation set."""
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        
        val_dataset = Flowers102(root=self.download_path, split="test",
                                 transform=transform, download=True)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False)

        return val_dataloader

    def test_dataloader(self):
        """Return a DataLoader for the test set."""
        return self.val_dataloader()

    @staticmethod
    def add_data_args(parser: ArgumentParser) -> ArgumentParser:
        """Add data arguments to the parser.
        
        Args:
            parser (ArgumentParser): The parser to add the arguments to.
        
        Returns:
            ArgumentParser: The parser with the added arguments.
        """
        parser = ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument("--download_path", type=str, default="data",
                            help="The path to download the dataset to.")
        parser.add_argument("--img_size", type=int, default=64,
                            help="The size of the images to use.")
        parser.add_argument("--batch_size", type=int, default=64,
                            help="The batch size to use.")
        parser.add_argument("--num_workers", type=int, default=0,
                            help="The number of workers to use in"
                                 " the DataLoader.")

        return parser
