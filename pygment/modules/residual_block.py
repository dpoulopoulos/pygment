from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class ResidualBlock(nn.Module):
    """Define a residual block module.
    
    The residual block is a foundational building block of ResNet architecture.
    Its main purpose is to help with the degradation issue in Deep Neural
    Networks.

    The residual block takes an input tensor, pass it through two convolutional
    layers and, finally, add the input tensor to the output of the second
    convolutional layer.

    Citation:
        He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
          for image recognition. In Proceedings of the IEEE conference on
          computer vision and pattern recognition (pp. 770-778).

    Args:
        input_channels (int): Number of channels in the input tensor.
        output_channels (int): Number of channels produced by the convolution.
    
    Returns:
        Tensor (torch.Tensor): The output tensor.

    Example:
        >>> import torch
        >>> from pygment.modules import ResidualBlock
        >>> x = torch.randn(1, 3, 32, 32)
        >>> residual_block = ResidualBlock(3, 64)
        >>> out = residual_block(x)
        >>> out.shape
        torch.Size([1, 3, 32, 32])
    """
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, input_channels,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Implement the forward pass of the residual block.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            Tensor (torch.Tensor): The output tensor.
        """
        out = self.conv(x) + x
        return F.relu(out)
