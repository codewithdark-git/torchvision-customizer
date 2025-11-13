"""3D convolutional block for volumetric data.

Implements 3D convolution for processing volumetric data like medical images,
videos, or point clouds.

Example:
    >>> from torchvision_customizer.blocks import Conv3DBlock
    >>> block = Conv3DBlock(in_channels=3, out_channels=64)
    >>> output = block(x)
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import get_activation


class Conv3DBlock(nn.Module):
    """3D convolutional block for volumetric data.

    Applies 3D convolution with normalization and activation functions.
    Useful for processing volumetric data, videos, and temporal sequences.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolutional kernel. Default is 3.
        stride: Stride of convolution. Default is 1.
        padding: Padding for convolution. Default is 1.
        activation: Activation function. Default is 'relu'.
        num_layers: Number of stacked 3D convolutions. Default is 1.

    Example:
        >>> # Basic 3D convolution
        >>> block = Conv3DBlock(in_channels=3, out_channels=64)
        >>> x = torch.randn(2, 3, 16, 64, 64)  # Batch, Channels, Depth, Height, Width
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([2, 64, 16, 64, 64])

        >>> # With downsampling
        >>> block = Conv3DBlock(
        ...     in_channels=64, out_channels=128,
        ...     kernel_size=3, stride=2, padding=1
        ... )

        >>> # Multiple stacked layers
        >>> block = Conv3DBlock(
        ...     in_channels=64, out_channels=64,
        ...     kernel_size=3, num_layers=3
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'relu',
        num_layers: int = 1,
    ) -> None:
        """Initialize Conv3DBlock."""
        super().__init__()

        self.activation_fn = get_activation(activation)
        layers = []

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels

            layers.append(
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm3d(out_ch))

            if i < num_layers - 1:
                layers.append(self.activation_fn)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D convolution.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D', H', W').
        """
        out = self.layers(x)
        out = self.activation_fn(out)
        return out

