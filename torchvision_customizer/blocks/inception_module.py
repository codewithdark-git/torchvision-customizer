"""Inception module with multiple parallel branches.

Implements the Inception module architecture with multiple parallel
convolutional branches at different kernel sizes.

Example:
    >>> from torchvision_customizer.blocks import InceptionModule
    >>> inception = InceptionModule(
    ...     in_channels=192, out_1x1=64, out_3x3=96, out_5x5=16
    ... )
    >>> output = inception(x)
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import get_activation, get_normalization


class InceptionModule(nn.Module):
    """Inception module with multiple parallel branches.

    Implements the Inception architecture with four parallel branches:
    1. 1x1 convolution
    2. 1x1 followed by 3x3 convolution
    3. 1x1 followed by 5x5 convolution
    4. 3x3 max pool followed by 1x1 convolution

    All branches are concatenated along the channel dimension.

    Args:
        in_channels: Number of input channels.
        out_1x1: Output channels for 1x1 branch.
        out_3x3: Output channels for 3x3 branch.
        out_5x5: Output channels for 5x5 branch.
        out_pool: Output channels for pool branch. Default is None (auto).
        activation: Activation function. Default is 'relu'.
        norm_type: Normalization type. Default is 'batch'.

    Example:
        >>> # Basic inception module
        >>> inception = InceptionModule(
        ...     in_channels=192,
        ...     out_1x1=64,
        ...     out_3x3=96,
        ...     out_5x5=16,
        ...     out_pool=32
        ... )
        >>> x = torch.randn(2, 192, 28, 28)
        >>> output = inception(x)
        >>> print(output.shape)
        torch.Size([2, 208, 28, 28])

        >>> # Different configuration
        >>> inception = InceptionModule(
        ...     in_channels=256,
        ...     out_1x1=128,
        ...     out_3x3=128,
        ...     out_5x5=32,
        ...     out_pool=64
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        out_3x3: int,
        out_5x5: int,
        out_pool: int | None = None,
        activation: str = 'relu',
        norm_type: str = 'batch',
    ) -> None:
        """Initialize InceptionModule."""
        super().__init__()

        self.activation_fn = get_activation(activation)
        
        if out_pool is None:
            out_pool = out_1x1 // 2

        # 1x1 branch
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1, bias=False),
            get_normalization(norm_type, out_1x1),
        )

        # 3x3 branch
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3 // 2, kernel_size=1, bias=False),
            get_normalization(norm_type, out_3x3 // 2),
            nn.Conv2d(out_3x3 // 2, out_3x3, kernel_size=3, padding=1, bias=False),
            get_normalization(norm_type, out_3x3),
        )

        # 5x5 branch (using 3x3 with dilation or two 3x3)
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=1, bias=False),
            get_normalization(norm_type, out_5x5),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=5, padding=2, bias=False),
            get_normalization(norm_type, out_5x5),
        )

        # Pool branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1, bias=False),
            get_normalization(norm_type, out_pool),
        )

        # Output channels
        self.out_channels = out_1x1 + out_3x3 + out_5x5 + out_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inception module.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Concatenated output from all branches of shape (B, C_out, H, W).
        """
        branch_1x1 = self.branch_1x1(x)
        branch_1x1 = self.activation_fn(branch_1x1)

        branch_3x3 = self.branch_3x3(x)
        branch_3x3 = self.activation_fn(branch_3x3)

        branch_5x5 = self.branch_5x5(x)
        branch_5x5 = self.activation_fn(branch_5x5)

        branch_pool = self.branch_pool(x)
        branch_pool = self.activation_fn(branch_pool)

        # Concatenate all branches
        output = torch.cat([branch_1x1, branch_3x3, branch_5x5, branch_pool], dim=1)

        return output
