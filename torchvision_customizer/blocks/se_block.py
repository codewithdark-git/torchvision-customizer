"""Squeeze-and-Excitation (SE) block for channel attention.

Implements the SE block from "Squeeze-and-Excitation Networks"
(https://arxiv.org/abs/1709.01507). Uses channel-wise attention to
recalibrate feature responses adaptively.

Example:
    >>> from torchvision_customizer.blocks import SEBlock
    >>> se_block = SEBlock(channels=64, reduction=16)
    >>> output = se_block(x)
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import get_activation


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Implements the SE block that uses channel-wise attention to recalibrate
    feature responses adaptively. Effective for improving model performance
    with minimal computational overhead.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the bottleneck. Default is 16.
        activation: Activation function. Default is 'relu'.

    Example:
        >>> se_block = SEBlock(channels=64, reduction=16)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = se_block(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation: str = 'relu',
    ) -> None:
        """Initialize SEBlock.

        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the bottleneck layer.
            activation: Activation function name.

        Raises:
            ValueError: If channels or reduction is not positive.
        """
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")

        self.channels = channels
        self.reduction = reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=True),
            get_activation(activation),
            nn.Linear(max(1, channels // reduction), channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE block.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W) with same shape as input.
        """
        batch, channels, _, _ = x.size()

        # Squeeze: global average pooling
        squeeze = self.squeeze(x).view(batch, channels)

        # Excitation: FC layers with sigmoid
        excitation = self.excitation(squeeze).view(batch, channels, 1, 1)

        # Scale: multiply with input
        return x * excitation
