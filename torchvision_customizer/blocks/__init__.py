"""Blocks module: Pre-built neural network blocks.

This module provides reusable building blocks for constructing CNN architectures,
including convolutional blocks, residual blocks, and specialized architectures.
"""

from torchvision_customizer.blocks.conv_block import ConvBlock

__all__ = [
    "ConvBlock",
]
