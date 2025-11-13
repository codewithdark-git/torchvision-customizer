"""Blocks module: Pre-built neural network blocks.

This module provides reusable building blocks for constructing CNN architectures,
including convolutional blocks, residual blocks, and specialized architectures.

Included Blocks:
    - ConvBlock: Basic convolutional block with activation, normalization, pooling
    - SEBlock: Squeeze-and-Excitation for channel attention
    - ResidualBlock: Skip connections with optional SE blocks
    - DepthwiseBlock: Depthwise separable convolutions
    - InceptionModule: Multi-branch Inception module
    - Conv3DBlock: 3D convolutional block for volumetric data
    - SuperConv2d: Enhanced 2D convolution with flexible options
    - SuperLinear: Enhanced linear layer with activation and dropout

Example:
    >>> from torchvision_customizer.blocks import (
    ...     ConvBlock,
    ...     ResidualBlock,
    ...     SEBlock,
    ...     InceptionModule
    ... )
    >>> block = ConvBlock(in_channels=3, out_channels=64)
    >>> res_block = ResidualBlock(in_channels=64, out_channels=64)
    >>> se_block = SEBlock(channels=64)
    >>> inception = InceptionModule(in_channels=192, out_1x1=64, ...)
"""

from torchvision_customizer.blocks.conv_block import ConvBlock
from torchvision_customizer.blocks.se_block import SEBlock
from torchvision_customizer.blocks.residual_block import ResidualBlock
from torchvision_customizer.blocks.depthwise_block import DepthwiseBlock
from torchvision_customizer.blocks.inception_module import InceptionModule
from torchvision_customizer.blocks.conv3d_block import Conv3DBlock
from torchvision_customizer.blocks.super_conv2d import SuperConv2d
from torchvision_customizer.blocks.super_linear import SuperLinear

__all__ = [
    # Basic blocks
    "ConvBlock",
    
    # Advanced blocks
    "SEBlock",
    "ResidualBlock",
    "DepthwiseBlock",
    "InceptionModule",
    "Conv3DBlock",
    "SuperConv2d",
    "SuperLinear",
]
