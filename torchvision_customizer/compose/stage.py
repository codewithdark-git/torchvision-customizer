"""Stage builder for network body.

A Stage is a group of blocks with the same spatial resolution,
typically used in architectures like ResNet, VGG, DenseNet, etc.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn

from torchvision_customizer.compose.operators import ComposableModule
from torchvision_customizer.blocks import (
    ConvBlock,
    ResidualBlock,
    SEBlock,
    DepthwiseBlock,
    DenseConnectionBlock,
    StandardBottleneck,
    WideBottleneck,
)
from torchvision_customizer.blocks.advanced_blocks import MBConv, FusedMBConv
from torchvision_customizer.layers import get_activation, get_normalization


# Pattern registry for stage patterns
_PATTERN_REGISTRY = {}


def register_pattern(name: str):
    """Decorator to register a stage pattern builder."""
    def decorator(fn):
        _PATTERN_REGISTRY[name] = fn
        return fn
    return decorator


class Stage(ComposableModule):
    """Network stage builder with pattern support.
    
    Creates a stage (group of blocks) with configurable patterns.
    Supports pattern mixing with '+' syntax.
    
    Args:
        channels: Output channels for this stage
        blocks: Number of blocks in the stage
        pattern: Block pattern. Available patterns:
            - 'conv': Standard convolution blocks
            - 'residual': ResNet-style residual blocks
            - 'bottleneck': Bottleneck residual blocks
            - 'dense': DenseNet-style dense connections
            - 'depthwise': Depthwise separable convolutions
            - 'mbconv': Mobile inverted bottleneck (EfficientNet style)
            - 'fused_mbconv': Fused MBConv (EfficientNetV2 style)
            Combine with '+se' for SE attention (e.g., 'residual+se')
        in_channels: Input channels (auto-detected if None)
        downsample: Whether to downsample at the start of stage
        stride: Stride for downsampling (default 2)
        activation: Activation function
        norm: Normalization type
        expansion: Bottleneck/MBConv expansion factor
        growth_rate: Growth rate for dense blocks
        se_reduction: SE block reduction ratio
        
    Example:
        >>> stage = Stage(channels=128, blocks=3, pattern='residual')
        >>> # Creates 3 residual blocks with 128 channels
        
        >>> stage = Stage(channels=256, blocks=2, pattern='residual+se', downsample=True)
        >>> # Creates 2 residual blocks with SE attention, first block downsamples
        
        >>> stage = Stage(channels=64, blocks=4, pattern='dense', growth_rate=32)
        >>> # Creates DenseNet-style stage with growth_rate=32
        
        >>> stage = Stage(channels=64, blocks=3, pattern='mbconv', expansion=4)
        >>> # Creates 3 MBConv blocks (EfficientNet style)
    """
    
    def __init__(
        self,
        channels: int,
        blocks: int = 2,
        pattern: str = 'residual',
        in_channels: Optional[int] = None,
        downsample: bool = False,
        stride: int = 2,
        activation: str = 'relu',
        norm: str = 'batch',
        expansion: int = 4,
        growth_rate: int = 32,
        se_reduction: int = 16,
        **kwargs,
    ):
        super().__init__()
        
        self._out_channels = channels
        self._in_channels = in_channels
        self._config = {
            'channels': channels,
            'blocks': blocks,
            'pattern': pattern,
            'downsample': downsample,
        }
        
        # Parse pattern (e.g., 'residual+se' -> ['residual', 'se'])
        patterns = [p.strip() for p in pattern.lower().split('+')]
        
        # Build blocks based on pattern
        self.stage = self._build_stage(
            in_channels=in_channels or channels,
            out_channels=channels,
            num_blocks=blocks,
            patterns=patterns,
            downsample=downsample,
            stride=stride,
            activation=activation,
            norm=norm,
            expansion=expansion,
            growth_rate=growth_rate,
            se_reduction=se_reduction,
            **kwargs,
        )
    
    def _build_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        patterns: List[str],
        downsample: bool,
        stride: int,
        activation: str,
        norm: str,
        expansion: int,
        growth_rate: int,
        se_reduction: int,
        **kwargs,
    ) -> nn.Sequential:
        """Build the stage based on patterns."""
        layers = []
        current_channels = in_channels
        
        for i in range(num_blocks):
            # First block may downsample
            block_stride = stride if (i == 0 and downsample) else 1
            needs_projection = (i == 0) and (current_channels != out_channels or downsample)
            
            # Build block based on primary pattern
            primary_pattern = patterns[0]
            
            if primary_pattern == 'conv':
                block = ConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=block_stride,
                    padding=1,
                    activation=activation,
                    use_batchnorm=(norm == 'batch'),
                )
            
            elif primary_pattern == 'residual':
                block = ResidualBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    stride=block_stride,
                    activation=activation,
                    norm_type=norm,
                    use_se='se' in patterns,
                    se_reduction=se_reduction,
                )
            
            elif primary_pattern == 'bottleneck':
                block = StandardBottleneck(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    stride=block_stride,
                    expansion=expansion,
                    activation=activation,
                    norm_type=norm,
                    use_downsample=needs_projection,
                )
            
            elif primary_pattern == 'wide_bottleneck':
                width_mult = kwargs.get('width_multiplier', 2.0)
                block = WideBottleneck(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    width_multiplier=width_mult,
                    stride=block_stride,
                    expansion=expansion,
                    activation=activation,
                    norm_type=norm,
                    use_downsample=needs_projection,
                )
            
            elif primary_pattern == 'depthwise':
                block = DepthwiseBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    stride=block_stride,
                )
            
            elif primary_pattern == 'dense':
                # For dense blocks, we use the specialized implementation
                block = DenseConnectionBlock(
                    num_layers=num_blocks,
                    in_channels=current_channels,
                    growth_rate=growth_rate,
                    activation=activation,
                    use_batchnorm=(norm == 'batch'),
                )
                # Dense block returns different output channels
                out_channels = current_channels + growth_rate * num_blocks
                self._out_channels = out_channels
                layers.append(block)
                break  # Dense block handles all layers internally
            
            elif primary_pattern == 'mbconv':
                # Mobile Inverted Bottleneck (EfficientNet style)
                block = MBConv(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    expansion=expansion,
                    stride=block_stride,
                    kernel_size=kwargs.get('kernel_size', 3),
                    use_se='se' in patterns or kwargs.get('use_se', True),
                    se_reduction=se_reduction,
                    drop_path=kwargs.get('drop_path', 0.0),
                )
            
            elif primary_pattern == 'fused_mbconv':
                # Fused MBConv (EfficientNetV2 style)
                block = FusedMBConv(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    expansion=expansion,
                    stride=block_stride,
                    kernel_size=kwargs.get('kernel_size', 3),
                    use_se='se' in patterns or kwargs.get('use_se', False),
                    drop_path=kwargs.get('drop_path', 0.0),
                )
            
            else:
                raise ValueError(f"Unknown pattern: {primary_pattern}. "
                               f"Available: conv, residual, bottleneck, dense, depthwise, mbconv, fused_mbconv")
            
            # Add SE block if pattern includes 'se' and not already in residual
            if 'se' in patterns and primary_pattern not in ['residual']:
                block = nn.Sequential(block, SEBlock(out_channels, reduction=se_reduction))
            
            layers.append(block)
            current_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)
    
    def __repr__(self) -> str:
        c = self._config
        parts = [f"{c['channels']}ch", f"{c['blocks']}x{c['pattern']}"]
        if c['downsample']:
            parts.append("v")
        return f"Stage({', '.join(parts)})"


class TransitionLayer(ComposableModule):
    """Transition layer for DenseNet-style architectures.
    
    Reduces feature map size and optionally compresses channels.
    
    Args:
        in_channels: Input channels
        compression: Channel compression factor (0.5 = halve channels)
        pool_size: Pooling kernel size
    """
    
    def __init__(
        self,
        in_channels: int,
        compression: float = 0.5,
        pool_size: int = 2,
    ):
        super().__init__()
        
        out_channels = int(in_channels * compression)
        self._out_channels = out_channels
        
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(pool_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)
    
    def __repr__(self) -> str:
        return f"Transition({self._out_channels})"
