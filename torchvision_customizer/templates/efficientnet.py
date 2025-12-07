"""EfficientNet Template: Parametric EfficientNet implementation.

Implements EfficientNet variants based on compound scaling:
- variant='b0': EfficientNet-B0 (baseline)
- variant='b1' to 'b7': Scaled variants

All implementations are from scratch using the package's building blocks.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision_customizer.templates.base import Template
from torchvision_customizer.layers import get_activation, get_normalization
from torchvision_customizer.blocks import SEBlock


# EfficientNet-B0 base configuration
# (expand_ratio, channels, num_layers, stride, kernel_size)
EFFICIENTNET_BASE_CONFIG = [
    (1, 16, 1, 1, 3),    # Stage 1
    (6, 24, 2, 2, 3),    # Stage 2
    (6, 40, 2, 2, 5),    # Stage 3
    (6, 80, 3, 2, 3),    # Stage 4
    (6, 112, 3, 1, 5),   # Stage 5
    (6, 192, 4, 2, 5),   # Stage 6
    (6, 320, 1, 1, 3),   # Stage 7
]

# Scaling coefficients: (width_mult, depth_mult, resolution, dropout)
EFFICIENTNET_SCALING = {
    'b0': (1.0, 1.0, 224, 0.2),
    'b1': (1.0, 1.1, 240, 0.2),
    'b2': (1.1, 1.2, 260, 0.3),
    'b3': (1.2, 1.4, 300, 0.3),
    'b4': (1.4, 1.8, 380, 0.4),
    'b5': (1.6, 2.2, 456, 0.4),
    'b6': (1.8, 2.6, 528, 0.5),
    'b7': (2.0, 3.1, 600, 0.5),
}


class Swish(nn.Module):
    """Swish activation (x * sigmoid(x))."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) for EfficientNet."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: float = 1.0,
        se_ratio: float = 0.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        mid_channels = int(in_channels * expand_ratio)
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                Swish(),
            ])
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, 
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            Swish(),
        ])
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        layers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, se_channels, 1),
            Swish(),
            nn.Conv2d(se_channels, mid_channels, 1),
            nn.Sigmoid(),
        ))
        
        self.se_idx = len(layers) - 1
        
        # Projection phase
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        for i, layer in enumerate(self.layers):
            if i == self.se_idx:
                # SE block applies multiplicatively
                se = layer(x)
                x = x * se
            else:
                x = layer(x)
        
        if self.use_residual:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity
        
        return x


class EfficientNetModel(nn.Module):
    """Complete EfficientNet model."""
    
    def __init__(
        self,
        variant: str = 'b0',
        num_classes: int = 1000,
        in_channels: int = 3,
    ):
        super().__init__()
        
        if variant not in EFFICIENTNET_SCALING:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(EFFICIENTNET_SCALING.keys())}")
        
        self.variant = variant
        width_mult, depth_mult, resolution, dropout = EFFICIENTNET_SCALING[variant]
        
        def round_channels(c): 
            return int(math.ceil(c * width_mult / 8) * 8)
        
        def round_layers(n):
            return int(math.ceil(n * depth_mult))
        
        # Stem
        stem_channels = round_channels(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            Swish(),
        )
        
        # MBConv blocks
        stages = []
        in_ch = stem_channels
        
        for expand, channels, num_layers, stride, kernel in EFFICIENTNET_BASE_CONFIG:
            out_ch = round_channels(channels)
            layers_count = round_layers(num_layers)
            
            for i in range(layers_count):
                s = stride if i == 0 else 1
                stages.append(MBConv(
                    in_ch, out_ch, kernel, s, expand_ratio=expand, dropout=dropout * 0.2
                ))
                in_ch = out_ch
        
        self.stages = nn.Sequential(*stages)
        
        # Head
        head_channels = round_channels(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_channels, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def explain(self) -> str:
        width_mult, depth_mult, resolution, dropout = EFFICIENTNET_SCALING[self.variant]
        
        lines = [
            "+" + "-" * 60 + "+",
            "|" + f" EfficientNet-{self.variant.upper()} ".center(60, "-") + "|",
            "+" + "-" * 60 + "+",
            f"| Width multiplier: {width_mult}x".ljust(61) + "|",
            f"| Depth multiplier: {depth_mult}x".ljust(61) + "|",
            f"| Input resolution: {resolution}x{resolution}".ljust(61) + "|",
            f"| Dropout: {dropout}".ljust(61) + "|",
            "|" + " " * 60 + "|",
            f"| Architecture: Stem -> 7 MBConv stages -> Head".ljust(61) + "|",
            "+" + "-" * 60 + "+",
            f"| Parameters: {self.count_parameters():,}".ljust(61) + "|",
            "+" + "-" * 60 + "+",
        ]
        
        return "\n".join(lines)


@Template.register('efficientnet')
class EfficientNetTemplate(Template):
    """EfficientNet architecture template."""
    
    def __init__(self, variant: str = 'b0', **kwargs):
        if variant not in EFFICIENTNET_SCALING:
            raise ValueError(f"variant must be one of {list(EFFICIENTNET_SCALING.keys())}, got {variant}")
        
        config = {
            'name': f'EfficientNet-{variant.upper()}',
            'variant': variant,
        }
        super().__init__(config)
    
    def build(self, num_classes: int = 1000, **kwargs) -> nn.Module:
        config = self.get_config()
        
        return EfficientNetModel(
            variant=config['variant'],
            num_classes=num_classes,
            **kwargs,
        )


def efficientnet(
    variant: str = 'b0',
    num_classes: int = 1000,
    **kwargs,
) -> nn.Module:
    """Create an EfficientNet model.
    
    Args:
        variant: Model variant ('b0' to 'b7')
        num_classes: Number of output classes
        **kwargs: Additional configuration
        
    Returns:
        EfficientNet model
        
    Example:
        >>> model = efficientnet(variant='b0', num_classes=10)
        >>> model = efficientnet(variant='b4', num_classes=1000)
    """
    return EfficientNetModel(variant=variant, num_classes=num_classes, **kwargs)
