"""ResNet Template: Parametric ResNet implementation.

Implements ResNet variants based on the layers parameter:
- layers=18: ResNet-18 (BasicBlock)
- layers=34: ResNet-34 (BasicBlock)
- layers=50: ResNet-50 (Bottleneck)
- layers=101: ResNet-101 (Bottleneck)
- layers=152: ResNet-152 (Bottleneck)

All implementations are from scratch using the package's building blocks.
"""

from typing import Any, Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn

from torchvision_customizer.templates.base import Template
from torchvision_customizer.compose import Stem, Stage, Head, VisionModel
from torchvision_customizer.blocks import (
    ResidualBlock,
    StandardBottleneck,
    SEBlock,
)
from torchvision_customizer.layers import get_activation, get_normalization


# ResNet configurations: layers -> (block_counts, use_bottleneck)
RESNET_CONFIGS = {
    18: ([2, 2, 2, 2], False),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
}


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34.
    
    Structure: conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (+skip) -> ReLU
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = get_normalization(norm_type, out_channels)
        self.act1 = get_activation(activation)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = get_normalization(norm_type, out_channels)
        self.act2 = get_activation(activation)
        
        # SE block
        self.se = SEBlock(out_channels, reduction=se_reduction) if use_se else None
        
        # Downsample path
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                get_normalization(norm_type, out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se is not None:
            out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.act2(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152.
    
    Structure: conv1x1 -> BN -> ReLU -> conv3x3 -> BN -> ReLU -> conv1x1 -> BN -> (+skip) -> ReLU
    """
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()
        
        mid_channels = out_channels // self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = get_normalization(norm_type, mid_channels)
        self.act1 = get_activation(activation)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = get_normalization(norm_type, mid_channels)
        self.act2 = get_activation(activation)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = get_normalization(norm_type, out_channels)
        self.act3 = get_activation(activation)
        
        # SE block
        self.se = SEBlock(out_channels, reduction=se_reduction) if use_se else None
        
        # Downsample path
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                get_normalization(norm_type, out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.se is not None:
            out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.act3(out)
        
        return out


class ResNetModel(nn.Module):
    """Complete ResNet model built from blocks."""
    
    def __init__(
        self,
        layers: int,
        num_classes: int = 1000,
        in_channels: int = 3,
        activation: str = 'relu',
        norm_type: str = 'batch',
        use_se: bool = False,
        se_reduction: int = 16,
        stem_channels: int = 64,
        channel_scale: float = 1.0,
        dropout: float = 0.0,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        
        if layers not in RESNET_CONFIGS:
            raise ValueError(f"Unsupported layers={layers}. Choose from {list(RESNET_CONFIGS.keys())}")
        
        block_counts, use_bottleneck = RESNET_CONFIGS[layers]
        block_cls = Bottleneck if use_bottleneck else BasicBlock
        expansion = block_cls.expansion
        
        # Apply channel scaling
        def scale(c): return int(c * channel_scale)
        
        self.layers = layers
        self.in_channels = scale(stem_channels)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, 7, stride=2, padding=3, bias=False),
            get_normalization(norm_type, self.in_channels),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Stages
        stage_channels = [scale(64), scale(128), scale(256), scale(512)]
        
        self.stage1 = self._make_stage(
            block_cls, stage_channels[0] * expansion, block_counts[0], stride=1,
            activation=activation, norm_type=norm_type, use_se=use_se, se_reduction=se_reduction
        )
        self.stage2 = self._make_stage(
            block_cls, stage_channels[1] * expansion, block_counts[1], stride=2,
            activation=activation, norm_type=norm_type, use_se=use_se, se_reduction=se_reduction
        )
        self.stage3 = self._make_stage(
            block_cls, stage_channels[2] * expansion, block_counts[2], stride=2,
            activation=activation, norm_type=norm_type, use_se=use_se, se_reduction=se_reduction
        )
        self.stage4 = self._make_stage(
            block_cls, stage_channels[3] * expansion, block_counts[3], stride=2,
            activation=activation, norm_type=norm_type, use_se=use_se, se_reduction=se_reduction
        )
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(stage_channels[3] * expansion, num_classes)
        
        # Weight initialization
        self._init_weights(zero_init_residual)
    
    def _make_stage(
        self,
        block_cls: Type[nn.Module],
        out_channels: int,
        num_blocks: int,
        stride: int,
        **kwargs,
    ) -> nn.Sequential:
        """Create a stage with multiple blocks."""
        blocks = []
        
        # First block may downsample
        blocks.append(block_cls(
            self.in_channels, out_channels, stride=stride, **kwargs
        ))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            blocks.append(block_cls(
                self.in_channels, out_channels, stride=1, **kwargs
            ))
        
        return nn.Sequential(*blocks)
    
    def _init_weights(self, zero_init_residual: bool):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get features from each stage."""
        features = []
        x = self.stem(x)
        features.append(x)
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.stage4(x)
        features.append(x)
        return features
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def explain(self) -> str:
        """Human-readable model description."""
        block_counts, use_bottleneck = RESNET_CONFIGS[self.layers]
        block_type = "Bottleneck" if use_bottleneck else "BasicBlock"
        
        lines = [
            "+" + "-" * 60 + "+",
            "|" + f" ResNet-{self.layers} ".center(60, "-") + "|",
            "+" + "-" * 60 + "+",
            f"| Stem: Conv(7x7, s2) -> BN -> ReLU -> MaxPool(3x3, s2)".ljust(61) + "|",
            "|" + " " * 60 + "|",
        ]
        
        for i, count in enumerate(block_counts):
            channels = [64, 128, 256, 512][i]
            if use_bottleneck:
                channels *= 4
            stride = "v" if i > 0 else " "
            lines.append(f"| Stage {i+1}: {count}x {block_type}({channels}ch) {stride}".ljust(61) + "|")
        
        lines.extend([
            "|" + " " * 60 + "|",
            f"| Head: AdaptiveAvgPool -> Flatten -> Linear".ljust(61) + "|",
            "+" + "-" * 60 + "+",
            f"| Parameters: {self.count_parameters():,}".ljust(61) + "|",
            "+" + "-" * 60 + "+",
        ])
        
        return "\n".join(lines)


@Template.register('resnet')
class ResNetTemplate(Template):
    """ResNet architecture template.
    
    Example:
        >>> template = ResNetTemplate(layers=50)
        >>> template.add_attention('se')
        >>> model = template.build(num_classes=100)
    """
    
    def __init__(self, layers: int = 50, **kwargs):
        if layers not in RESNET_CONFIGS:
            raise ValueError(f"layers must be one of {list(RESNET_CONFIGS.keys())}, got {layers}")
        
        config = {
            'name': f'ResNet-{layers}',
            'layers': layers,
            'activation': kwargs.get('activation', 'relu'),
            'norm_type': kwargs.get('norm_type', 'batch'),
            'stem_channels': kwargs.get('stem_channels', 64),
            'use_se': kwargs.get('use_se', False),
            'channel_scale': kwargs.get('channel_scale', 1.0),
            'dropout': kwargs.get('dropout', 0.0),
        }
        super().__init__(config)
    
    def build(self, num_classes: int = 1000, **kwargs) -> nn.Module:
        """Build ResNet model from template configuration."""
        config = self.get_config()
        
        # Apply attention modification
        use_se = config.get('use_se', False)
        if 'attention' in config and config['attention'].get('type') == 'se':
            use_se = True
        
        return ResNetModel(
            layers=config['layers'],
            num_classes=num_classes,
            activation=config.get('activation', 'relu'),
            norm_type=config.get('norm_type', 'batch'),
            use_se=use_se,
            stem_channels=config.get('stem_channels', 64),
            channel_scale=config.get('channel_scale', 1.0),
            dropout=config.get('dropout', 0.0),
            **kwargs,
        )


def resnet(
    layers: int = 50,
    num_classes: int = 1000,
    **kwargs,
) -> nn.Module:
    """Create a ResNet model.
    
    Parametric factory function for ResNet architectures.
    
    Args:
        layers: Number of layers (18, 34, 50, 101, 152)
        num_classes: Number of output classes
        **kwargs: Additional configuration (activation, norm_type, use_se, etc.)
        
    Returns:
        ResNet model
        
    Example:
        >>> model = resnet(layers=18, num_classes=10)
        >>> model = resnet(layers=50, num_classes=1000, use_se=True)
        >>> model = resnet(layers=101, activation='gelu')
    """
    return ResNetModel(layers=layers, num_classes=num_classes, **kwargs)
