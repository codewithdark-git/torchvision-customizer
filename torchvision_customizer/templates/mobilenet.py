"""MobileNet Template: Parametric MobileNet implementation.

Implements MobileNet variants based on the version parameter:
- version=1: MobileNetV1 (depthwise separable convolutions)
- version=2: MobileNetV2 (inverted residuals with linear bottleneck)
- version=3: MobileNetV3 (SE blocks + hard-swish activation)

All implementations are from scratch using the package's building blocks.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision_customizer.templates.base import Template
from torchvision_customizer.layers import get_activation, get_normalization
from torchvision_customizer.blocks import SEBlock


class HardSwish(nn.Module):
    """Hard-Swish activation for MobileNetV3."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu(x + 3) / 6


class HardSigmoid(nn.Module):
    """Hard-Sigmoid activation for MobileNetV3."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + 3) / 6


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for MobileNetV1."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = 'relu',
    ):
        super().__init__()
        
        # Depthwise
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            get_activation(activation),
        )
        
        # Pointwise
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2/V3."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 6.0,
        use_se: bool = False,
        se_ratio: float = 0.25,
        activation: str = 'relu',
    ):
        super().__init__()
        
        mid_channels = int(in_channels * expand_ratio)
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        layers = []
        
        # Expansion (if expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                get_activation(activation),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            get_activation(activation),
        ])
        
        # SE block
        if use_se:
            layers.append(SEBlock(mid_channels, reduction=int(1/se_ratio)))
        
        # Projection (linear, no activation)
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


# MobileNetV2 configuration: (expand_ratio, out_channels, num_blocks, stride)
MOBILENETV2_CONFIG = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]

# MobileNetV3-Small configuration: (expand, out, se, activation, stride)
MOBILENETV3_SMALL_CONFIG = [
    (1, 16, True, 'relu', 2),
    (72/16, 24, False, 'relu', 2),
    (88/24, 24, False, 'relu', 1),
    (4, 40, True, 'hswish', 2),
    (6, 40, True, 'hswish', 1),
    (6, 40, True, 'hswish', 1),
    (3, 48, True, 'hswish', 1),
    (3, 48, True, 'hswish', 1),
    (6, 96, True, 'hswish', 2),
    (6, 96, True, 'hswish', 1),
    (6, 96, True, 'hswish', 1),
]


class MobileNetV1Model(nn.Module):
    """MobileNetV1 implementation."""
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_multiplier: float = 1.0,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        def scale(c): return int(c * width_multiplier)
        
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, scale(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(scale(32)),
            get_activation(activation),
            
            # Depthwise separable blocks
            DepthwiseSeparableConv(scale(32), scale(64), stride=1, activation=activation),
            DepthwiseSeparableConv(scale(64), scale(128), stride=2, activation=activation),
            DepthwiseSeparableConv(scale(128), scale(128), stride=1, activation=activation),
            DepthwiseSeparableConv(scale(128), scale(256), stride=2, activation=activation),
            DepthwiseSeparableConv(scale(256), scale(256), stride=1, activation=activation),
            DepthwiseSeparableConv(scale(256), scale(512), stride=2, activation=activation),
            # 5x repeated
            DepthwiseSeparableConv(scale(512), scale(512), stride=1, activation=activation),
            DepthwiseSeparableConv(scale(512), scale(512), stride=1, activation=activation),
            DepthwiseSeparableConv(scale(512), scale(512), stride=1, activation=activation),
            DepthwiseSeparableConv(scale(512), scale(512), stride=1, activation=activation),
            DepthwiseSeparableConv(scale(512), scale(512), stride=1, activation=activation),
            # Final blocks
            DepthwiseSeparableConv(scale(512), scale(1024), stride=2, activation=activation),
            DepthwiseSeparableConv(scale(1024), scale(1024), stride=1, activation=activation),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(scale(1024), num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MobileNetV2Model(nn.Module):
    """MobileNetV2 implementation with inverted residuals."""
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_multiplier: float = 1.0,
        activation: str = 'relu',
        dropout: float = 0.2,
    ):
        super().__init__()
        
        def scale(c): return max(1, int(c * width_multiplier))
        
        # Initial conv
        features = [
            nn.Conv2d(3, scale(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(scale(32)),
            get_activation(activation),
        ]
        
        # Inverted residual blocks
        in_channels = scale(32)
        for expand_ratio, out_channels, num_blocks, stride in MOBILENETV2_CONFIG:
            out_channels = scale(out_channels)
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                features.append(InvertedResidual(
                    in_channels, out_channels, s, expand_ratio, activation=activation
                ))
                in_channels = out_channels
        
        # Final conv
        last_channels = scale(1280)
        features.extend([
            nn.Conv2d(in_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            get_activation(activation),
        ])
        
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(last_channels, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MobileNetV3Model(nn.Module):
    """MobileNetV3 implementation with SE and hard-swish."""
    
    def __init__(
        self,
        num_classes: int = 1000,
        mode: str = 'small',  # 'small' or 'large'
        width_multiplier: float = 1.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        def scale(c): return max(1, int(c * width_multiplier))
        
        # Use hardswish for V3
        def get_act(name):
            if name == 'hswish':
                return HardSwish()
            return get_activation('relu')
        
        # Initial conv
        init_channels = scale(16)
        features = [
            nn.Conv2d(3, init_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            HardSwish(),
        ]
        
        config = MOBILENETV3_SMALL_CONFIG if mode == 'small' else MOBILENETV3_SMALL_CONFIG  # TODO: add large config
        
        in_channels = init_channels
        for expand, out, use_se, act, stride in config:
            out_channels = scale(out)
            features.append(InvertedResidual(
                in_channels, out_channels, stride, 
                expand_ratio=expand, use_se=use_se, activation='relu'
            ))
            in_channels = out_channels
        
        # Final stages
        last_channels = scale(576) if mode == 'small' else scale(960)
        features.extend([
            nn.Conv2d(in_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            HardSwish(),
        ])
        
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with extra FC
        final_channels = scale(1024) if mode == 'small' else scale(1280)
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, final_channels),
            HardSwish(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@Template.register('mobilenet')
class MobileNetTemplate(Template):
    """MobileNet architecture template."""
    
    def __init__(self, version: int = 2, **kwargs):
        if version not in [1, 2, 3]:
            raise ValueError(f"version must be 1, 2, or 3, got {version}")
        
        config = {
            'name': f'MobileNetV{version}',
            'version': version,
            'width_multiplier': kwargs.get('width_multiplier', 1.0),
            'activation': kwargs.get('activation', 'relu' if version == 2 else 'relu'),
            'dropout': kwargs.get('dropout', 0.2),
            'mode': kwargs.get('mode', 'small'),  # For V3
        }
        super().__init__(config)
    
    def build(self, num_classes: int = 1000, **kwargs) -> nn.Module:
        config = self.get_config()
        version = config['version']
        
        if version == 1:
            return MobileNetV1Model(
                num_classes=num_classes,
                width_multiplier=config.get('width_multiplier', 1.0),
                activation=config.get('activation', 'relu'),
                dropout=config.get('dropout', 0.0),
            )
        elif version == 2:
            return MobileNetV2Model(
                num_classes=num_classes,
                width_multiplier=config.get('width_multiplier', 1.0),
                activation=config.get('activation', 'relu'),
                dropout=config.get('dropout', 0.2),
            )
        else:  # V3
            return MobileNetV3Model(
                num_classes=num_classes,
                mode=config.get('mode', 'small'),
                width_multiplier=config.get('width_multiplier', 1.0),
                dropout=config.get('dropout', 0.2),
            )


def mobilenet(
    version: int = 2,
    num_classes: int = 1000,
    **kwargs,
) -> nn.Module:
    """Create a MobileNet model.
    
    Args:
        version: MobileNet version (1, 2, or 3)
        num_classes: Number of output classes
        **kwargs: Additional configuration (width_multiplier, mode, etc.)
        
    Returns:
        MobileNet model
        
    Example:
        >>> model = mobilenet(version=1, num_classes=10)
        >>> model = mobilenet(version=2, width_multiplier=0.5)
        >>> model = mobilenet(version=3, mode='small')
    """
    template = MobileNetTemplate(version=version, **kwargs)
    return template.build(num_classes=num_classes)
