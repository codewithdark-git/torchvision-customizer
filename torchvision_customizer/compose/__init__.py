"""Model Composer: Intuitive model composition with operators.

The compose module provides a fluent API for building neural networks
using Stem, Stage, and Head components with operator overloading.

Example:
    >>> from torchvision_customizer.compose import Stem, Stage, Head
    
    >>> model = (
    ...     Stem(channels=64, kernel=7, stride=2)
    ...     >> Stage(channels=64, blocks=2, pattern='residual')
    ...     >> Stage(channels=128, blocks=2, pattern='residual', downsample=True)
    ...     >> Head(num_classes=1000)
    ... )
"""

from torchvision_customizer.compose.stem import Stem, SimpleStem, PatchStem
from torchvision_customizer.compose.stage import Stage, TransitionLayer
from torchvision_customizer.compose.head import Head, SegmentationHead, DetectionHead
from torchvision_customizer.compose.composer import (
    Compose,
    Sequential,
    VisionModel,
    build_vision_model,
)
from torchvision_customizer.compose.operators import (
    ComposableModule,
    ComposedModule,
    BranchedModule,
    compose,
    repeat,
    branch,
    residual,
)

__all__ = [
    # Stem
    'Stem',
    'SimpleStem',
    'PatchStem',
    
    # Stage
    'Stage',
    'TransitionLayer',
    
    # Head
    'Head',
    'SegmentationHead',
    'DetectionHead',
    
    # Composer
    'Compose',
    'Sequential',
    'VisionModel',
    'build_vision_model',
    
    # Operators
    'ComposableModule',
    'ComposedModule',
    'BranchedModule',
    'compose',
    'repeat',
    'branch',
    'residual',
]
