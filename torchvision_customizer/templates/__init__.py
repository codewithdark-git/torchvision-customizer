"""Architecture Templates: Pre-defined model architectures with customization.

Templates provide starting points based on well-known architectures,
with full customization capabilities.

Example:
    >>> from torchvision_customizer.templates import resnet, vgg
    
    >>> # Simple usage
    >>> model = resnet(layers=50, num_classes=1000)
    
    >>> # With customization
    >>> model = (Template.resnet(layers=50)
    ...     .replace_activation('gelu')
    ...     .add_attention('se')
    ...     .build(num_classes=100))
"""

from torchvision_customizer.templates.base import Template
from torchvision_customizer.templates.resnet import resnet, ResNetTemplate
from torchvision_customizer.templates.vgg import vgg, VGGTemplate
from torchvision_customizer.templates.mobilenet import mobilenet, MobileNetTemplate
from torchvision_customizer.templates.densenet import densenet, DenseNetTemplate
from torchvision_customizer.templates.efficientnet import efficientnet, EfficientNetTemplate

__all__ = [
    # Base
    'Template',
    
    # Factory functions (parametric)
    'resnet',
    'vgg',
    'mobilenet',
    'densenet',
    'efficientnet',
    
    # Template classes
    'ResNetTemplate',
    'VGGTemplate',
    'MobileNetTemplate',
    'DenseNetTemplate',
    'EfficientNetTemplate',
]
