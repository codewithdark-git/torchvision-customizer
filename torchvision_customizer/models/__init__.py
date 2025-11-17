"""Models module: Pre-built CNN models and architectures.

This module provides ready-to-use CNN models including CustomCNN for quick
prototyping and benchmarking.

Included Models:
    - CustomCNN: Flexible sequential CNN model with configurable architecture

Example:
    >>> from torchvision_customizer.models import CustomCNN
    >>> model = CustomCNN(
    ...     input_shape=(3, 224, 224),
    ...     num_classes=1000,
    ...     num_conv_blocks=5,
    ...     channels='auto'
    ... )
    >>> output = model(x)
"""

from torchvision_customizer.models.custom_cnn import CustomCNN

__all__ = ["CustomCNN"]
