"""torchvision-customizer: Build highly customizable CNNs from scratch.

A production-ready Python package that empowers researchers and developers
to create flexible, modular CNNs with fine-grained control over every
architectural decision.

Features:
    - Component Registry: Discover and use all building blocks
    - Model Composer: Intuitive >> operator for model composition
    - Architecture Recipes: Declarative model definitions
    - Templates: Pre-built parametric architectures (ResNet, VGG, etc.)

Example:
    >>> from torchvision_customizer import resnet, vgg
    >>> model = resnet(layers=50, num_classes=1000)
    >>> model = vgg(layers=16, num_classes=100)
    
    >>> from torchvision_customizer import Stem, Stage, Head
    >>> model = Stem(64) >> Stage(128, blocks=3) >> Head(num_classes=10)
"""

from torchvision_customizer.__version__ import (
    __author__,
    __author_email__,
    __description__,
    __license__,
    __url__,
    __version__,
)

# Registry
from torchvision_customizer import registry

# Values for recipe
from torchvision_customizer.recipe import Recipe, build_recipe

# Compose system
from torchvision_customizer.compose import (
    Stem,
    Stage,
    Head,
    Compose,
    Sequential,
    VisionModel,
)

# Templates (parametric architectures)
from torchvision_customizer.templates import (
    Template,
    resnet,
    vgg,
    mobilenet,
    densenet,
    efficientnet,
)

# Blocks (for direct access)
from torchvision_customizer.blocks import (
    ConvBlock,
    ResidualBlock,
    SEBlock,
    DepthwiseBlock,
    InceptionModule,
    DenseConnectionBlock,
    StandardBottleneck,
    WideBottleneck,
)

# Layers (utilities)
from torchvision_customizer.layers import (
    get_activation,
    get_normalization,
    get_pooling,
)

# Utils
from torchvision_customizer.utils import (
    print_model_summary,
    validate_architecture,
    get_model_flops,
)


__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__url__",
    
    # Registry
    "registry",
    
    # Recipes
    "Recipe",
    "build_recipe",
    
    # Compose
    "Stem",
    "Stage",
    "Head",
    "Compose",
    "Sequential",
    "VisionModel",
    
    # Templates (parametric)
    "Template",
    "resnet",
    "vgg",
    "mobilenet",
    "densenet",
    "efficientnet",
    
    # Blocks
    "ConvBlock",
    "ResidualBlock",
    "SEBlock",
    "DepthwiseBlock",
    "InceptionModule",
    "DenseConnectionBlock",
    "StandardBottleneck",
    "WideBottleneck",
    
    # Layers
    "get_activation",
    "get_normalization",
    "get_pooling",
    
    # Utils
    "print_model_summary",
    "validate_architecture",
    "get_model_flops",
]
