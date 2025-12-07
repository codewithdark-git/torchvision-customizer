# torchvision-customizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

> **Build highly customizable convolutional neural networks from scratch with a 3-tier component-based API.**

**torchvision-customizer** empowers researchers to build parametric models and explore new architectures without boilerplate. No pre-trained weights, just pure architectural flexibility.



## 🌟 Key Features

### 1. Component Registry (Tier 1)
Centralized discovery for all building blocks.
```python
from torchvision_customizer import registry
block = registry.get('residual')(64, 64)
```

### 2. Architecture Recipes (Tier 2)
Declarative, human-readable blueprints for rapid prototyping.
```python
from torchvision_customizer import Recipe, build_recipe

recipe = Recipe(
    stem="conv(64, k=7, s=2)",
    stages=["residual(64) x 2", "residual(128) x 2 | downsample"],
    head="linear(10)"
)
model = build_recipe(recipe)
```

### 3. Model Composer (Tier 3)
Fluent API with operator overloading (`>>`, `|`, `+`) for intuitive construction.
```python
from torchvision_customizer import Stem, Stage, Head

model = (
    Stem(64)
    >> Stage(64, blocks=2, pattern='residual')
    >> Stage(128, blocks=2, pattern='residual+se', downsample=True)
    >> Head(num_classes=10)
)
```

### 4. Parametric Templates
Standard architectures built from scratch with modern customization options.
```python
from torchvision_customizer import resnet, vgg, efficientnet, Template

# Standard
model = resnet(layers=50, num_classes=100)

# Customized
model = (Template.resnet(layers=50)
         .replace_activation('gelu')
         .add_attention('se')
         .build(num_classes=10))
```

## 📦 Installation

### From GitHub (Recommended)
```bash
pip install git+https://github.com/codewithdark-git/torchvision-customizer.git
```

### From Source (Development)
```bash
git clone https://github.com/codewithdark-git/torchvision-customizer.git
cd torchvision-customizer
pip install -e .
```

## 🚀 Quick Start

### Building a Custom ResNet-like Model
```python
from torchvision_customizer import Stem, Stage, Head

model = (
    Stem(64, kernel=7, stride=2)
    >> Stage(64, blocks=3, pattern='residual')                           # Stage 1
    >> Stage(128, blocks=4, pattern='residual', downsample=True)         # Stage 2
    >> Stage(256, blocks=6, pattern='residual+se', downsample=True)      # Stage 3 (with SE)
    >> Stage(512, blocks=3, pattern='residual', downsample=True)         # Stage 4
    >> Head(num_classes=1000)
)

print(model.explain())
```

### Using a Recipe
```python
from torchvision_customizer import Recipe, build_recipe

recipe = Recipe(
    name="MobileNetV2-like",
    stem="conv(32, stride=2)",
    stages=[
        "bottleneck(16, expansion=1) x 1",
        "bottleneck(24, expansion=6) x 2 | downsample",
        "bottleneck(32, expansion=6) x 3 | downsample",
    ],
    head="linear(100)"
)
model = build_recipe(recipe)
```

## 📚 Documentation

Full documentation is available in the `docs/` directory.

- **Registry**: `registry.list('block')` to see available blocks
- **Templates**: `resnet`, `vgg`, `densenet`, `mobilenet`, `efficientnet`
- **Composers**: `Stem`, `Stage`, `Head`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
