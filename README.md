# torchvision-customizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/pytorch-2.9+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Build highly customizable convolutional neural networks from scratch with an intuitive Python API.**

A production-ready Python package that empowers researchers and developers to create flexible, modular CNNs with fine-grained control over every architectural decision while maintaining full compatibility with the PyTorch ecosystem.

## 🎯 Key Features

- **🔧 Granular Control**: Customize network depth, channels, activations, normalization, pooling, and more
- **🧩 Modular Blocks**: Pre-built components (ConvBlock, ResidualBlock, InceptionBlock, SEBlock, etc.)
- **🏗️ Multiple APIs**: Simple interface for quick prototyping, builder pattern for advanced users
- **📊 Model Introspection**: Get parameter counts, FLOPs, memory footprint, and architecture summaries
- **⚙️ Configuration-Based**: Define models via YAML/JSON for reproducibility and sharing
- **🚀 Production-Ready**: Full type hints, comprehensive tests, and CI/CD integration
- **📈 Architecture Patterns**: Sequential, ResNet, DenseNet, Inception, MobileNet-style blocks

## 📦 Installation

### From PyPI (Coming Soon)
```bash
pip install torchvision-customizer
```

### From Source
```bash
git clone https://github.com/codewithdark-git/torchvision-customizer.git
cd torchvision-customizer
pip install -e ".[dev]"
```

### Requirements
- **Python**: 3.13 or higher
- **PyTorch**: 2.9.0 or higher
- **torchvision**: 0.24.1 or higher

## 🚀 Quick Start

### Basic Usage (Simple API)
```python
from torchvision_customizer import CustomCNN

# Create a simple 4-layer CNN in one line
model = CustomCNN(
    input_shape=(3, 224, 224),
    num_classes=1000,
    num_conv_blocks=4,
    channels=[64, 128, 256, 512],
    activation='relu'
)

# Inspect your model
print(model.summary())
print(f"Total parameters: {model.count_parameters():,}")
```

### Advanced Usage (Builder Pattern)
```python
from torchvision_customizer import CNNBuilder

model = (CNNBuilder(input_shape=(3, 224, 224))
    .add_conv_block(64, kernel_size=7, stride=2, activation='relu')
    .add_pooling('max', kernel_size=3, stride=2)
    .add_residual_block(64, num_convs=2)
    .add_residual_block(128, num_convs=2, downsample=True)
    .add_global_pooling('adaptive')
    .add_classifier([512, 256], num_classes=1000, dropout=0.3)
    .build()
)
```

### Configuration-Based (YAML)
```python
from torchvision_customizer import CustomCNN

# Load from YAML configuration
model = CustomCNN.from_yaml('model_config.yaml')

# Or from dictionary
config = {
    'input_shape': [3, 224, 224],
    'num_classes': 10,
    'blocks': [
        {'type': 'conv', 'channels': 64, 'kernel_size': 7},
        {'type': 'residual', 'channels': 128, 'num_layers': 2},
    ]
}
model = CustomCNN.from_config(config)
```

## 📚 Documentation

- [**API Reference**](docs/api_reference.md) - Complete API documentation
- [**User Guide**](docs/quickstart.md) - Getting started and common patterns
- [**Examples**](examples/) - Sample code and use cases
- [**Advanced Topics**](docs/advanced_usage.md) - Custom layers, optimization, deployment

## 🔬 Example Use Cases

### 1. Simple Image Classifier
```python
model = CustomCNN(
    input_shape=(3, 224, 224),
    num_classes=1000,
    num_conv_blocks=5
)
```

### 2. Lightweight Network for Edge Devices
```python
model = CustomCNN(
    input_shape=(3, 96, 96),
    channels=[32, 64, 128],
    use_depthwise_separable=True,
    num_classes=10
)
```

### 3. Deep Residual Network
```python
model = CustomCNN(
    num_conv_blocks=50,
    architecture='residual',
    channels='auto',
    num_classes=1000
)
```

## 🛠️ Development

### Setup Development Environment
```bash
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/ -v
pytest tests/ --cov=torchvision_customizer
```

### Code Quality
```bash
# Format code
black torchvision_customizer tests

# Sort imports
isort torchvision_customizer tests

# Lint
flake8 torchvision_customizer tests

# Type checking
mypy torchvision_customizer
```

## 📋 Package Structure

```
torchvision-customizer/
├── torchvision_customizer/
│   ├── __init__.py
│   ├── __version__.py
│   ├── models/              # Main model classes
│   ├── blocks/              # Building blocks (Conv, Residual, etc.)
│   ├── layers/              # Activations, normalizations, pooling
│   └── utils/               # Utilities (summary, validation, FLOPs)
├── tests/                   # Unit and integration tests
├── examples/                # Example scripts
├── docs/                    # Documentation
├── pyproject.toml
├── setup.py
├── README.md
└── LICENSE
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed changes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use torchvision-customizer in your research, please cite:

```bibtex
@software{torchvision_customizer,
  title={torchvision-customizer: Build Customizable CNNs from Scratch},
  author={Ahsan Umar, ComputerVision Team},
  url={https://github.com/codewithdark-git/torchvision-customizer},
  year={2025}
}
```

## 🎓 Acknowledgments

Built with 💙 for the computer vision and deep learning research community.

## 📞 Support

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/codewithdark-git/torchvision-customizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/codewithdark-git/torchvision-customizer/discussions)
