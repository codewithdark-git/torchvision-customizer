=======================
Installation Guide
=======================

Requirements
============

Before installing torchvision-customizer, ensure you have:

- **Python**: 3.13 or higher
- **pip**: 21.0 or higher
- **PyTorch**: 2.9.0 or higher
- **torchvision**: 0.24.1 or higher
- **CUDA** (optional): For GPU acceleration

Check Your Python Version
--------------------------

.. code-block:: bash

    python --version
    # Should show Python 3.13.x or higher

Check Your PyTorch Installation
--------------------------------

.. code-block:: bash

    python -c "import torch; print(torch.__version__)"
    # Should show 2.9.0 or higher

Installation Methods
====================

Method 1: From Source (Recommended for Development)
---------------------------------------------------

Clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/codewithdark-git/torchvision-customizer.git
    cd torchvision-customizer
    pip install -e ".[dev]"

This installs the package in editable mode with all development dependencies.

**What's included:**

- ✅ Core package
- ✅ Development tools (pytest, black, mypy)
- ✅ Documentation tools (sphinx, sphinx-rtd-theme)
- ✅ Optional dependencies

Method 2: From Source (Production)
-----------------------------------

If you only need the core package:

.. code-block:: bash

    git clone https://github.com/codewithdark-git/torchvision-customizer.git
    cd torchvision-customizer
    pip install .

Method 3: From PyPI (Coming Soon)
----------------------------------

Once released on PyPI:

.. code-block:: bash

    pip install torchvision-customizer

Optional Dependencies
====================

GPU Support
-----------

For CUDA acceleration, install PyTorch with GPU support:

.. code-block:: bash

    # CUDA 12.1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    
    # CUDA 11.8
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    # CPU only
    pip install torch torchvision

Documentation
--------------

To build documentation locally:

.. code-block:: bash

    pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

Development Tools
------------------

For development and testing:

.. code-block:: bash

    pip install pytest pytest-cov black mypy flake8

Verify Installation
===================

Test that everything is installed correctly:

.. code-block:: python

    # Basic import test
    from torchvision_customizer import CustomCNN, CNNBuilder
    from torchvision_customizer.blocks import ConvBlock, ResidualBlock
    from torchvision_customizer.layers import get_activation
    from torchvision_customizer.utils import get_model_summary
    
    print("✅ All imports successful!")

Create your first model to verify:

.. code-block:: python

    from torchvision_customizer import CustomCNN
    import torch
    
    # Create a simple model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10,
        num_conv_blocks=2
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    
    print(f"✅ Model created successfully!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

Troubleshooting
===============

ImportError: No module named 'torch'
------------------------------------

PyTorch is not installed. Install it with:

.. code-block:: bash

    pip install torch torchvision

CUDA not available
-------------------

If you want GPU support but CUDA is not available:

.. code-block:: bash

    # Check CUDA availability
    python -c "import torch; print(torch.cuda.is_available())"
    
    # Install PyTorch with correct CUDA version
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

ImportError: cannot import name 'CustomCNN'
--------------------------------------------

The package is not installed in your current Python environment:

.. code-block:: bash

    # Check if installed
    pip show torchvision-customizer
    
    # Reinstall
    pip install -e .

Version Conflicts
-----------------

If you have version conflicts, create a fresh virtual environment:

.. code-block:: bash

    # Create virtual environment
    python -m venv myenv
    
    # Activate it
    source myenv/bin/activate  # On Windows: myenv\Scripts\activate
    
    # Install dependencies
    pip install torch torchvision
    pip install -e .

Virtual Environment Setup
==========================

Using venv (Recommended)
------------------------

.. code-block:: bash

    # Create environment
    python -m venv venv
    
    # Activate on Windows
    venv\Scripts\activate
    
    # Activate on Linux/Mac
    source venv/bin/activate
    
    # Install package
    pip install -e ".[dev]"

Using Conda
-----------

.. code-block:: bash

    # Create environment
    conda create -n torchvis python=3.13
    conda activate torchvis
    
    # Install PyTorch
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    
    # Install package
    pip install -e ".[dev]"

Development Setup
=================

If you plan to contribute or develop:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/codewithdark-git/torchvision-customizer.git
    cd torchvision-customizer
    
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    
    # Install all dependencies
    pip install -e ".[dev]"
    
    # Run tests to verify
    pytest tests/ -v
    
    # Check code quality
    black --check torchvision_customizer/
    mypy torchvision_customizer/

Docker Installation (Optional)
==============================

For containerized development:

.. code-block:: dockerfile

    FROM pytorch/pytorch:2.9.0-cuda12.1-cudnn8-runtime
    
    WORKDIR /workspace
    COPY . .
    
    RUN pip install -e ".[dev]"
    
    CMD ["python"]

Build and run:

.. code-block:: bash

    docker build -t torchvis:latest .
    docker run -it torchvis:latest

Next Steps
==========

After installation, check out:

- :doc:`Quick Start <quick_start>` - Create your first model
- :doc:`User Guide <user_guide/basics>` - Learn the basics
- :doc:`Examples <examples/basic_usage>` - Explore real-world examples
- :doc:`API Reference <api/blocks>` - Detailed API documentation

Getting Help
============

If you encounter issues:

1. **Check the FAQ** - Common problems and solutions
2. **Search GitHub Issues** - Your problem might be solved
3. **Read Documentation** - Comprehensive guides and examples
4. **Open an Issue** - Report bugs with reproduction steps

System Information
==================

Check your system compatibility:

.. code-block:: python

    import torch
    import torchvision
    import sys
    
    print("=== System Information ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
