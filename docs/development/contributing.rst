====================
Contributing Guide
====================

Thank you for your interest in contributing to torchvision-customizer! This guide will help you get started.

Getting Started
===============

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

    git clone https://github.com/YOUR_USERNAME/torchvision-customizer.git
    cd torchvision-customizer

3. Create a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate

4. Install in development mode:

.. code-block:: bash

    pip install -e ".[dev]"

5. Create a branch for your changes:

.. code-block:: bash

    git checkout -b feature/your-feature-name

Development Workflow
====================

Code Style
----------

We use Black for code formatting:

.. code-block:: bash

    black torchvision_customizer/
    black tests/

Type Checking
-------------

We use mypy for type hints:

.. code-block:: bash

    mypy torchvision_customizer/

Linting
-------

We use flake8 for linting:

.. code-block:: bash

    flake8 torchvision_customizer/ tests/

Testing
-------

Run tests before committing:

.. code-block:: bash

    pytest tests/ -v
    pytest tests/ --cov=torchvision_customizer  # With coverage

Pre-commit Checks
-----------------

Run all checks:

.. code-block:: bash

    black --check torchvision_customizer/
    mypy torchvision_customizer/
    flake8 torchvision_customizer/ tests/
    pytest tests/ -v

Adding Features
===============

Step 1: Create Tests First
~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow Test-Driven Development (TDD):

.. code-block:: python

    # tests/test_new_feature.py
    import pytest
    import torch
    from torchvision_customizer import NewFeature
    
    def test_new_feature_basic():
        feature = NewFeature()
        assert feature is not None
    
    def test_new_feature_forward():
        feature = NewFeature(input_dim=64, output_dim=128)
        x = torch.randn(1, 64, 224, 224)
        y = feature(x)
        assert y.shape == (1, 128, 224, 224)

Step 2: Implement Feature
~~~~~~~~~~~~~~~~~~~~~~~~~

Add your implementation:

.. code-block:: python

    # torchvision_customizer/blocks/new_feature.py
    import torch.nn as nn
    from typing import Optional
    
    class NewFeature(nn.Module):
        """
        Description of your feature.
        
        Args:
            input_dim: Number of input channels
            output_dim: Number of output channels
            ...
        """
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            **kwargs
        ) -> None:
            super().__init__()
            # Implementation here
        
        def forward(self, x):
            # Forward pass implementation
            return x

Step 3: Add Tests
~~~~~~~~~~~~~~~~~

Test edge cases and normal cases:

.. code-block:: python

    def test_new_feature_edge_case():
        feature = NewFeature(1, 1)
        x = torch.randn(1, 1, 1, 1)
        y = feature(x)
        assert y.shape == (1, 1, 1, 1)
    
    def test_new_feature_batch():
        feature = NewFeature(64, 128)
        x = torch.randn(32, 64, 224, 224)
        y = feature(x)
        assert y.shape == (32, 128, 224, 224)

Step 4: Documentation
~~~~~~~~~~~~~~~~~~~~~

Document your feature:

.. code-block:: python

    class NewFeature(nn.Module):
        """
        Brief description.
        
        Longer description with details about what this feature does,
        when to use it, and key characteristics.
        
        Args:
            input_dim (int): Number of input channels
            output_dim (int): Number of output channels
            kernel_size (int, optional): Kernel size. Default: 3
            stride (int, optional): Stride. Default: 1
            padding (int, optional): Padding. Default: 1
            activation (str, optional): Activation function. Default: 'relu'
        
        Returns:
            Tensor: Output tensor of shape (B, output_dim, H, W)
        
        Raises:
            ValueError: If input_dim or output_dim is invalid
        
        Example:
            >>> feature = NewFeature(64, 128)
            >>> x = torch.randn(1, 64, 224, 224)
            >>> y = feature(x)
            >>> print(y.shape)
            torch.Size([1, 128, 224, 224])
        """

Step 5: Run All Tests
~~~~~~~~~~~~~~~~~~~~~

Ensure all tests pass:

.. code-block:: bash

    pytest tests/ -v
    pytest tests/test_new_feature.py -v

Step 6: Submit Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Push to your fork and create a pull request:

.. code-block:: bash

    git push origin feature/your-feature-name

Commit Message Guidelines
==========================

Use clear, descriptive commit messages:

**Good:**

.. code-block:: text

    Add SEBlock with channel attention mechanism
    
    - Implement Squeeze-and-Excitation block
    - Add comprehensive tests (10 test cases)
    - Add examples and documentation
    - Fixes #42

**Avoid:**

.. code-block:: text

    Update code
    Fix bug
    Changes

Code Review Process
===================

1. **Automated Checks**: GitHub Actions runs tests automatically
2. **Code Review**: Maintainers review your code
3. **Revisions**: Update code based on feedback
4. **Merge**: Once approved, your code is merged

Review Criteria
---------------

- ✅ All tests pass
- ✅ Code follows style guidelines
- ✅ Type hints are complete
- ✅ Documentation is clear
- ✅ No breaking changes (unless major version)
- ✅ Includes tests for new features

Reporting Issues
================

When reporting bugs, include:

1. **Description**: What's the problem?
2. **Reproduction**: How to reproduce it?
3. **Environment**:
   - Python version
   - PyTorch version
   - OS

4. **Expected vs Actual**: What should happen vs what actually happens?

Example Issue:

.. code-block:: text

    Title: ResidualBlock fails with stride=2 and downsample=False
    
    Description:
    ResidualBlock produces shape mismatch when stride=2 and downsample=False
    
    Reproduction:
    >>> from torchvision_customizer.blocks import ResidualBlock
    >>> import torch
    >>> block = ResidualBlock(64, 64, stride=2, downsample=False)
    >>> x = torch.randn(1, 64, 224, 224)
    >>> y = block(x)  # RuntimeError: size mismatch
    
    Environment:
    - Python 3.13
    - PyTorch 2.9.1
    - Ubuntu 22.04
    
    Expected:
    Output shape should be (1, 64, 112, 112)
    
    Actual:
    RuntimeError: size mismatch: expected tensor of shape [1, 64, 112, 112]

Documentation Standards
=======================

All docstrings must follow Google style:

.. code-block:: python

    def function_name(
        param1: int,
        param2: str,
        param3: Optional[float] = None
    ) -> Tuple[int, str]:
        """
        One-line summary.
        
        Longer description explaining what the function does,
        including any important details or caveats.
        
        Args:
            param1: Description of param1
            param2: Description of param2
            param3: Optional description. Default is None
        
        Returns:
            Description of return value
        
        Raises:
            ValueError: When this error occurs
            TypeError: When this error occurs
        
        Example:
            >>> result = function_name(1, "test")
            >>> print(result)
            (1, 'test')
        
        Note:
            Important notes about usage or behavior
        """

Testing Standards
=================

All new code must include tests:

.. code-block:: python

    import pytest
    import torch
    from torchvision_customizer import MyFeature
    
    class TestMyFeature:
        @pytest.fixture
        def feature(self):
            return MyFeature(input_dim=64, output_dim=128)
        
        def test_initialization(self, feature):
            assert feature is not None
        
        def test_forward_shape(self, feature):
            x = torch.randn(1, 64, 224, 224)
            y = feature(x)
            assert y.shape == (1, 128, 224, 224)
        
        def test_batch_processing(self, feature):
            x = torch.randn(32, 64, 224, 224)
            y = feature(x)
            assert y.shape == (32, 128, 224, 224)
        
        def test_gradient_flow(self, feature):
            x = torch.randn(1, 64, 224, 224, requires_grad=True)
            y = feature(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None

Coverage Requirements
~~~~~~~~~~~~~~~~~~~~~

- Minimum 90% code coverage
- All new code must have tests
- Edge cases must be covered

Check coverage:

.. code-block:: bash

    pytest tests/ --cov=torchvision_customizer --cov-report=html

Performance Considerations
==========================

When adding features:

1. **Benchmark**: Measure performance impact
2. **Optimize**: Use efficient implementations
3. **Document**: Note any performance characteristics

Example benchmark:

.. code-block:: python

    import time
    import torch
    from torchvision_customizer import MyFeature
    
    feature = MyFeature()
    x = torch.randn(100, 64, 224, 224)
    
    start = time.time()
    for _ in range(10):
        _ = feature(x)
    elapsed = time.time() - start
    
    print(f"Average time: {elapsed / 10:.4f}s")

CI/CD Pipeline
==============

Automated checks on every pull request:

1. **Tests**: pytest with coverage
2. **Linting**: flake8 and pylint
3. **Type Checking**: mypy
4. **Code Formatting**: black
5. **Documentation**: sphinx build

All checks must pass before merging.

Resources
=========

- **Style Guide**: PEP 8
- **Type Hints**: PEP 484
- **Documentation**: Google Style Guide
- **Testing**: pytest documentation
- **Git Workflow**: GitHub Flow

Questions?
==========

- Check existing issues/PRs
- Read API documentation
- Look at examples
- Open a discussion

Thank you for contributing!
