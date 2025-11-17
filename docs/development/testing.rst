================
Testing Guide
================

Comprehensive testing guidelines for the project.

Running Tests
=============

Basic Test Run
--------------

.. code-block:: bash

    pytest tests/

With Verbose Output
--------------------

.. code-block:: bash

    pytest tests/ -v

With Coverage Report
---------------------

.. code-block:: bash

    pytest tests/ --cov=torchvision_customizer --cov-report=html

Run Specific Test File
-----------------------

.. code-block:: bash

    pytest tests/test_blocks.py -v

Run Specific Test Function
---------------------------

.. code-block:: bash

    pytest tests/test_blocks.py::TestConvBlock::test_forward -v

Test Organization
=================

.. code-block:: text

    tests/
    ├── __init__.py
    ├── test_blocks.py           # Block tests
    ├── test_layers.py           # Layer tests
    ├── test_models.py           # Model tests
    ├── test_utils.py            # Utility tests
    ├── test_activations.py      # Activation tests
    ├── test_pooling.py          # Pooling tests
    └── conftest.py              # Shared fixtures

Test Files in Project
=====================

The project includes 382 tests across 8 test files:

- **test_blocks.py**: ~70 tests for block components
- **test_advanced_architecture.py**: ~68 tests for advanced patterns
- **test_custom_cnn.py**: ~51 tests for CNN models
- **test_pooling.py**: ~67 tests for pooling operations
- **test_residual_architecture.py**: ~66 tests for residual networks
- **test_activations.py**: ~28 tests for activation functions
- **test_residual_block.py**: ~34 tests for residual blocks
- **test_all_blocks.py**: ~6 integration tests

Coverage Summary
~~~~~~~~~~~~~~~~

Current coverage: **100%** across all modules

- torchvision_customizer/blocks/: 100%
- torchvision_customizer/layers/: 100%
- torchvision_customizer/models/: 100%
- torchvision_customizer/utils/: 100%

Writing Tests
=============

Test Structure
--------------

.. code-block:: python

    import pytest
    import torch
    from torchvision_customizer import MyComponent
    
    class TestMyComponent:
        """Test suite for MyComponent"""
        
        @pytest.fixture
        def component(self):
            """Create a component for testing"""
            return MyComponent(input_dim=64, output_dim=128)
        
        def test_initialization(self, component):
            """Test component initialization"""
            assert component is not None
            assert component.input_dim == 64
        
        def test_forward_pass(self, component):
            """Test forward pass"""
            x = torch.randn(1, 64, 224, 224)
            y = component(x)
            assert y.shape == (1, 128, 224, 224)

Fixtures
--------

Common test fixtures:

.. code-block:: python

    @pytest.fixture
    def sample_batch():
        """Create a sample batch"""
        return torch.randn(4, 3, 224, 224)
    
    @pytest.fixture
    def target_batch():
        """Create target labels"""
        return torch.randint(0, 10, (4,))

Parametrized Tests
-------------------

Test multiple inputs:

.. code-block:: python

    @pytest.mark.parametrize("activation", ["relu", "gelu", "mish"])
    def test_different_activations(activation):
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10,
            activation=activation
        )
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        assert y.shape == (1, 10)

Test Categories
===============

Unit Tests
----------

Test individual components:

.. code-block:: python

    def test_conv_block_output_shape():
        block = ConvBlock(3, 64, kernel_size=7, stride=2)
        x = torch.randn(1, 3, 224, 224)
        y = block(x)
        assert y.shape == (1, 64, 112, 112)

Integration Tests
-----------------

Test component interactions:

.. code-block:: python

    def test_conv_block_with_residual():
        conv = ConvBlock(3, 64)
        residual = ResidualBlock(64, 64)
        
        x = torch.randn(1, 3, 224, 224)
        x = conv(x)
        x = residual(x)
        
        assert x.shape == (1, 64, 224, 224)

Performance Tests
-----------------

Test execution speed:

.. code-block:: python

    def test_conv_block_performance():
        import time
        block = ConvBlock(3, 64)
        x = torch.randn(100, 3, 224, 224)
        
        start = time.time()
        for _ in range(10):
            _ = block(x)
        elapsed = time.time() - start
        
        # Should be reasonably fast
        assert elapsed < 10  # seconds for 10 iterations

Edge Case Tests
---------------

Test boundary conditions:

.. code-block:: python

    def test_conv_block_single_pixel():
        """Test with 1x1 input"""
        block = ConvBlock(3, 64, kernel_size=1)
        x = torch.randn(1, 3, 1, 1)
        y = block(x)
        assert y.shape == (1, 64, 1, 1)
    
    def test_large_batch_size():
        """Test with large batch"""
        block = ConvBlock(3, 64)
        x = torch.randn(128, 3, 224, 224)
        y = block(x)
        assert y.shape == (1 28, 64, 224, 224)

Gradient Flow Tests
-------------------

Test backward pass:

.. code-block:: python

    def test_gradient_flow():
        model = CustomCNN(
            input_shape=(3, 224, 224),
            num_classes=10
        )
        
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert model.conv1.weight.grad is not None

Device Tests
------------

Test CPU and GPU:

.. code-block:: python

    @pytest.mark.parametrize("device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"])
    def test_device_compatibility(device):
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
        model = model.to(device)
        
        x = torch.randn(1, 3, 224, 224).to(device)
        y = model(x)
        
        assert y.device.type == device

Mock Objects
============

Using pytest.mock:

.. code-block:: python

    from unittest.mock import Mock, patch
    
    def test_with_mock():
        # Create mock
        mock_optimizer = Mock()
        
        # Use in test
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
        # ... do something with mock_optimizer ...
        
        # Assert mock was called
        mock_optimizer.step.assert_called()

Assertions
==========

Common assertions:

.. code-block:: python

    # Shape assertions
    assert y.shape == (1, 10)
    assert y.shape[0] == 1
    
    # Value assertions
    assert y.min() >= 0
    assert y.max() <= 1
    
    # Dtype assertions
    assert y.dtype == torch.float32
    
    # Device assertions
    assert y.device.type == 'cuda'
    
    # Existence assertions
    assert model.weight is not None
    
    # Error assertions
    with pytest.raises(ValueError):
        invalid_model = CustomCNN(input_shape=(3, 0, 0))

Debugging Failed Tests
======================

Print Debug Information
-----------------------

.. code-block:: python

    def test_debug():
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        
        print(f"Input shape: {x.shape}")
        y = model(x)
        print(f"Output shape: {y.shape}")
        print(f"Output values: {y}")

Run with Print Output
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pytest tests/test_blocks.py::test_debug -s

Use pdb Debugger
-----------------

.. code-block:: python

    def test_with_breakpoint():
        model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        
        # Breakpoint here
        import pdb; pdb.set_trace()
        
        y = model(x)

Run with pdb
~~~~~~~~~~~~

.. code-block:: bash

    pytest tests/test_blocks.py::test_with_breakpoint -s --pdb

CI/CD Integration
=================

GitHub Actions Configuration
-----------------------------

Tests run automatically on:

- Every push to main branch
- Every pull request
- Daily scheduled runs

Test Matrix
~~~~~~~~~~~

- Python 3.13
- Ubuntu 22.04
- PyTorch 2.9.x

Performance Benchmarking
========================

Track performance over time:

.. code-block:: bash

    # Run with timing
    pytest tests/ -v --durations=10

Identify Slow Tests
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pytest tests/ --durations=0

Coverage Analysis
=================

Generate Coverage Report
------------------------

.. code-block:: bash

    pytest tests/ --cov=torchvision_customizer --cov-report=html

View Coverage
~~~~~~~~~~~~~

Open `htmlcov/index.html` in browser to see detailed coverage report.

Target Coverage
~~~~~~~~~~~~~~~

- Minimum: 90%
- Target: 95%
- Current: 100%

Best Practices
==============

✅ Write tests alongside code  
✅ Use descriptive test names  
✅ Keep tests simple and focused  
✅ Use fixtures for reusable setup  
✅ Test edge cases  
✅ Test error conditions  
✅ Avoid test interdependencies  
✅ Use parametrize for variations  
✅ Keep tests fast  
✅ Document complex tests  

Continuous Testing
==================

Run tests during development:

.. code-block:: bash

    # Watch for changes and rerun
    pytest-watch tests/
    
    # Or
    ptw tests/

Next Steps
==========

- :doc:`../development/contributing` - Contributing guide
- :doc:`../development/performance` - Performance guidelines
- Review existing test files in `tests/` directory
