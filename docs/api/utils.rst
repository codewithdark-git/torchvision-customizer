==========================
API Reference: Utilities
==========================

.. py:module:: torchvision_customizer.utils

Analysis and optimization utilities.

Model Summary
=============

get_model_summary()
~~~~~~~~~~~~~~~~~~~

Get detailed model statistics:

.. code-block:: python

    from torchvision_customizer.utils import get_model_summary
    
    model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
    
    summary = get_model_summary(model, input_size=(1, 3, 224, 224))
    
    print(summary)

**Output includes:**

- Total parameters
- Trainable parameters
- Total FLOPs
- Memory footprint
- Layer-by-layer breakdown
- Model architecture diagram

Architecture Search
===================

find_optimal_architecture()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search for optimal configurations:

.. code-block:: python

    from torchvision_customizer.utils import find_optimal_architecture
    
    optimal_config = find_optimal_architecture(
        search_space={
            'num_conv_blocks': [3, 4, 5],
            'channels': [[64, 128, 256], [32, 64, 128, 256]],
            'activation': ['relu', 'gelu']
        },
        target_params=10_000_000,
        input_shape=(3, 224, 224)
    )

Validators
==========

validate_model_config()
~~~~~~~~~~~~~~~~~~~~~~

Validate model configuration:

.. code-block:: python

    from torchvision_customizer.utils import validate_model_config
    
    config = {
        'input_shape': (3, 224, 224),
        'num_classes': 10,
        'num_conv_blocks': 4,
        'channels': [64, 128, 256, 512]
    }
    
    is_valid, errors = validate_model_config(config)
    
    if not is_valid:
        for error in errors:
            print(f"Error: {error}")

validate_pooling_config()
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchvision_customizer.utils import validate_pooling_config
    
    config = {
        'pooling_type': 'max',
        'kernel_size': 2,
        'stride': 2
    }
    
    is_valid = validate_pooling_config(config)

Performance Analysis
====================

calculate_flops()
~~~~~~~~~~~~~~~~~

Calculate FLOPs:

.. code-block:: python

    from torchvision_customizer.utils import calculate_flops
    
    model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
    flops = calculate_flops(model, input_size=(1, 3, 224, 224))
    print(f"FLOPs: {flops:,}")

calculate_memory()
~~~~~~~~~~~~~~~~~~

Calculate memory usage:

.. code-block:: python

    from torchvision_customizer.utils import calculate_memory
    
    model = CustomCNN(input_shape=(3, 224, 224), num_classes=10)
    memory_mb = calculate_memory(model)
    print(f"Memory: {memory_mb:.2f} MB")

count_parameters()
~~~~~~~~~~~~~~~~~~

Count model parameters:

.. code-block:: python

    total_params = model.count_parameters()
    trainable_params = model.count_parameters(trainable_only=True)
    
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")

Export & Serialization
======================

export_onnx()
~~~~~~~~~~~~~

Export to ONNX format:

.. code-block:: python

    from torchvision_customizer.utils import export_onnx
    
    export_onnx(
        model,
        output_path='model.onnx',
        input_shape=(1, 3, 224, 224)
    )

export_torchscript()
~~~~~~~~~~~~~~~~~~~~

Export to TorchScript:

.. code-block:: python

    from torchvision_customizer.utils import export_torchscript
    
    scripted = export_torchscript(model)
    scripted.save('model.pt')

save_config()
~~~~~~~~~~~~~

Save model configuration:

.. code-block:: python

    from torchvision_customizer.utils import save_config
    
    config = {
        'input_shape': (3, 224, 224),
        'num_classes': 10,
        'num_conv_blocks': 4,
        'channels': [64, 128, 256, 512]
    }
    
    save_config(config, 'config.yaml')

load_config()
~~~~~~~~~~~~~

Load model configuration:

.. code-block:: python

    from torchvision_customizer.utils import load_config
    
    config = load_config('config.yaml')
    model = CustomCNN(**config)

Visualization
=============

visualize_architecture()
~~~~~~~~~~~~~~~~~~~~~~~~

Visualize model architecture:

.. code-block:: python

    from torchvision_customizer.utils import visualize_architecture
    
    visualize_architecture(model, output_path='architecture.png')

plot_layer_sizes()
~~~~~~~~~~~~~~~~~~

Plot layer output sizes:

.. code-block:: python

    from torchvision_customizer.utils import plot_layer_sizes
    
    plot_layer_sizes(model, input_size=(1, 3, 224, 224))

Quantization & Optimization
============================

quantize_model()
~~~~~~~~~~~~~~~~

Quantize model for inference:

.. code-block:: python

    from torchvision_customizer.utils import quantize_model
    
    quantized_model = quantize_model(
        model,
        quantization_type='int8'
    )

prune_model()
~~~~~~~~~~~~~

Remove unnecessary parameters:

.. code-block:: python

    from torchvision_customizer.utils import prune_model
    
    pruned_model = prune_model(
        model,
        pruning_ratio=0.3  # Remove 30% of parameters
    )

Distance Metrics
================

Calculate layer-wise differences:

.. code-block:: python

    from torchvision_customizer.utils import compare_models
    
    diff = compare_models(model1, model2)
    print(f"Parameter difference: {diff['param_diff']}")
    print(f"Output difference: {diff['output_diff']}")

Profiling
=========

profile_model()
~~~~~~~~~~~~~~~

Profile model execution:

.. code-block:: python

    from torchvision_customizer.utils import profile_model
    
    stats = profile_model(
        model,
        input_size=(1, 3, 224, 224),
        num_iterations=100
    )
    
    print(f"Forward time: {stats['forward_time_ms']:.2f} ms")
    print(f"Memory peak: {stats['memory_peak_mb']:.2f} MB")

Complete Utility Example
========================

.. code-block:: python

    from torchvision_customizer import CustomCNN
    from torchvision_customizer.utils import (
        get_model_summary,
        calculate_flops,
        calculate_memory,
        profile_model,
        export_torchscript
    )
    
    # Create model
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=10
    )
    
    # Get summary
    summary = get_model_summary(model)
    print("=== Model Summary ===")
    print(f"Parameters: {summary['total_params']:,}")
    print(f"Memory: {summary['total_memory_mb']:.2f} MB")
    
    # Calculate FLOPs
    flops = calculate_flops(model, input_size=(1, 3, 224, 224))
    print(f"FLOPs: {flops:,}")
    
    # Profile execution
    stats = profile_model(model, input_size=(1, 3, 224, 224))
    print(f"Forward time: {stats['forward_time_ms']:.2f} ms")
    
    # Export for production
    export_torchscript(model, 'model.pt')

Next Steps
==========

- :doc:`../examples/basic_usage` - See working examples
- :doc:`../user_guide/advanced` - Advanced patterns
- :doc:`../development/performance` - Performance optimization
