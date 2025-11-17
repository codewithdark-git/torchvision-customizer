"""Examples demonstrating CustomCNN usage."""

import torch
import torch.nn as nn
from torchvision_customizer.models import CustomCNN


def example_basic_usage() -> None:
    """Example 1: Basic CustomCNN usage with default parameters."""
    print("=" * 70)
    print("Example 1: Basic CustomCNN Usage")
    print("=" * 70)

    # Create model with default parameters
    model = CustomCNN(input_shape=(3, 224, 224), num_classes=1000)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Input shape: (3, 224, 224)")
    print(f"Output classes: 1000")
    print(f"Number of conv blocks: {model.num_conv_blocks}")
    print(f"Auto-generated channels: {model.channels}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    print(f"\nInput batch shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print()


def example_custom_channels() -> None:
    """Example 2: CustomCNN with custom channel specification."""
    print("=" * 70)
    print("Example 2: Custom Channel Specification")
    print("=" * 70)

    # Define custom channels
    custom_channels = [32, 64, 128, 256, 512]
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=5,
        channels=custom_channels,
    )
    print(f"Custom channels: {custom_channels}")
    print(f"Model channels: {model.channels}")

    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_cifar10_model() -> None:
    """Example 3: Small model for CIFAR-10 classification."""
    print("=" * 70)
    print("Example 3: CIFAR-10 Classification Model")
    print("=" * 70)

    model = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=3,
        channels=[32, 64, 128],
        activation='relu',
        use_batchnorm=True,
        dropout_rate=0.3,
        use_pooling=True,
    )

    print("Model Configuration:")
    config = model.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Test with typical CIFAR-10 batch
    batch_size = 32
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)

    print(f"\nBatch size: {batch_size}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.count_parameters():,}")
    print()


def example_different_activations() -> None:
    """Example 4: Models with different activation functions."""
    print("=" * 70)
    print("Example 4: Different Activation Functions")
    print("=" * 70)

    activations = ['relu', 'leaky_relu', 'gelu', 'silu']
    x = torch.randn(2, 3, 32, 32)

    print("Testing different activation functions:\n")
    for activation in activations:
        model = CustomCNN(
            input_shape=(3, 32, 32),
            num_classes=10,
            num_conv_blocks=2,
            activation=activation,
        )
        output = model(x)
        params = model.count_parameters()
        print(f"{activation:12} → Output shape: {output.shape}, Parameters: {params:,}")
    print()


def example_without_batchnorm() -> None:
    """Example 5: Model without batch normalization."""
    print("=" * 70)
    print("Example 5: Model Without Batch Normalization")
    print("=" * 70)

    model_with_bn = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=3,
        use_batchnorm=True,
    )

    model_without_bn = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=3,
        use_batchnorm=False,
    )

    x = torch.randn(2, 3, 32, 32)

    params_with_bn = model_with_bn.count_parameters()
    params_without_bn = model_without_bn.count_parameters()

    print(f"With BatchNorm:    {params_with_bn:,} parameters")
    print(f"Without BatchNorm: {params_without_bn:,} parameters")
    print(f"Difference:        {params_with_bn - params_without_bn:,} parameters")

    output_with_bn = model_with_bn(x)
    output_without_bn = model_without_bn(x)

    print(f"\nOutput shapes match: {output_with_bn.shape == output_without_bn.shape}")
    print()


def example_with_dropout() -> None:
    """Example 6: Model with dropout regularization."""
    print("=" * 70)
    print("Example 6: Model With Dropout Regularization")
    print("=" * 70)

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
        dropout_rate=0.5,
    )

    x = torch.randn(4, 3, 224, 224)

    print("Testing dropout in training and eval modes:\n")

    # Training mode (dropout active)
    model.train()
    output_train1 = model(x)
    output_train2 = model(x)

    # Outputs should differ due to dropout randomness
    difference = (output_train1 - output_train2).abs().max().item()
    print(f"Training mode - Max output difference: {difference:.6f} (due to dropout)")

    # Eval mode (dropout disabled)
    model.eval()
    with torch.no_grad():
        output_eval1 = model(x)
        output_eval2 = model(x)

    difference_eval = (output_eval1 - output_eval2).abs().max().item()
    print(f"Eval mode - Max output difference: {difference_eval:.6f} (no dropout)")
    print()


def example_without_pooling() -> None:
    """Example 7: Model without pooling layers."""
    print("=" * 70)
    print("Example 7: Model Without Pooling")
    print("=" * 70)

    model_with_pool = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=3,
        use_pooling=True,
    )

    model_without_pool = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=3,
        use_pooling=False,
    )

    x = torch.randn(2, 3, 32, 32)

    params_with_pool = model_with_pool.count_parameters()
    params_without_pool = model_without_pool.count_parameters()

    print(f"With pooling:    {params_with_pool:,} parameters")
    print(f"Without pooling: {params_without_pool:,} parameters")

    output_with_pool = model_with_pool(x)
    output_without_pool = model_without_pool(x)

    print(f"\nWith pooling output:    {output_with_pool.shape}")
    print(f"Without pooling output: {output_without_pool.shape}")
    print()


def example_different_pooling_types() -> None:
    """Example 8: Different pooling strategies."""
    print("=" * 70)
    print("Example 8: Different Pooling Types")
    print("=" * 70)

    pooling_types = ['max', 'avg']
    x = torch.randn(2, 3, 64, 64)

    print("Testing different pooling types:\n")
    for pool_type in pooling_types:
        model = CustomCNN(
            input_shape=(3, 64, 64),
            num_classes=10,
            num_conv_blocks=2,
            pooling_type=pool_type,
            use_pooling=True,
        )
        output = model(x)
        params = model.count_parameters()
        print(f"{pool_type:5} pooling → Output shape: {output.shape}, Parameters: {params:,}")
    print()


def example_parameter_counting() -> None:
    """Example 9: Parameter counting and analysis."""
    print("=" * 70)
    print("Example 9: Parameter Counting")
    print("=" * 70)

    # Small model
    small_model = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=2,
        channels=[16, 32],
    )

    # Medium model
    medium_model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
        channels=[64, 128, 256, 512],
    )

    # Large model
    large_model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=6,
    )

    small_params = small_model.count_parameters()
    medium_params = medium_model.count_parameters()
    large_params = large_model.count_parameters()

    print("Model Size Comparison:")
    print(f"  Small model:  {small_params:>15,} parameters")
    print(f"  Medium model: {medium_params:>15,} parameters")
    print(f"  Large model:  {large_params:>15,} parameters")

    print(f"\nMedium/Small ratio:   {medium_params / small_params:>6.1f}x")
    print(f"Large/Medium ratio:   {large_params / medium_params:>6.1f}x")
    print()


def example_model_configuration() -> None:
    """Example 10: Retrieving and inspecting model configuration."""
    print("=" * 70)
    print("Example 10: Model Configuration")
    print("=" * 70)

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=5,
        channels=[64, 128, 256, 512, 512],
        activation='leaky_relu',
        use_batchnorm=True,
        dropout_rate=0.2,
        pooling_type='max',
        use_pooling=True,
    )

    # Get and display configuration
    config = model.get_config()

    print("Model Configuration:")
    print("-" * 70)
    for key, value in config.items():
        if isinstance(value, list):
            print(f"  {key:25} : {value}")
        elif isinstance(value, tuple):
            print(f"  {key:25} : {value}")
        else:
            print(f"  {key:25} : {value}")

    print("-" * 70)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Trainable params: {model.count_parameters(trainable_only=True):,}")
    print()


def example_forward_pass_gradient_flow() -> None:
    """Example 11: Forward pass and gradient computation."""
    print("=" * 70)
    print("Example 11: Forward Pass & Gradient Flow")
    print("=" * 70)

    model = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=3,
    )

    # Create input with gradient tracking
    x = torch.randn(4, 3, 32, 32, requires_grad=True)
    print(f"Input shape: {x.shape}")
    print(f"Requires grad: {x.requires_grad}")

    # Forward pass
    output = model(x)
    print(f"\nOutput shape: {output.shape}")

    # Compute loss and backward pass
    loss = output.sum()
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    print(f"\nGradient computed for input: {x.grad is not None}")
    print(f"Input gradient shape: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"Input gradient non-zero elements: {(x.grad != 0).sum().item() if x.grad is not None else 0}")
    print()


def example_string_representation() -> None:
    """Example 12: Model string representation."""
    print("=" * 70)
    print("Example 12: Model String Representation")
    print("=" * 70)

    model = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=3,
        channels=[32, 64, 128],
        activation='relu',
    )

    print(model)
    print()


if __name__ == '__main__':
    example_basic_usage()
    example_custom_channels()
    example_cifar10_model()
    example_different_activations()
    example_without_batchnorm()
    example_with_dropout()
    example_without_pooling()
    example_different_pooling_types()
    example_parameter_counting()
    example_model_configuration()
    example_forward_pass_gradient_flow()
    example_string_representation()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
