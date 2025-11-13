"""Example usage of activation function utilities.

This module demonstrates various ways to use the get_activation() function
and related utilities for working with different activation functions in
neural networks.

Examples covered:
1. Basic activation creation with default parameters
2. Custom activation parameters
3. Case-insensitive activation names
4. Checking supported activations
5. Using activations in neural network modules
6. Comparing different activations
7. Integrated with ConvBlock (future)
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import (
    get_activation,
    is_activation_supported,
    get_supported_activations,
    ActivationFactory,
)


def example_1_basic_activation_creation():
    """Example 1: Basic activation creation with default parameters.
    
    Demonstrates how to create activation functions with standard settings.
    """
    print("=" * 70)
    print("Example 1: Basic Activation Creation with Default Parameters")
    print("=" * 70)
    
    # Create sample input
    x = torch.randn(2, 64, 32, 32)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]\n")
    
    # Create different activations
    activations = ['relu', 'leaky_relu', 'gelu', 'silu', 'sigmoid', 'tanh']
    
    for activation_name in activations:
        activation = get_activation(activation_name)
        output = activation(x)
        print(f"{activation_name:12} -> Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print()


def example_2_custom_activation_parameters():
    """Example 2: Creating activations with custom parameters.
    
    Shows how to pass custom parameters to activation functions.
    """
    print("=" * 70)
    print("Example 2: Custom Activation Parameters")
    print("=" * 70)
    
    x = torch.randn(2, 64, 32, 32)
    
    # Leaky ReLU with different slopes
    print("Leaky ReLU with different negative slopes:")
    slopes = [0.01, 0.1, 0.2, 0.5]
    
    for slope in slopes:
        leaky_relu = get_activation('leaky_relu', negative_slope=slope)
        output = leaky_relu(x)
        print(f"  slope={slope:4.2f} -> Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print()
    
    # ELU with different alphas
    print("ELU with different alpha values:")
    alphas = [0.5, 1.0, 2.0]
    
    for alpha in alphas:
        elu = get_activation('elu', alpha=alpha)
        output = elu(x)
        print(f"  alpha={alpha:4.1f} -> Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print()


def example_3_case_insensitive_names():
    """Example 3: Case-insensitive activation names.
    
    Demonstrates that activation names are case-insensitive and whitespace-tolerant.
    """
    print("=" * 70)
    print("Example 3: Case-Insensitive Activation Names")
    print("=" * 70)
    
    x = torch.randn(2, 64, 32, 32)
    
    # Test various case formats
    test_names = [
        'relu',
        'RELU',
        'ReLU',
        'rElU',
        ' relu ',  # with whitespace
        '\trelu\n',  # with tabs and newlines
    ]
    
    print("Testing case-insensitivity for ReLU:")
    for name in test_names:
        try:
            activation = get_activation(name)
            output = activation(x)
            print(f"  '{repr(name):15}' ✓ Successfully created and executed")
        except ValueError as e:
            print(f"  '{repr(name):15}' ✗ Error: {e}")
    
    print()


def example_4_check_supported_activations():
    """Example 4: Checking supported activations.
    
    Shows how to query which activations are supported.
    """
    print("=" * 70)
    print("Example 4: Check Supported Activations")
    print("=" * 70)
    
    # Get list of all supported activations
    supported = get_supported_activations()
    print(f"Total supported activations: {len(supported)}\n")
    print("Supported activations:")
    for i, activation in enumerate(supported, 1):
        print(f"  {i:2}. {activation}")
    
    print("\n" + "-" * 70)
    print("Checking if specific activations are supported:\n")
    
    test_names = ['relu', 'gelu', 'custom_activation', 'unsupported']
    
    for name in test_names:
        is_supported = is_activation_supported(name)
        status = "✓ Supported" if is_supported else "✗ Not supported"
        print(f"  {name:20} -> {status}")
    
    print()


def example_5_activation_in_sequential():
    """Example 5: Using get_activation() in nn.Sequential models.
    
    Demonstrates how to build neural networks with flexible activation functions.
    """
    print("=" * 70)
    print("Example 5: Using Activations in nn.Sequential")
    print("=" * 70)
    
    # Create a simple CNN with custom activations
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        get_activation('relu'),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        get_activation('leaky_relu', negative_slope=0.2),
        
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 10),
    )
    
    print("Model architecture:")
    for i, module in enumerate(model, 1):
        print(f"  {i:2}. {module}")
    
    print("\nTesting forward pass:")
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print()


def example_6_custom_module_with_activation():
    """Example 6: Using activations in custom neural network modules.
    
    Shows how to create reusable blocks with configurable activations.
    """
    print("=" * 70)
    print("Example 6: Custom Module with Configurable Activation")
    print("=" * 70)
    
    class ConvBlock(nn.Module):
        """Simple convolutional block with configurable activation."""
        
        def __init__(self, in_channels, out_channels, activation='relu'):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.activation = get_activation(activation)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation(x)
            return x
    
    # Create blocks with different activations
    print("Creating ConvBlocks with different activations:\n")
    
    activations = ['relu', 'leaky_relu', 'gelu', 'silu']
    x = torch.randn(2, 3, 32, 32)
    
    for activation_name in activations:
        block = ConvBlock(3, 64, activation=activation_name)
        output = block(x)
        print(f"  {activation_name:12} -> Output shape: {output.shape}, "
              f"Range: [{output.min():.4f}, {output.max():.4f}]")
    
    print()


def example_7_compare_activations():
    """Example 7: Comparing different activations on the same input.
    
    Visually compares how different activations transform the same input.
    """
    print("=" * 70)
    print("Example 7: Compare Different Activations")
    print("=" * 70)
    
    # Create a simple input range for visualization
    x = torch.linspace(-3, 3, 7)
    
    print("Input values:", x.tolist())
    print("\nActivation outputs for the same input:\n")
    
    activations = [
        'relu',
        'leaky_relu',
        'gelu',
        'silu',
        'sigmoid',
        'tanh',
        'elu',
        'selu',
    ]
    
    for activation_name in activations:
        activation = get_activation(activation_name)
        output = activation(x)
        print(f"{activation_name:12}: {output.tolist()}")
    
    print()


def example_8_activation_factory():
    """Example 8: Using the ActivationFactory class.
    
    Demonstrates the factory pattern for managing activations.
    """
    print("=" * 70)
    print("Example 8: ActivationFactory Class")
    print("=" * 70)
    
    factory = ActivationFactory()
    
    print("Factory methods:")
    print(f"  1. create(name, **kwargs) - Create activation instance")
    print(f"  2. is_supported(name) - Check if activation is supported")
    print(f"  3. supported_activations() - Get list of supported activations")
    print(f"  4. get_defaults(name) - Get default parameters\n")
    
    # Example: Get defaults for different activations
    print("Default parameters for each activation:\n")
    
    activations = ['relu', 'leaky_relu', 'elu', 'prelu']
    
    for activation_name in activations:
        defaults = factory.get_defaults(activation_name)
        if defaults:
            print(f"  {activation_name:12}: {defaults}")
        else:
            print(f"  {activation_name:12}: {{}}")
    
    print()


def example_9_gradient_flow():
    """Example 9: Verify gradient flow through activations.
    
    Demonstrates that gradients properly flow through activation functions.
    """
    print("=" * 70)
    print("Example 9: Gradient Flow Through Activations")
    print("=" * 70)
    
    activations = ['relu', 'leaky_relu', 'gelu', 'silu', 'sigmoid', 'tanh']
    
    print("Testing gradient flow for different activations:\n")
    
    for activation_name in activations:
        # Create input with gradient tracking
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        
        # Forward pass
        activation = get_activation(activation_name)
        output = activation(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = x.grad is not None
        grad_mean = x.grad.abs().mean().item() if has_grad else 0
        grad_zero = torch.allclose(x.grad, torch.zeros_like(x.grad)) if has_grad else True
        
        status = "✓" if has_grad and not grad_zero else "✗"
        print(f"  {activation_name:12} {status} Gradient mean: {grad_mean:.6f}")
    
    print()


def example_10_activation_statistics():
    """Example 10: Analyze activation function behavior.
    
    Compares various statistics of different activation functions.
    """
    print("=" * 70)
    print("Example 10: Activation Function Statistics")
    print("=" * 70)
    
    # Create larger input for better statistics
    x = torch.randn(100, 64, 32, 32)
    
    print("Input statistics:")
    print(f"  Mean:  {x.mean():.6f}")
    print(f"  Std:   {x.std():.6f}")
    print(f"  Min:   {x.min():.6f}")
    print(f"  Max:   {x.max():.6f}\n")
    
    activations = [
        'relu',
        'leaky_relu',
        'gelu',
        'silu',
        'sigmoid',
        'tanh',
        'elu',
        'selu',
    ]
    
    print("Output statistics after activation:\n")
    print(f"{'Activation':12} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 65)
    
    for activation_name in activations:
        activation = get_activation(activation_name)
        output = activation(x)
        
        print(f"{activation_name:12} | {output.mean():8.4f} | {output.std():8.4f} | "
              f"{output.min():8.4f} | {output.max():8.4f}")
    
    print()


if __name__ == '__main__':
    # Run all examples
    example_1_basic_activation_creation()
    example_2_custom_activation_parameters()
    example_3_case_insensitive_names()
    example_4_check_supported_activations()
    example_5_activation_in_sequential()
    example_6_custom_module_with_activation()
    example_7_compare_activations()
    example_8_activation_factory()
    example_9_gradient_flow()
    example_10_activation_statistics()
    
    print("=" * 70)
    print("All examples completed successfully! ✓")
    print("=" * 70)
