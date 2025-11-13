"""Examples demonstrating ResidualBlock usage."""

import torch
import torch.nn as nn
from torchvision_customizer.blocks import ResidualBlock


def example_basic_residual_block() -> None:
    """Example 1: Basic residual block with default settings."""
    print("=" * 70)
    print("Example 1: Basic Residual Block")
    print("=" * 70)

    # Create basic residual block
    res_block = ResidualBlock(in_channels=64, out_channels=64)
    print(f"ResidualBlock created:")
    print(f"  Input channels: 64")
    print(f"  Output channels: 64")
    print(f"  Stride: 1")

    # Create input tensor
    x = torch.randn(2, 64, 32, 32)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = res_block(x)
    print(f"Output shape: {output.shape}")

    # Verify skip connection (output should have residual component)
    print(f"\nOutput has residual connection from input")
    print()


def example_residual_block_with_stride() -> None:
    """Example 2: Residual block with stride for downsampling."""
    print("=" * 70)
    print("Example 2: Residual Block with Stride")
    print("=" * 70)

    # Create residual block with stride
    res_block = ResidualBlock(
        in_channels=64, out_channels=128, stride=2, downsample=True
    )
    print(f"ResidualBlock created:")
    print(f"  Input channels: 64")
    print(f"  Output channels: 128")
    print(f"  Stride: 2 (downsampling)")
    print(f"  Downsampling enabled: True")

    # Create input tensor
    x = torch.randn(2, 64, 32, 32)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = res_block(x)
    print(f"Output shape: {output.shape}")
    print(f"\nNote: Spatial dimensions reduced from 32x32 to 16x16")
    print(f"Channel dimensions increased from 64 to 128")
    print()


def example_bottleneck_residual_block() -> None:
    """Example 3: Bottleneck residual block for parameter efficiency."""
    print("=" * 70)
    print("Example 3: Bottleneck Residual Block")
    print("=" * 70)

    # Create bottleneck block
    res_block = ResidualBlock(
        in_channels=64, out_channels=256, bottleneck=True
    )
    print(f"ResidualBlock (Bottleneck) created:")
    print(f"  Input channels: 64")
    print(f"  Output channels: 256")
    print(f"  Architecture: 1x1 → 3x3 → 1x1 (reduces intermediate channels)")

    # Create input tensor
    x = torch.randn(2, 64, 32, 32)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = res_block(x)
    print(f"Output shape: {output.shape}")

    # Compare parameter counts
    non_bottleneck = ResidualBlock(in_channels=64, out_channels=256, bottleneck=False)
    bottleneck = ResidualBlock(in_channels=64, out_channels=256, bottleneck=True)

    non_bottleneck_params = sum(p.numel() for p in non_bottleneck.parameters())
    bottleneck_params = sum(p.numel() for p in bottleneck.parameters())

    print(f"\nParameter Comparison:")
    print(f"  Non-bottleneck: {non_bottleneck_params:,} parameters")
    print(f"  Bottleneck: {bottleneck_params:,} parameters")
    print(f"  Reduction: {(1 - bottleneck_params/non_bottleneck_params)*100:.1f}%")
    print()


def example_residual_block_with_se() -> None:
    """Example 4: Residual block with Squeeze-and-Excitation (SE) block."""
    print("=" * 70)
    print("Example 4: Residual Block with SE Block")
    print("=" * 70)

    # Create residual block with SE block
    res_block = ResidualBlock(
        in_channels=64, out_channels=64, use_se=True, se_reduction=16
    )
    print(f"ResidualBlock with SE Block created:")
    print(f"  Input channels: 64")
    print(f"  Output channels: 64")
    print(f"  SE Block enabled: True")
    print(f"  SE Reduction: 16")

    # Create input tensor
    x = torch.randn(2, 64, 32, 32)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = res_block(x)
    print(f"Output shape: {output.shape}")

    # Compare parameter counts
    without_se = ResidualBlock(in_channels=64, out_channels=64, use_se=False)
    with_se = ResidualBlock(in_channels=64, out_channels=64, use_se=True)

    params_without_se = sum(p.numel() for p in without_se.parameters())
    params_with_se = sum(p.numel() for p in with_se.parameters())

    print(f"\nParameter Impact:")
    print(f"  Without SE: {params_without_se:,} parameters")
    print(f"  With SE: {params_with_se:,} parameters")
    print(f"  Added: {params_with_se - params_without_se:,} parameters")
    print()


def example_different_activations() -> None:
    """Example 5: Residual blocks with different activation functions."""
    print("=" * 70)
    print("Example 5: Different Activation Functions")
    print("=" * 70)

    activations = ['relu', 'leaky_relu', 'gelu', 'silu', 'tanh']
    x = torch.randn(2, 64, 32, 32)

    print(f"Testing residual block with different activations:")
    print(f"Input shape: {x.shape}\n")

    for activation in activations:
        res_block = ResidualBlock(
            in_channels=64, out_channels=64, activation=activation
        )
        output = res_block(x)

        # Calculate statistics
        output_mean = output.mean().item()
        output_std = output.std().item()

        print(f"{activation:15} → Output mean: {output_mean:7.4f}, std: {output_std:7.4f}")

    print()


def example_different_normalizations() -> None:
    """Example 6: Residual blocks with different normalization layers."""
    print("=" * 70)
    print("Example 6: Different Normalization Layers")
    print("=" * 70)

    normalizations = ['batch', 'group', 'instance', 'layer']
    x = torch.randn(2, 64, 32, 32)

    print(f"Testing residual block with different normalizations:")
    print(f"Input shape: {x.shape}\n")

    for norm_type in normalizations:
        try:
            res_block = ResidualBlock(
                in_channels=64, out_channels=64, norm_type=norm_type
            )
            output = res_block(x)

            # Calculate statistics
            output_mean = output.mean().item()
            output_std = output.std().item()

            print(f"{norm_type:15} → Output mean: {output_mean:7.4f}, std: {output_std:7.4f}")
        except Exception as e:
            print(f"{norm_type:15} → Error: {str(e)[:40]}")

    print()


def example_stacked_residual_blocks() -> None:
    """Example 7: Stacking multiple residual blocks."""
    print("=" * 70)
    print("Example 7: Stacked Residual Blocks")
    print("=" * 70)

    # Build a simple residual network
    network = nn.Sequential(
        ResidualBlock(in_channels=3, out_channels=64),
        ResidualBlock(in_channels=64, out_channels=64),
        ResidualBlock(in_channels=64, out_channels=128, stride=2, downsample=True),
        ResidualBlock(in_channels=128, out_channels=128),
        ResidualBlock(in_channels=128, out_channels=256, stride=2, downsample=True),
        ResidualBlock(in_channels=256, out_channels=256),
    )

    print("Network Architecture:")
    for i, layer in enumerate(network, 1):
        print(f"  Block {i}: {layer}")

    # Create input (like ImageNet images)
    x = torch.randn(2, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")

    # Forward pass through network
    output = network(x)
    print(f"Output shape: {output.shape}")
    print(f"\nSpatial dimensions: 224 → 112 → 56 (after 2 downsampling blocks)")
    print()


def example_residual_block_training() -> None:
    """Example 8: Training a simple model with residual blocks."""
    print("=" * 70)
    print("Example 8: Training with Residual Blocks")
    print("=" * 70)

    # Simple model with residual blocks
    class SimpleResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                ResidualBlock(in_channels=3, out_channels=64),
                ResidualBlock(in_channels=64, out_channels=128, stride=2, downsample=True),
                ResidualBlock(in_channels=128, out_channels=128),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleResNet()
    model.train()

    print("SimpleResNet created:")
    print(f"  Input: (B, 3, 224, 224)")
    print(f"  Output: (B, 10)")

    # Create dummy input and target
    x = torch.randn(4, 3, 224, 224)
    target = torch.randint(0, 10, (4,))

    print(f"\nTraining step:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {target.shape}")

    # Forward pass
    output = model(x)
    print(f"  Output shape: {output.shape}")

    # Calculate loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    print(f"  Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print(f"\n  Gradients computed successfully")
    print()


def example_residual_block_comparison() -> None:
    """Example 9: Comparing residual vs non-residual blocks."""
    print("=" * 70)
    print("Example 9: Residual vs Non-Residual")
    print("=" * 70)

    # Non-residual block (just convolutions)
    class SimpleConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            return x

    # Residual block
    residual_block = ResidualBlock(in_channels=64, out_channels=64)
    non_residual_block = SimpleConvBlock(in_channels=64, out_channels=64)

    x = torch.randn(2, 64, 32, 32)

    print("Architecture Comparison:")
    print(f"  Residual block parameters: {sum(p.numel() for p in residual_block.parameters()):,}")
    print(f"  Non-residual block parameters: {sum(p.numel() for p in non_residual_block.parameters()):,}")

    # Forward pass
    res_output = residual_block(x)
    non_res_output = non_residual_block(x)

    print(f"\nOutput shapes:")
    print(f"  Residual: {res_output.shape}")
    print(f"  Non-residual: {non_res_output.shape}")

    # Analyze gradient flow
    x1 = torch.randn(2, 64, 32, 32, requires_grad=True)
    residual_block.eval()
    res_output = residual_block(x1)
    res_output.sum().backward()

    x2 = torch.randn(2, 64, 32, 32, requires_grad=True)
    non_residual_block.eval()
    non_res_output = non_residual_block(x2)
    non_res_output.sum().backward()

    print(f"\nGradient statistics:")
    print(f"  Residual input gradient std: {x1.grad.std().item():.6f}")
    print(f"  Non-residual input gradient std: {x2.grad.std().item():.6f}")
    print(f"\nNote: Residual connections help preserve gradient magnitudes")
    print()


def example_advanced_residual_configuration() -> None:
    """Example 10: Advanced configuration with multiple options."""
    print("=" * 70)
    print("Example 10: Advanced Residual Block Configuration")
    print("=" * 70)

    # Create advanced configuration
    res_block = ResidualBlock(
        in_channels=64,
        out_channels=256,
        stride=2,
        downsample=True,
        activation='silu',
        norm_type='group',
        use_se=True,
        se_reduction=8,
        bottleneck=True,
    )

    print("Advanced ResidualBlock created with:")
    print(f"  Input channels: 64")
    print(f"  Output channels: 256")
    print(f"  Stride: 2 (downsampling)")
    print(f"  Activation: SiLU (Swish)")
    print(f"  Normalization: Group Norm")
    print(f"  SE Block: Enabled (reduction=8)")
    print(f"  Architecture: Bottleneck (1x1 → 3x3 → 1x1)")

    # Create input
    x = torch.randn(2, 64, 32, 32)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = res_block(x)
    print(f"Output shape: {output.shape}")

    # Count parameters
    params = sum(p.numel() for p in res_block.parameters())
    print(f"\nTotal parameters: {params:,}")

    # Benchmark forward pass
    import time

    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = res_block(x)
        elapsed = time.time() - start

    print(f"Average forward time (100 iterations): {elapsed/100*1000:.2f} ms")
    print()


if __name__ == '__main__':
    example_basic_residual_block()
    example_residual_block_with_stride()
    example_bottleneck_residual_block()
    example_residual_block_with_se()
    example_different_activations()
    example_different_normalizations()
    example_stacked_residual_blocks()
    example_residual_block_training()
    example_residual_block_comparison()
    example_advanced_residual_configuration()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
