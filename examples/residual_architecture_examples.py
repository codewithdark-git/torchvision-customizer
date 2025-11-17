"""Examples demonstrating residual architecture features.

Comprehensive examples showing how to use bottleneck blocks, residual stages,
sequences, and the fluent architecture builder.
"""

import torch
import torch.nn as nn
from torchvision_customizer.blocks.bottleneck_block import (
    StandardBottleneck,
    WideBottleneck,
    GroupedBottleneck,
    MultiScaleBottleneck,
    AsymmetricBottleneck,
    create_bottleneck,
)
from torchvision_customizer.blocks.residual_architecture import (
    ResidualStage,
    ResidualSequence,
    ResidualBottleneckStage,
    ResidualArchitectureBuilder,
)


def example_1_standard_bottleneck():
    """Example 1: Using standard bottleneck blocks."""
    print("\n" + "=" * 60)
    print("Example 1: Standard Bottleneck Block")
    print("=" * 60)

    # Create a standard bottleneck block
    block = StandardBottleneck(
        in_channels=256,
        out_channels=256,
        stride=1,
        expansion=4,
        activation='relu',
    )

    # Create input
    x = torch.randn(2, 256, 32, 32)
    print(f"Input shape: {x.shape}")

    # Forward pass
    output = block(x)
    print(f"Output shape: {output.shape}")

    # Count parameters
    params = sum(p.numel() for p in block.parameters())
    print(f"Total parameters: {params:,}")

    # Test with downsampling
    block_down = StandardBottleneck(
        in_channels=256,
        out_channels=512,
        stride=2,
    )
    output_down = block_down(x)
    print(f"With stride=2, output shape: {output_down.shape}")


def example_2_wide_bottleneck():
    """Example 2: Using wide bottleneck blocks."""
    print("\n" + "=" * 60)
    print("Example 2: Wide Bottleneck Block")
    print("=" * 60)

    # Create standard and wide bottlenecks
    block_std = StandardBottleneck(in_channels=256, out_channels=256)
    block_wide = WideBottleneck(in_channels=256, out_channels=256, width_multiplier=1.5)

    x = torch.randn(2, 256, 32, 32)

    # Compare parameters
    params_std = sum(p.numel() for p in block_std.parameters())
    params_wide = sum(p.numel() for p in block_wide.parameters())

    print(f"Standard bottleneck parameters: {params_std:,}")
    print(f"Wide bottleneck parameters: {params_wide:,}")
    print(f"Difference: {params_wide - params_std:,}")

    # Forward passes
    out_std = block_std(x)
    out_wide = block_wide(x)

    print(f"Standard output shape: {out_std.shape}")
    print(f"Wide output shape: {out_wide.shape}")


def example_3_grouped_bottleneck():
    """Example 3: Using grouped bottleneck blocks."""
    print("\n" + "=" * 60)
    print("Example 3: Grouped Bottleneck Block (ShuffleNet style)")
    print("=" * 60)

    # Create grouped bottlenecks with different group counts
    blocks = {
        'standard': StandardBottleneck(256, 256),
        'grouped_2': GroupedBottleneck(256, 256, num_groups=2),
        'grouped_4': GroupedBottleneck(256, 256, num_groups=4),
        'grouped_8': GroupedBottleneck(256, 256, num_groups=8),
    }

    x = torch.randn(2, 256, 32, 32)

    print("Parameter comparison:")
    for name, block in blocks.items():
        params = sum(p.numel() for p in block.parameters())
        output = block(x)
        print(f"{name:15} - Parameters: {params:8,}, Output shape: {output.shape}")


def example_4_multi_scale_bottleneck():
    """Example 4: Using multi-scale bottleneck blocks."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Scale Bottleneck Block")
    print("=" * 60)

    # Create multi-scale bottleneck with and without dilation
    block_standard = MultiScaleBottleneck(
        in_channels=256,
        out_channels=256,
        use_dilation=False,
    )

    block_dilated = MultiScaleBottleneck(
        in_channels=256,
        out_channels=256,
        use_dilation=True,
    )

    x = torch.randn(2, 256, 32, 32)

    print("Multi-scale bottleneck (standard kernels):")
    out1 = block_standard(x)
    print(f"Output shape: {out1.shape}")

    print("\nMulti-scale bottleneck (with dilation):")
    out2 = block_dilated(x)
    print(f"Output shape: {out2.shape}")

    # Compare receptive fields
    print("\nReceptive field information:")
    print("Standard (3x3 + 5x5): captures local and broader context")
    print("Dilated (3x3 + 3x3 with dilation=2): captures multi-scale features")


def example_5_asymmetric_bottleneck():
    """Example 5: Using asymmetric bottleneck blocks."""
    print("\n" + "=" * 60)
    print("Example 5: Asymmetric Bottleneck Block")
    print("=" * 60)

    # Create asymmetric bottleneck
    block = AsymmetricBottleneck(
        in_channels=256,
        out_channels=256,
        stride=1,
    )

    x = torch.randn(2, 256, 32, 32)
    output = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with stride
    block_stride = AsymmetricBottleneck(
        in_channels=256,
        out_channels=512,
        stride=2,
    )
    output_stride = block_stride(x)
    print(f"With stride=2, output shape: {output_stride.shape}")

    print("\nAsymmetric kernels (1x5 and 5x1) capture directional features efficiently")


def example_6_bottleneck_factory():
    """Example 6: Using bottleneck factory function."""
    print("\n" + "=" * 60)
    print("Example 6: Bottleneck Factory Function")
    print("=" * 60)

    bottleneck_types = [
        'standard',
        'wide',
        'grouped',
        'multi_scale',
        'asymmetric',
    ]

    x = torch.randn(2, 256, 32, 32)

    print("Creating all bottleneck types using factory:")
    for btype in bottleneck_types:
        kwargs = {}
        if btype == 'wide':
            kwargs['width_multiplier'] = 1.5
        elif btype == 'grouped':
            kwargs['num_groups'] = 4

        block = create_bottleneck(btype, 256, 256, **kwargs)
        output = block(x)
        params = sum(p.numel() for p in block.parameters())
        print(f"{btype:15} - Parameters: {params:8,}, Output: {output.shape}")


def example_7_residual_stage():
    """Example 7: Using residual stages."""
    print("\n" + "=" * 60)
    print("Example 7: Residual Stage")
    print("=" * 60)

    # Create a residual stage without bottleneck
    stage1 = ResidualStage(
        in_channels=64,
        out_channels=64,
        num_blocks=3,
        use_bottleneck=False,
        activation='relu',
    )

    # Create a residual stage with bottleneck
    stage2 = ResidualStage(
        in_channels=64,
        out_channels=128,
        num_blocks=3,
        stride=2,
        use_bottleneck=True,
        bottleneck_type='standard',
    )

    x = torch.randn(2, 64, 32, 32)

    print("Stage 1 (no bottleneck, identity):")
    out1 = stage1(x)
    print(f"Output shape: {out1.shape}")
    params1 = sum(p.numel() for p in stage1.parameters())
    print(f"Parameters: {params1:,}")

    print("\nStage 2 (with bottleneck, stride=2):")
    out2 = stage2(x)
    print(f"Output shape: {out2.shape}")
    params2 = sum(p.numel() for p in stage2.parameters())
    print(f"Parameters: {params2:,}")


def example_8_residual_sequence():
    """Example 8: Using residual sequence (full network)."""
    print("\n" + "=" * 60)
    print("Example 8: Residual Sequence (Full Network)")
    print("=" * 60)

    # Create a residual sequence (like ResNet)
    sequence = ResidualSequence(
        in_channels=3,
        num_stages=4,
        blocks_per_stage=3,
        channels_per_stage=[64, 128, 256, 512],
        stride_schedule='later',
        use_bottleneck=True,
    )

    x = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {x.shape}")

    output = sequence(x)
    print(f"Output shape: {output.shape}")
    print(f"Output channels: {sequence.out_channels}")

    # Count total parameters
    total_params = sum(p.numel() for p in sequence.parameters())
    print(f"Total parameters: {total_params:,}")

    # Show memory usage
    total_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    print(f"Approximate memory: {total_memory:.2f} MB")


def example_9_mixed_bottleneck_stage():
    """Example 9: Using mixed bottleneck types in a stage."""
    print("\n" + "=" * 60)
    print("Example 9: Mixed Bottleneck Stage")
    print("=" * 60)

    # Create a stage with different bottleneck types
    stage = ResidualBottleneckStage(
        in_channels=64,
        out_channels=128,
        num_blocks=4,
        bottleneck_types=['standard', 'wide', 'grouped', 'multi_scale'],
        stride=2,
    )

    x = torch.randn(2, 64, 32, 32)
    output = stage(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("\nStage composition:")
    print("Block 1: Standard bottleneck (baseline)")
    print("Block 2: Wide bottleneck (increased capacity)")
    print("Block 3: Grouped bottleneck (efficient)")
    print("Block 4: Multi-scale bottleneck (multi-receptive-field)")


def example_10_architecture_builder_basic():
    """Example 10: Basic architecture builder."""
    print("\n" + "=" * 60)
    print("Example 10: Architecture Builder (Basic)")
    print("=" * 60)

    # Build a simple residual network
    model = (
        ResidualArchitectureBuilder()
        .add_initial_conv(3, 64, kernel_size=7, stride=2, padding=3)
        .add_max_pool(kernel_size=3, stride=2, padding=1)
        .add_stage(None, 64, 3, use_bottleneck=False)
        .add_stage(None, 128, 4, stride=2, use_bottleneck=True)
        .build()
    )

    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


def example_11_architecture_builder_full():
    """Example 11: Full ResNet-like architecture using builder."""
    print("\n" + "=" * 60)
    print("Example 11: Architecture Builder (Full ResNet-50 style)")
    print("=" * 60)

    # Build ResNet-50-like architecture
    model = (
        ResidualArchitectureBuilder()
        .add_initial_conv(3, 64, kernel_size=7, stride=2, padding=3)
        .add_max_pool(kernel_size=3, stride=2, padding=1)
        .add_stage(None, 256, 3, stride=1, use_bottleneck=True, expansion=4)
        .add_stage(None, 512, 4, stride=2, use_bottleneck=True, expansion=4)
        .add_stage(None, 1024, 6, stride=2, use_bottleneck=True, expansion=4)
        .add_stage(None, 2048, 3, stride=2, use_bottleneck=True, expansion=4)
        .add_global_avg_pool()
        .build()
    )

    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Memory estimate
    memory_mb = (total_params * 4) / (1024 ** 2)
    print(f"Approximate model size: {memory_mb:.2f} MB")


def example_12_architecture_builder_custom():
    """Example 12: Custom architecture with mixed features."""
    print("\n" + "=" * 60)
    print("Example 12: Custom Architecture with Mixed Features")
    print("=" * 60)

    # Build custom architecture with mixed bottleneck types
    model = (
        ResidualArchitectureBuilder()
        .add_initial_conv(3, 32, kernel_size=3, stride=1, padding=1)
        .add_stage(None, 64, 2, use_bottleneck=False)
        .add_mixed_bottleneck_stage(
            None,
            128,
            3,
            ['standard', 'wide', 'grouped'],
            stride=2,
        )
        .add_stage(None, 256, 2, stride=2, use_bottleneck=True, use_se=True)
        .add_global_avg_pool()
        .build()
    )

    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print("\nGradient flow: OK (model supports backpropagation)")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RESIDUAL ARCHITECTURE EXAMPLES")
    print("=" * 60)

    example_1_standard_bottleneck()
    example_2_wide_bottleneck()
    example_3_grouped_bottleneck()
    example_4_multi_scale_bottleneck()
    example_5_asymmetric_bottleneck()
    example_6_bottleneck_factory()
    example_7_residual_stage()
    example_8_residual_sequence()
    example_9_mixed_bottleneck_stage()
    example_10_architecture_builder_basic()
    example_11_architecture_builder_full()
    example_12_architecture_builder_custom()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()
