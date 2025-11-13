"""Example usage of ConvBlock - Basic convolutional building block."""

import torch
from torchvision_customizer.blocks import ConvBlock


def example_1_basic_conv_block():
    """Example 1: Create and use a basic convolutional block."""
    print("=" * 80)
    print("Example 1: Basic Convolutional Block")
    print("=" * 80)

    # Create a simple convolutional block
    block = ConvBlock(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        activation="relu",
        use_batchnorm=True,
    )

    print(f"\nBlock Configuration:\n{block}")

    # Create sample input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape: {input_tensor.shape}")

    # Forward pass
    output = block(input_tensor)
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def example_2_conv_with_pooling():
    """Example 2: Convolutional block with pooling."""
    print("\n" + "=" * 80)
    print("Example 2: Convolutional Block with Pooling")
    print("=" * 80)

    block = ConvBlock(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        activation="relu",
        use_batchnorm=True,
        dropout_rate=0.1,
        pooling_type="max",
        pooling_kernel_size=2,
        pooling_stride=2,
    )

    print(f"\nBlock Configuration:\n{block}")

    input_tensor = torch.randn(4, 3, 224, 224)
    print(f"\nInput shape: {input_tensor.shape}")

    output = block(input_tensor)
    print(f"Output shape: {output.shape}")

    # Calculate expected output shape
    out_h, out_w = block.calculate_output_shape(224, 224)
    print(f"Calculated output spatial dims: {out_h} x {out_w}")
    print(f"Output channels: {block.get_output_channels}")


def example_3_different_activations():
    """Example 3: ConvBlock with different activation functions."""
    print("\n" + "=" * 80)
    print("Example 3: Different Activation Functions")
    print("=" * 80)

    activations = ["relu", "leaky_relu", "gelu", "elu", "sigmoid"]

    input_tensor = torch.randn(2, 3, 64, 64)

    for activation in activations:
        block = ConvBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            activation=activation,
            use_batchnorm=False,
        )

        output = block(input_tensor)
        print(f"\nActivation: {activation:15s} | Output shape: {output.shape}")


def example_4_stride_and_padding():
    """Example 4: ConvBlock with different stride and padding configurations."""
    print("\n" + "=" * 80)
    print("Example 4: Stride and Padding Configurations")
    print("=" * 80)

    configs = [
        {"stride": 1, "padding": 1, "kernel_size": 3},  # Keep spatial dimensions
        {"stride": 2, "padding": 1, "kernel_size": 3},  # Downsample by 2
        {"stride": 1, "padding": 0, "kernel_size": 3},  # Reduce spatial dimensions
    ]

    input_tensor = torch.randn(1, 3, 64, 64)
    print(f"\nInput shape: {input_tensor.shape}\n")

    for i, config in enumerate(configs, 1):
        block = ConvBlock(
            in_channels=3,
            out_channels=32,
            **config,
            activation="relu",
            use_batchnorm=True,
        )

        output = block(input_tensor)
        print(
            f"Config {i}: stride={config['stride']}, padding={config['padding']}, "
            f"kernel_size={config['kernel_size']} -> Output: {output.shape}"
        )


def example_5_stacking_blocks():
    """Example 5: Stack multiple ConvBlocks to create a simple CNN."""
    print("\n" + "=" * 80)
    print("Example 5: Stacking Multiple ConvBlocks")
    print("=" * 80)

    class SimpleCNN(torch.nn.Module):
        """Simple CNN made from stacked ConvBlocks."""

        def __init__(self):
            super().__init__()
            self.block1 = ConvBlock(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                activation="relu",
                use_batchnorm=True,
                pooling_type="max",
            )
            self.block2 = ConvBlock(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                activation="relu",
                use_batchnorm=True,
                pooling_type="max",
            )
            self.block3 = ConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                activation="relu",
                use_batchnorm=True,
                pooling_type="max",
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x

    model = SimpleCNN()
    print(f"\nModel:\n{model}")

    input_tensor = torch.randn(2, 3, 256, 256)
    print(f"\nInput shape: {input_tensor.shape}")

    output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


def example_6_dropout_effect():
    """Example 6: Demonstrate dropout in training vs evaluation mode."""
    print("\n" + "=" * 80)
    print("Example 6: Dropout Effect (Training vs Evaluation)")
    print("=" * 80)

    block = ConvBlock(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        padding=1,
        activation="relu",
        use_batchnorm=True,
        dropout_rate=0.5,  # High dropout for demonstration
    )

    input_tensor = torch.randn(1, 3, 32, 32)

    # Training mode
    block.train()
    with torch.no_grad():
        output_train = block(input_tensor)
    print(f"\nTraining mode output mean: {output_train.mean():.4f}")
    print(f"Training mode output std:  {output_train.std():.4f}")

    # Evaluation mode
    block.eval()
    with torch.no_grad():
        output_eval = block(input_tensor)
    print(f"\nEvaluation mode output mean: {output_eval.mean():.4f}")
    print(f"Evaluation mode output std:  {output_eval.std():.4f}")


def example_7_output_shape_calculation():
    """Example 7: Calculate output shapes for various configurations."""
    print("\n" + "=" * 80)
    print("Example 7: Output Shape Calculation")
    print("=" * 80)

    block = ConvBlock(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        activation="relu",
        use_batchnorm=True,
        pooling_type="max",
        pooling_kernel_size=2,
        pooling_stride=2,
    )

    input_sizes = [(224, 224), (256, 256), (512, 512), (64, 64)]

    print("\nInput Size -> Output Size (after conv + pooling)")
    print("-" * 50)
    for h, w in input_sizes:
        out_h, out_w = block.calculate_output_shape(h, w)
        print(f"({h:3d}, {w:3d}) -> ({out_h:3d}, {out_w:3d})")


if __name__ == "__main__":
    """Run all examples."""
    example_1_basic_conv_block()
    example_2_conv_with_pooling()
    example_3_different_activations()
    example_4_stride_and_padding()
    example_5_stacking_blocks()
    example_6_dropout_effect()
    example_7_output_shape_calculation()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
