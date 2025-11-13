"""Unit tests for ConvBlock."""

import pytest
import torch
from torch import nn, Tensor

from torchvision_customizer.blocks import ConvBlock


class TestConvBlockInitialization:
    """Test ConvBlock initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic ConvBlock initialization with default parameters."""
        block = ConvBlock(in_channels=3, out_channels=64)
        assert block.in_channels == 3
        assert block.out_channels == 64
        assert block.get_output_channels == 64

    def test_custom_kernel_size(self):
        """Test ConvBlock with custom kernel sizes."""
        block = ConvBlock(in_channels=3, out_channels=64, kernel_size=5)
        assert block.kernel_size == (5, 5)

    def test_custom_stride(self):
        """Test ConvBlock with custom stride."""
        block = ConvBlock(in_channels=3, out_channels=64, stride=2)
        assert block.stride == (2, 2)

    def test_tuple_parameters(self):
        """Test ConvBlock with tuple parameters."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 5),
            stride=(1, 2),
            padding=(1, 2),
        )
        assert block.kernel_size == (3, 5)
        assert block.stride == (1, 2)
        assert block.padding == (1, 2)

    def test_invalid_in_channels(self):
        """Test that invalid in_channels raises ValueError."""
        with pytest.raises(ValueError):
            ConvBlock(in_channels=0, out_channels=64)

        with pytest.raises(ValueError):
            ConvBlock(in_channels=-3, out_channels=64)

    def test_invalid_out_channels(self):
        """Test that invalid out_channels raises ValueError."""
        with pytest.raises(ValueError):
            ConvBlock(in_channels=3, out_channels=0)

        with pytest.raises(ValueError):
            ConvBlock(in_channels=3, out_channels=-64)

    def test_invalid_dropout_rate(self):
        """Test that invalid dropout_rate raises ValueError."""
        with pytest.raises(ValueError):
            ConvBlock(in_channels=3, out_channels=64, dropout_rate=-0.1)

        with pytest.raises(ValueError):
            ConvBlock(in_channels=3, out_channels=64, dropout_rate=1.5)

    def test_invalid_activation(self):
        """Test that invalid activation raises ValueError."""
        block = ConvBlock(in_channels=3, out_channels=64, activation="relu")
        with pytest.raises(ValueError):
            block._get_activation("invalid_activation")

    def test_invalid_pooling_type(self):
        """Test that invalid pooling type raises ValueError."""
        block = ConvBlock(in_channels=3, out_channels=64)
        with pytest.raises(ValueError):
            block._get_pooling("invalid_pool", (2, 2), (2, 2))

    def test_all_activation_functions(self):
        """Test that all supported activation functions can be initialized."""
        activations = ["relu", "leaky_relu", "gelu", "elu", "selu", "sigmoid", "tanh"]
        for activation in activations:
            block = ConvBlock(
                in_channels=3,
                out_channels=64,
                activation=activation,
            )
            assert block.activation_type == activation
            assert block.activation is not None

    def test_no_activation(self):
        """Test ConvBlock with no activation function."""
        block = ConvBlock(in_channels=3, out_channels=64, activation=None)
        assert block.activation is None

    def test_all_pooling_types(self):
        """Test that all supported pooling types can be initialized."""
        pooling_types = ["max", "avg", "adaptive_avg"]
        for pooling_type in pooling_types:
            block = ConvBlock(
                in_channels=3,
                out_channels=64,
                pooling_type=pooling_type,
            )
            assert block.pooling_type == pooling_type
            assert block.pooling is not None

    def test_no_pooling(self):
        """Test ConvBlock with no pooling."""
        block = ConvBlock(in_channels=3, out_channels=64, pooling_type=None)
        assert block.pooling is None

    def test_batchnorm_disabled(self):
        """Test ConvBlock with batch normalization disabled."""
        block = ConvBlock(in_channels=3, out_channels=64, use_batchnorm=False)
        assert block.bn is None

    def test_dropout_disabled(self):
        """Test ConvBlock with dropout disabled."""
        block = ConvBlock(in_channels=3, out_channels=64, dropout_rate=0.0)
        assert block.dropout is None


class TestConvBlockForwardPass:
    """Test ConvBlock forward pass functionality."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        block = ConvBlock(in_channels=3, out_channels=64, activation="relu")
        input_tensor = torch.randn(4, 3, 224, 224)
        output = block(input_tensor)

        assert output.shape[0] == 4  # Batch size preserved
        assert output.shape[1] == 64  # Output channels
        assert output.ndim == 4  # Still 4D tensor

    def test_forward_with_pooling(self):
        """Test forward pass with pooling."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            activation="relu",
            pooling_type="max",
            pooling_kernel_size=2,
            pooling_stride=2,
        )
        input_tensor = torch.randn(2, 3, 224, 224)
        output = block(input_tensor)

        # With padding=1, stride=1, and max pooling with kernel=2, stride=2
        # Output should be half the input spatial dimensions
        assert output.shape == (2, 64, 112, 112)

    def test_forward_with_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        block = ConvBlock(in_channels=3, out_channels=64)
        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 32, 32)
            output = block(input_tensor)
            assert output.shape[0] == batch_size

    def test_forward_with_different_input_sizes(self):
        """Test forward pass with different input spatial dimensions."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=1,
            activation="relu",
        )
        input_sizes = [(32, 32), (64, 64), (128, 128), (224, 224), (256, 256)]

        for h, w in input_sizes:
            input_tensor = torch.randn(1, 3, h, w)
            output = block(input_tensor)
            # With padding=1 and stride=1, spatial dimensions should be preserved
            assert output.shape == (1, 64, h, w)

    def test_forward_with_dropout_train_mode(self):
        """Test forward pass with dropout in training mode."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            dropout_rate=0.5,
        )
        block.train()

        input_tensor = torch.randn(4, 3, 32, 32)
        output = block(input_tensor)

        assert output.shape[0] == 4
        assert output.shape[1] == 64

    def test_forward_with_dropout_eval_mode(self):
        """Test forward pass with dropout in evaluation mode."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            dropout_rate=0.5,
        )
        block.eval()

        input_tensor = torch.randn(4, 3, 32, 32)
        output = block(input_tensor)

        assert output.shape[0] == 4
        assert output.shape[1] == 64

    def test_forward_multiple_times(self):
        """Test that forward pass is consistent."""
        block = ConvBlock(in_channels=3, out_channels=64, activation="relu")
        input_tensor = torch.randn(2, 3, 32, 32)

        block.eval()
        with torch.no_grad():
            output1 = block(input_tensor)
            output2 = block(input_tensor)

        # In eval mode with same input, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_forward_gradient_flow(self):
        """Test that gradients flow correctly through the block."""
        block = ConvBlock(in_channels=3, out_channels=64, activation="relu")
        input_tensor = torch.randn(2, 3, 32, 32, requires_grad=True)

        output = block(input_tensor)
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        for param in block.parameters():
            assert param.grad is not None

    def test_forward_stride_downsampling(self):
        """Test forward pass with stride for downsampling."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            activation="relu",
        )
        input_tensor = torch.randn(2, 3, 64, 64)
        output = block(input_tensor)

        # With stride=2, spatial dimensions should be halved
        assert output.shape == (2, 64, 32, 32)

    def test_forward_no_padding(self):
        """Test forward pass with no padding (spatial reduction)."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            activation="relu",
        )
        input_tensor = torch.randn(2, 3, 32, 32)
        output = block(input_tensor)

        # With kernel_size=3, stride=1, padding=0: out = 32 - 3 + 1 = 30
        assert output.shape == (2, 64, 30, 30)


class TestConvBlockOutputShape:
    """Test output shape calculation methods."""

    def test_calculate_output_shape_basic(self):
        """Test basic output shape calculation."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            pooling_type=None,
        )
        out_h, out_w = block.calculate_output_shape(224, 224)
        # With kernel=3, stride=1, padding=1: output = input
        assert (out_h, out_w) == (224, 224)

    def test_calculate_output_shape_with_pooling(self):
        """Test output shape calculation with pooling."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            pooling_type="max",
            pooling_kernel_size=2,
            pooling_stride=2,
        )
        out_h, out_w = block.calculate_output_shape(224, 224)
        # After conv: 224x224, after pooling: 112x112
        assert (out_h, out_w) == (112, 112)

    def test_calculate_output_shape_stride(self):
        """Test output shape calculation with stride."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            pooling_type=None,
        )
        out_h, out_w = block.calculate_output_shape(224, 224)
        # With stride=2: output = 224 / 2 = 112
        assert (out_h, out_w) == (112, 112)

    def test_calculate_output_shape_no_padding(self):
        """Test output shape calculation with no padding."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            pooling_type=None,
        )
        out_h, out_w = block.calculate_output_shape(32, 32)
        # With kernel=3, stride=1, padding=0: output = 32 - 3 + 1 = 30
        assert (out_h, out_w) == (30, 30)

    def test_calculate_output_shape_multiple_inputs(self):
        """Test output shape calculation for multiple input sizes."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            pooling_type="max",
            pooling_kernel_size=2,
            pooling_stride=2,
        )
        
        test_cases = [
            ((256, 256), (128, 128)),
            ((512, 512), (256, 256)),
            ((64, 64), (32, 32)),
        ]
        
        for (h_in, w_in), (h_out, w_out) in test_cases:
            out_h, out_w = block.calculate_output_shape(h_in, w_in)
            assert (out_h, out_w) == (h_out, w_out)


class TestConvBlockProperties:
    """Test ConvBlock properties and utility methods."""

    def test_get_output_channels(self):
        """Test get_output_channels property."""
        block = ConvBlock(in_channels=3, out_channels=128)
        assert block.get_output_channels == 128

    def test_repr_string(self):
        """Test __repr__ method."""
        block = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            activation="relu",
        )
        repr_str = repr(block)
        assert "ConvBlock" in repr_str
        assert "in_channels=3" in repr_str
        assert "out_channels=64" in repr_str

    def test_module_parameters(self):
        """Test that ConvBlock has learnable parameters."""
        block = ConvBlock(in_channels=3, out_channels=64)
        params = list(block.parameters())
        
        # Should have conv weight and bias
        assert len(params) >= 2  # At least conv weight and bias
        
        # All parameters should be learnable
        for param in params:
            assert param.requires_grad

    def test_module_device_movement(self):
        """Test that ConvBlock can be moved between devices."""
        block = ConvBlock(in_channels=3, out_channels=64)
        
        # Should be on CPU by default
        assert next(block.parameters()).device.type == "cpu"
        
        # Move to CPU explicitly
        block.cpu()
        assert next(block.parameters()).device.type == "cpu"


class TestConvBlockIntegration:
    """Integration tests for ConvBlock in realistic scenarios."""

    def test_sequential_blocks(self):
        """Test stacking multiple ConvBlocks in sequence."""
        block1 = ConvBlock(in_channels=3, out_channels=32, padding=1)
        block2 = ConvBlock(in_channels=32, out_channels=64, padding=1)
        block3 = ConvBlock(in_channels=64, out_channels=128, padding=1)

        input_tensor = torch.randn(2, 3, 64, 64)
        x = block1(input_tensor)
        assert x.shape == (2, 32, 64, 64)
        
        x = block2(x)
        assert x.shape == (2, 64, 64, 64)
        
        x = block3(x)
        assert x.shape == (2, 128, 64, 64)

    def test_in_sequential_module(self):
        """Test ConvBlock within nn.Sequential."""
        model = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, padding=1),
            ConvBlock(in_channels=32, out_channels=64, padding=1),
            ConvBlock(in_channels=64, out_channels=128, padding=1),
        )

        input_tensor = torch.randn(2, 3, 64, 64)
        output = model(input_tensor)
        assert output.shape == (2, 128, 64, 64)


# Pytest fixtures for common test data
@pytest.fixture
def standard_input():
    """Fixture providing standard test input."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def conv_block():
    """Fixture providing a standard ConvBlock."""
    return ConvBlock(in_channels=3, out_channels=64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
