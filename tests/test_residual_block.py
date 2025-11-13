"""Tests for ResidualBlock."""

import pytest
import torch
import torch.nn as nn
from torchvision_customizer.blocks import ResidualBlock


class TestResidualBlockInitialization:
    """Test ResidualBlock initialization."""

    def test_basic_residual_block_initialization(self) -> None:
        """Test basic residual block initialization."""
        block = ResidualBlock(in_channels=64, out_channels=64)
        assert isinstance(block, nn.Module)
        assert block.stride == 1
        assert block.downsample_layer is None

    def test_residual_block_with_stride(self) -> None:
        """Test residual block with stride."""
        block = ResidualBlock(in_channels=64, out_channels=128, stride=2, downsample=True)
        assert block.stride == 2
        assert block.downsample_layer is not None

    def test_bottleneck_residual_block(self) -> None:
        """Test bottleneck residual block."""
        block = ResidualBlock(in_channels=64, out_channels=256, bottleneck=True)
        assert hasattr(block, 'conv3')
        assert hasattr(block, 'bn3')

    def test_residual_block_with_se(self) -> None:
        """Test residual block with SE block."""
        block = ResidualBlock(in_channels=64, out_channels=64, use_se=True)
        assert block.use_se is True
        assert hasattr(block, 'se_block')

    def test_different_activations(self) -> None:
        """Test residual block with different activations."""
        activations = ['relu', 'leaky_relu', 'gelu', 'silu']
        for activation in activations:
            block = ResidualBlock(in_channels=64, out_channels=64, activation=activation)
            assert block.activation_fn is not None

    def test_different_normalizations(self) -> None:
        """Test residual block with different normalizations."""
        normalizations = ['batch', 'group', 'instance', 'layer']
        for norm_type in normalizations:
            block = ResidualBlock(in_channels=64, out_channels=64, norm_type=norm_type)
            assert block.bn1 is not None


class TestResidualBlockForward:
    """Test ResidualBlock forward pass."""

    def test_basic_forward(self) -> None:
        """Test basic forward pass."""
        block = ResidualBlock(in_channels=64, out_channels=64)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        assert output.shape == torch.Size([2, 64, 32, 32])

    def test_forward_with_stride(self) -> None:
        """Test forward pass with stride."""
        block = ResidualBlock(in_channels=64, out_channels=128, stride=2, downsample=True)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        assert output.shape == torch.Size([2, 128, 16, 16])

    def test_forward_with_different_spatial_sizes(self) -> None:
        """Test forward pass with different spatial sizes."""
        block = ResidualBlock(in_channels=64, out_channels=64)
        for h, w in [(32, 32), (64, 64), (16, 16), (8, 8)]:
            x = torch.randn(2, 64, h, w)
            output = block(x)
            assert output.shape == torch.Size([2, 64, h, w])

    def test_bottleneck_forward(self) -> None:
        """Test bottleneck forward pass."""
        block = ResidualBlock(in_channels=64, out_channels=256, bottleneck=True)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        assert output.shape == torch.Size([2, 256, 32, 32])

    def test_forward_with_se_block(self) -> None:
        """Test forward pass with SE block."""
        block = ResidualBlock(in_channels=64, out_channels=64, use_se=True)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        assert output.shape == torch.Size([2, 64, 32, 32])

    def test_forward_with_batch_size_1(self) -> None:
        """Test forward pass with batch size 1."""
        block = ResidualBlock(in_channels=64, out_channels=64)
        x = torch.randn(1, 64, 32, 32)
        output = block(x)
        assert output.shape == torch.Size([1, 64, 32, 32])

    def test_forward_with_large_batch(self) -> None:
        """Test forward pass with large batch."""
        block = ResidualBlock(in_channels=64, out_channels=64)
        x = torch.randn(32, 64, 32, 32)
        output = block(x)
        assert output.shape == torch.Size([32, 64, 32, 32])


class TestResidualBlockGradient:
    """Test gradient flow through ResidualBlock."""

    def test_gradient_flow_basic(self) -> None:
        """Test gradient flow in basic block."""
        block = ResidualBlock(in_channels=64, out_channels=64)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flow_with_stride(self) -> None:
        """Test gradient flow with stride."""
        block = ResidualBlock(in_channels=64, out_channels=128, stride=2, downsample=True)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_gradient_flow_with_se(self) -> None:
        """Test gradient flow with SE block."""
        block = ResidualBlock(in_channels=64, out_channels=64, use_se=True)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_gradient_flow_bottleneck(self) -> None:
        """Test gradient flow in bottleneck."""
        block = ResidualBlock(in_channels=64, out_channels=256, bottleneck=True)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestResidualBlockParameterCounts:
    """Test parameter counts for ResidualBlock."""

    def test_parameter_count_basic(self) -> None:
        """Test parameter count for basic block."""
        block = ResidualBlock(in_channels=64, out_channels=64)
        params = sum(p.numel() for p in block.parameters())
        assert params > 0

    def test_parameter_count_bottleneck(self) -> None:
        """Test parameter count for bottleneck block."""
        block = ResidualBlock(in_channels=64, out_channels=256, bottleneck=True)
        params = sum(p.numel() for p in block.parameters())
        assert params > 0

    def test_parameter_count_with_se(self) -> None:
        """Test parameter count with SE block."""
        block_without_se = ResidualBlock(in_channels=64, out_channels=64, use_se=False)
        block_with_se = ResidualBlock(in_channels=64, out_channels=64, use_se=True)

        params_without_se = sum(p.numel() for p in block_without_se.parameters())
        params_with_se = sum(p.numel() for p in block_with_se.parameters())

        assert params_with_se > params_without_se


class TestResidualBlockIntegration:
    """Test ResidualBlock integration with other modules."""

    def test_residual_block_in_sequential(self) -> None:
        """Test residual block in nn.Sequential."""
        model = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
        )
        x = torch.randn(2, 64, 32, 32)
        output = model(x)
        assert output.shape == torch.Size([2, 64, 32, 32])

    def test_residual_stack_with_stride(self) -> None:
        """Test stacked residual blocks with stride."""
        blocks = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=128, stride=2, downsample=True),
            ResidualBlock(in_channels=128, out_channels=128),
        )
        x = torch.randn(2, 64, 32, 32)
        output = blocks(x)
        assert output.shape == torch.Size([2, 128, 16, 16])

    def test_residual_in_custom_module(self) -> None:
        """Test residual block in custom module."""

        class CustomNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.res_block = ResidualBlock(in_channels=64, out_channels=64)
                self.linear = nn.Linear(64 * 32 * 32, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.res_block(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                return x

        model = CustomNet()
        x = torch.randn(2, 64, 32, 32)
        output = model(x)
        assert output.shape == torch.Size([2, 10])

    def test_training_mode(self) -> None:
        """Test residual block in training mode."""
        block = ResidualBlock(in_channels=64, out_channels=64, norm_type='batch')
        block.train()
        x1 = torch.randn(8, 64, 32, 32)
        x2 = torch.randn(8, 64, 32, 32)
        output1 = block(x1)
        output2 = block(x2)
        # Due to different inputs, outputs should differ
        assert not torch.allclose(output1, output2, atol=1e-5)

    def test_eval_mode(self) -> None:
        """Test residual block in eval mode."""
        block = ResidualBlock(in_channels=64, out_channels=64, norm_type='batch')
        block.eval()
        x = torch.randn(2, 64, 32, 32)
        output1 = block(x)
        output2 = block(x)
        # In eval mode, BatchNorm is deterministic
        assert torch.allclose(output1, output2, atol=1e-5)


class TestResidualBlockEdgeCases:
    """Test edge cases for ResidualBlock."""

    def test_single_channel_input(self) -> None:
        """Test with single channel input."""
        block = ResidualBlock(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 32, 32)
        output = block(x)
        assert output.shape == torch.Size([2, 1, 32, 32])

    def test_large_channel_count(self) -> None:
        """Test with large channel count."""
        block = ResidualBlock(in_channels=512, out_channels=512)
        x = torch.randn(1, 512, 16, 16)
        output = block(x)
        assert output.shape == torch.Size([1, 512, 16, 16])

    def test_combined_options(self) -> None:
        """Test with multiple options combined."""
        block = ResidualBlock(
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
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        assert output.shape == torch.Size([2, 256, 16, 16])
