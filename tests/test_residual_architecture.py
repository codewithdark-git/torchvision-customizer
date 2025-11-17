"""Tests for residual architecture components.

Comprehensive tests for bottleneck blocks, residual stages, sequences,
and builders with various configurations and edge cases.
"""

import pytest
import torch
import torch.nn as nn
from torchvision_customizer.blocks import ResidualBlock
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
    ResidualStageConfig,
)


class TestStandardBottleneck:
    """Tests for StandardBottleneck."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        block = StandardBottleneck(in_channels=256, out_channels=256)
        assert block is not None

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        block = StandardBottleneck(in_channels=256, out_channels=256)
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 256, 32, 32)

    def test_stride_downsampling(self) -> None:
        """Test downsampling with stride=2."""
        block = StandardBottleneck(in_channels=256, out_channels=512, stride=2)
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 512, 16, 16)

    def test_channel_expansion(self) -> None:
        """Test channel expansion."""
        block = StandardBottleneck(in_channels=64, out_channels=256)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        assert output.shape == (2, 256, 32, 32)

    def test_custom_expansion_ratio(self) -> None:
        """Test custom expansion ratio."""
        block = StandardBottleneck(in_channels=256, out_channels=512, expansion=2)
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 512, 32, 32)

    def test_gradient_flow(self) -> None:
        """Test gradient flow through block."""
        block = StandardBottleneck(in_channels=256, out_channels=256)
        x = torch.randn(2, 256, 32, 32, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_activations(self) -> None:
        """Test different activation functions."""
        for activation in ['relu', 'gelu', 'leaky_relu']:
            block = StandardBottleneck(
                in_channels=256, out_channels=256, activation=activation
            )
            x = torch.randn(2, 256, 32, 32)
            output = block(x)
            assert output.shape == (2, 256, 32, 32)

    def test_skip_connection(self) -> None:
        """Test skip connection contribution."""
        block = StandardBottleneck(in_channels=256, out_channels=256)
        x = torch.randn(2, 256, 32, 32)
        
        # Forward pass
        output = block(x)
        
        # Output should be influenced by skip connection
        # (not just zeros or random)
        assert not torch.allclose(output, torch.zeros_like(output))


class TestWideBottleneck:
    """Tests for WideBottleneck."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        block = WideBottleneck(in_channels=256, out_channels=256)
        assert block is not None

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        block = WideBottleneck(in_channels=256, out_channels=256)
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 256, 32, 32)

    def test_width_multiplier(self) -> None:
        """Test width multiplier effect."""
        block1 = WideBottleneck(in_channels=256, out_channels=256, width_multiplier=1.0)
        block2 = WideBottleneck(in_channels=256, out_channels=256, width_multiplier=1.5)
        
        # block2 should have more parameters
        params1 = sum(p.numel() for p in block1.parameters())
        params2 = sum(p.numel() for p in block2.parameters())
        assert params2 > params1

    def test_stride_and_expansion(self) -> None:
        """Test stride and expansion combined."""
        block = WideBottleneck(
            in_channels=256,
            out_channels=512,
            stride=2,
            expansion=4,
            width_multiplier=1.5,
        )
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 512, 16, 16)


class TestGroupedBottleneck:
    """Tests for GroupedBottleneck."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        block = GroupedBottleneck(in_channels=256, out_channels=256, num_groups=4)
        assert block is not None

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        block = GroupedBottleneck(in_channels=256, out_channels=256, num_groups=4)
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 256, 32, 32)

    def test_different_group_counts(self) -> None:
        """Test different numbers of groups."""
        for num_groups in [2, 4, 8]:
            block = GroupedBottleneck(
                in_channels=256, out_channels=256, num_groups=num_groups
            )
            x = torch.randn(2, 256, 32, 32)
            output = block(x)
            assert output.shape == (2, 256, 32, 32)

    def test_group_efficiency(self) -> None:
        """Test that grouped bottleneck has fewer params than standard."""
        block1 = StandardBottleneck(in_channels=256, out_channels=256)
        block2 = GroupedBottleneck(in_channels=256, out_channels=256, num_groups=8)
        
        params1 = sum(p.numel() for p in block1.parameters())
        params2 = sum(p.numel() for p in block2.parameters())
        assert params2 <= params1


class TestMultiScaleBottleneck:
    """Tests for MultiScaleBottleneck."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        block = MultiScaleBottleneck(in_channels=256, out_channels=256)
        assert block is not None

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        block = MultiScaleBottleneck(in_channels=256, out_channels=256)
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 256, 32, 32)

    def test_with_dilation(self) -> None:
        """Test multi-scale with dilation."""
        block = MultiScaleBottleneck(
            in_channels=256, out_channels=256, use_dilation=True
        )
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 256, 32, 32)

    def test_without_dilation(self) -> None:
        """Test multi-scale without dilation."""
        block = MultiScaleBottleneck(
            in_channels=256, out_channels=256, use_dilation=False
        )
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 256, 32, 32)


class TestAsymmetricBottleneck:
    """Tests for AsymmetricBottleneck."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        block = AsymmetricBottleneck(in_channels=256, out_channels=256)
        assert block is not None

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        block = AsymmetricBottleneck(in_channels=256, out_channels=256)
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 256, 32, 32)

    def test_stride_handling(self) -> None:
        """Test stride handling with asymmetric kernels."""
        block = AsymmetricBottleneck(in_channels=256, out_channels=512, stride=2)
        x = torch.randn(2, 256, 32, 32)
        output = block(x)
        assert output.shape == (2, 512, 16, 16)


class TestCreateBottleneck:
    """Tests for create_bottleneck factory function."""

    def test_standard_creation(self) -> None:
        """Test creating standard bottleneck."""
        block = create_bottleneck('standard', 256, 256)
        assert isinstance(block, StandardBottleneck)

    def test_wide_creation(self) -> None:
        """Test creating wide bottleneck."""
        block = create_bottleneck('wide', 256, 256, width_multiplier=1.5)
        assert isinstance(block, WideBottleneck)

    def test_grouped_creation(self) -> None:
        """Test creating grouped bottleneck."""
        block = create_bottleneck('grouped', 256, 256, num_groups=4)
        assert isinstance(block, GroupedBottleneck)

    def test_multi_scale_creation(self) -> None:
        """Test creating multi-scale bottleneck."""
        block = create_bottleneck('multi_scale', 256, 256)
        assert isinstance(block, MultiScaleBottleneck)

    def test_asymmetric_creation(self) -> None:
        """Test creating asymmetric bottleneck."""
        block = create_bottleneck('asymmetric', 256, 256)
        assert isinstance(block, AsymmetricBottleneck)

    def test_invalid_type(self) -> None:
        """Test invalid bottleneck type raises error."""
        with pytest.raises(ValueError):
            create_bottleneck('invalid', 256, 256)


class TestResidualStage:
    """Tests for ResidualStage."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        stage = ResidualStage(in_channels=64, out_channels=64, num_blocks=3)
        assert stage is not None

    def test_forward_shape_identity(self) -> None:
        """Test forward pass with identity dimensions."""
        stage = ResidualStage(in_channels=64, out_channels=64, num_blocks=3)
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 64, 32, 32)

    def test_forward_shape_downsample(self) -> None:
        """Test forward pass with downsampling."""
        stage = ResidualStage(
            in_channels=64, out_channels=128, num_blocks=3, stride=2
        )
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 128, 16, 16)

    def test_multiple_blocks(self) -> None:
        """Test stage with different numbers of blocks."""
        for num_blocks in [1, 2, 3, 4]:
            stage = ResidualStage(
                in_channels=64, out_channels=64, num_blocks=num_blocks
            )
            x = torch.randn(2, 64, 32, 32)
            output = stage(x)
            assert output.shape == (2, 64, 32, 32)

    def test_with_bottleneck(self) -> None:
        """Test stage with bottleneck blocks."""
        stage = ResidualStage(
            in_channels=64,
            out_channels=64,
            num_blocks=3,
            use_bottleneck=True,
        )
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 64, 32, 32)

    def test_without_bottleneck(self) -> None:
        """Test stage without bottleneck blocks."""
        stage = ResidualStage(
            in_channels=64,
            out_channels=64,
            num_blocks=3,
            use_bottleneck=False,
        )
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 64, 32, 32)

    def test_with_se_blocks(self) -> None:
        """Test stage with SE blocks."""
        stage = ResidualStage(
            in_channels=64,
            out_channels=64,
            num_blocks=3,
            use_bottleneck=False,
            use_se=True,
        )
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 64, 32, 32)

    def test_config_dataclass(self) -> None:
        """Test ResidualStageConfig dataclass."""
        config = ResidualStageConfig(
            in_channels=64,
            out_channels=128,
            num_blocks=3,
            stride=2,
        )
        assert config.in_channels == 64
        assert config.out_channels == 128
        assert config.num_blocks == 3


class TestResidualSequence:
    """Tests for ResidualSequence."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        seq = ResidualSequence(
            in_channels=3,
            num_stages=4,
            blocks_per_stage=3,
            channels_per_stage=[64, 128, 256, 512],
        )
        assert seq is not None

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        seq = ResidualSequence(
            in_channels=3,
            num_stages=4,
            blocks_per_stage=3,
            channels_per_stage=[64, 128, 256, 512],
        )
        x = torch.randn(2, 3, 224, 224)
        output = seq(x)
        # With stride_schedule='later': stride[0]=1, others=2
        # 224 -> 224 (stride=1) -> 112 (stride=2) -> 56 (stride=2) -> 28 (stride=2)
        assert output.shape == (2, 512, 28, 28)

    def test_per_stage_block_count(self) -> None:
        """Test different block counts per stage."""
        seq = ResidualSequence(
            in_channels=3,
            num_stages=4,
            blocks_per_stage=[3, 4, 6, 3],
            channels_per_stage=[64, 128, 256, 512],
        )
        x = torch.randn(2, 3, 224, 224)
        output = seq(x)
        # With stride_schedule='later': stride[0]=1, others=2
        # 224 -> 224 (stride=1) -> 112 (stride=2) -> 56 (stride=2) -> 28 (stride=2)
        assert output.shape == (2, 512, 28, 28)

    def test_stride_schedule_later(self) -> None:
        """Test stride schedule 'later'."""
        seq = ResidualSequence(
            in_channels=3,
            num_stages=4,
            blocks_per_stage=3,
            channels_per_stage=[64, 128, 256, 512],
            stride_schedule='later',
        )
        x = torch.randn(1, 3, 224, 224)
        output = seq(x)
        assert output.shape[1] == 512

    def test_stride_schedule_all(self) -> None:
        """Test stride schedule 'all'."""
        seq = ResidualSequence(
            in_channels=3,
            num_stages=4,
            blocks_per_stage=3,
            channels_per_stage=[64, 128, 256, 512],
            stride_schedule='all',
        )
        x = torch.randn(1, 3, 224, 224)
        output = seq(x)
        assert output.shape[1] == 512

    def test_different_activations(self) -> None:
        """Test different activation functions."""
        for activation in ['relu', 'gelu', 'leaky_relu']:
            seq = ResidualSequence(
                in_channels=3,
                num_stages=2,
                blocks_per_stage=2,
                channels_per_stage=[64, 128],
                activation=activation,
            )
            x = torch.randn(2, 3, 224, 224)
            output = seq(x)
            assert output.shape[1] == 128


class TestResidualBottleneckStage:
    """Tests for ResidualBottleneckStage."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        stage = ResidualBottleneckStage(
            in_channels=64,
            out_channels=128,
            num_blocks=3,
            bottleneck_types='standard',
        )
        assert stage is not None

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        stage = ResidualBottleneckStage(
            in_channels=64,
            out_channels=128,
            num_blocks=3,
            bottleneck_types='standard',
            stride=2,
        )
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 128, 16, 16)

    def test_mixed_bottleneck_types(self) -> None:
        """Test mixed bottleneck types."""
        stage = ResidualBottleneckStage(
            in_channels=64,
            out_channels=128,
            num_blocks=3,
            bottleneck_types=['standard', 'wide', 'grouped'],
            stride=2,
        )
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 128, 16, 16)

    def test_repeated_bottleneck_type(self) -> None:
        """Test repeated bottleneck type specification."""
        stage = ResidualBottleneckStage(
            in_channels=64,
            out_channels=128,
            num_blocks=3,
            bottleneck_types=['wide'] * 3,
        )
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 128, 32, 32)


class TestResidualArchitectureBuilder:
    """Tests for ResidualArchitectureBuilder."""

    def test_builder_initialization(self) -> None:
        """Test builder initialization."""
        builder = ResidualArchitectureBuilder()
        assert builder is not None

    def test_minimal_builder(self) -> None:
        """Test building minimal architecture."""
        model = (
            ResidualArchitectureBuilder()
            .add_initial_conv(3, 64)
            .add_stage(None, 64, 3)
            .build()
        )
        assert model is not None
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape[1] == 64

    def test_full_resnet_builder(self) -> None:
        """Test building full ResNet-like architecture."""
        model = (
            ResidualArchitectureBuilder()
            .add_initial_conv(3, 64)
            .add_max_pool()
            .add_stage(None, 64, 3)
            .add_stage(None, 128, 4, stride=2, use_bottleneck=True)
            .add_stage(None, 256, 6, stride=2, use_bottleneck=True)
            .add_stage(None, 512, 3, stride=2, use_bottleneck=True)
            .add_global_avg_pool()
            .build()
        )
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 512, 1, 1)

    def test_builder_chaining(self) -> None:
        """Test builder chaining."""
        builder = ResidualArchitectureBuilder()
        result = builder.add_initial_conv(3, 64)
        assert result is builder

    def test_builder_without_layers(self) -> None:
        """Test building without adding layers raises error."""
        builder = ResidualArchitectureBuilder()
        with pytest.raises(ValueError):
            builder.build()

    def test_builder_initial_conv_params(self) -> None:
        """Test initial conv with custom parameters."""
        model = (
            ResidualArchitectureBuilder()
            .add_initial_conv(3, 32, kernel_size=3, stride=1, padding=1)
            .add_stage(None, 32, 2)
            .build()
        )
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 32, 224, 224)

    def test_builder_max_pool(self) -> None:
        """Test adding max pool to builder."""
        model = (
            ResidualArchitectureBuilder()
            .add_initial_conv(3, 64)
            .add_max_pool(kernel_size=3, stride=2, padding=1)
            .add_stage(None, 64, 2)
            .build()
        )
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape[1] == 64

    def test_builder_mixed_bottleneck_stage(self) -> None:
        """Test builder with mixed bottleneck stage."""
        model = (
            ResidualArchitectureBuilder()
            .add_initial_conv(3, 64)
            .add_mixed_bottleneck_stage(
                None, 128, 3, ['standard', 'wide', 'grouped'], stride=2
            )
            .build()
        )
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape[1] == 128

    def test_builder_current_channels_tracking(self) -> None:
        """Test builder tracks current channels correctly."""
        builder = ResidualArchitectureBuilder()
        builder.add_initial_conv(3, 64)
        assert builder.current_channels == 64
        
        builder.add_stage(None, 128, 2, stride=2)
        assert builder.current_channels == 128

    def test_builder_stage_without_in_channels(self) -> None:
        """Test builder stage without in_channels raises error."""
        builder = ResidualArchitectureBuilder()
        with pytest.raises(ValueError):
            builder.add_stage(None, 64, 3).build()


class TestIntegration:
    """Integration tests for residual architectures."""

    def test_bottleneck_in_stage(self) -> None:
        """Test using bottleneck blocks in stage."""
        stage = ResidualStage(
            in_channels=64,
            out_channels=256,
            num_blocks=3,
            use_bottleneck=True,
            bottleneck_type='wide',
            stride=2,
        )
        x = torch.randn(2, 64, 32, 32)
        output = stage(x)
        assert output.shape == (2, 256, 16, 16)

    def test_gradient_flow_through_sequence(self) -> None:
        """Test gradient flow through sequence."""
        seq = ResidualSequence(
            in_channels=3,
            num_stages=2,
            blocks_per_stage=2,
            channels_per_stage=[64, 128],
        )
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        output = seq(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_backward_builder(self) -> None:
        """Test forward and backward pass through builder model."""
        model = (
            ResidualArchitectureBuilder()
            .add_initial_conv(3, 64)
            .add_stage(None, 128, 2, stride=2)
            .build()
        )
        
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        # Initial conv with stride=2 (default): 224 -> 112
        # Stage with stride=2: 112 -> 56
        assert output.shape == (2, 128, 56, 56)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        stage = ResidualStage(
            in_channels=64, out_channels=128, num_blocks=3, stride=2
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 64, 32, 32)
            output = stage(x)
            assert output.shape == (batch_size, 128, 16, 16)

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of different bottleneck types."""
        standard = StandardBottleneck(256, 256)
        grouped = GroupedBottleneck(256, 256, num_groups=8)
        
        params_std = sum(p.numel() for p in standard.parameters())
        params_grp = sum(p.numel() for p in grouped.parameters())
        
        # Grouped should be more efficient
        assert params_grp <= params_std


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_block_stage(self) -> None:
        """Test stage with single block."""
        stage = ResidualStage(in_channels=64, out_channels=128, num_blocks=1, stride=2)
        x = torch.randn(1, 64, 32, 32)
        output = stage(x)
        assert output.shape == (1, 128, 16, 16)

    def test_large_channel_count(self) -> None:
        """Test with large channel counts."""
        block = StandardBottleneck(in_channels=2048, out_channels=2048)
        x = torch.randn(1, 2048, 7, 7)
        output = block(x)
        assert output.shape == (1, 2048, 7, 7)

    def test_small_input_size(self) -> None:
        """Test with small input sizes."""
        stage = ResidualStage(in_channels=64, out_channels=64, num_blocks=2)
        x = torch.randn(1, 64, 8, 8)
        output = stage(x)
        assert output.shape == (1, 64, 8, 8)

    def test_asymmetric_input(self) -> None:
        """Test with non-square inputs."""
        stage = ResidualStage(in_channels=64, out_channels=64, num_blocks=2)
        x = torch.randn(1, 64, 32, 48)
        output = stage(x)
        assert output.shape == (1, 64, 32, 48)
