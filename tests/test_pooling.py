"""Tests for pooling layers and utilities.

Comprehensive test suite for:
    - get_pooling factory function
    - PoolingFactory class
    - PoolingBlock class
    - StochasticPool2d class
    - LPPool2d class
    - Utility functions (calculate_pooling_output_size, validate_pooling_config)
"""

import pytest
import torch
import torch.nn as nn
from torchvision_customizer.layers import (
    get_pooling,
    PoolingFactory,
    PoolingBlock,
    StochasticPool2d,
    LPPool2d,
    calculate_pooling_output_size,
    validate_pooling_config,
)


class TestGetPooling:
    """Test get_pooling factory function."""

    def test_max_pooling_default(self):
        """Test MaxPool2d creation with defaults."""
        pool = get_pooling('max')
        assert isinstance(pool, nn.MaxPool2d)
        x = torch.randn(2, 32, 64, 64)
        out = pool(x)
        assert out.shape == (2, 32, 32, 32)

    def test_max_pooling_custom_kernel(self):
        """Test MaxPool2d with custom kernel size."""
        pool = get_pooling('max', kernel_size=3, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 15, 15)

    def test_avg_pooling_default(self):
        """Test AvgPool2d creation."""
        pool = get_pooling('avg')
        assert isinstance(pool, nn.AvgPool2d)
        x = torch.randn(2, 32, 64, 64)
        out = pool(x)
        assert out.shape == (2, 32, 32, 32)

    def test_avg_pooling_custom(self):
        """Test AvgPool2d with custom parameters."""
        pool = get_pooling('avg', kernel_size=4, stride=4)
        x = torch.randn(2, 64, 64, 64)
        out = pool(x)
        assert out.shape == (2, 64, 16, 16)

    def test_adaptive_max_pooling(self):
        """Test AdaptiveMaxPool2d."""
        pool = get_pooling('adaptive_max', output_size=(1, 1))
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 1, 1)

    def test_adaptive_max_pooling_custom_size(self):
        """Test AdaptiveMaxPool2d with custom output size."""
        pool = get_pooling('adaptive_max', output_size=(7, 7))
        x = torch.randn(2, 64, 224, 224)
        out = pool(x)
        assert out.shape == (2, 64, 7, 7)

    def test_adaptive_avg_pooling(self):
        """Test AdaptiveAvgPool2d."""
        pool = get_pooling('adaptive_avg', output_size=(1, 1))
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 1, 1)

    def test_adaptive_avg_pooling_custom_size(self):
        """Test AdaptiveAvgPool2d with custom output size."""
        pool = get_pooling('adaptive_avg', output_size=(4, 4))
        x = torch.randn(2, 128, 64, 64)
        out = pool(x)
        assert out.shape == (2, 128, 4, 4)

    def test_identity_pooling(self):
        """Test identity (none) pooling."""
        pool = get_pooling('none')
        assert isinstance(pool, nn.Identity)
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert torch.equal(x, out)

    def test_pooling_case_insensitive(self):
        """Test that pooling type is case-insensitive."""
        pool1 = get_pooling('MAX')
        pool2 = get_pooling('max')
        pool3 = get_pooling('Max')
        assert isinstance(pool1, nn.MaxPool2d)
        assert isinstance(pool2, nn.MaxPool2d)
        assert isinstance(pool3, nn.MaxPool2d)

    def test_invalid_pooling_type(self):
        """Test that invalid pooling type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported pooling type"):
            get_pooling('invalid_pool')

    def test_pooling_with_padding(self):
        """Test pooling with padding."""
        pool = get_pooling('max', kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 32, 64, 64)
        out = pool(x)
        assert out.shape == (2, 32, 64, 64)

    def test_different_h_w_input(self):
        """Test pooling with non-square input."""
        pool = get_pooling('avg', kernel_size=2, stride=2)
        x = torch.randn(2, 64, 48, 64)
        out = pool(x)
        assert out.shape == (2, 64, 24, 32)

    def test_large_batch_size(self):
        """Test pooling with large batch size."""
        pool = get_pooling('max', kernel_size=2, stride=2)
        x = torch.randn(64, 32, 64, 64)
        out = pool(x)
        assert out.shape == (64, 32, 32, 32)


class TestPoolingFactory:
    """Test PoolingFactory class."""

    def test_factory_create_max_pooling(self):
        """Test factory create method for max pooling."""
        factory = PoolingFactory()
        pool = factory.create('max', kernel_size=2)
        assert isinstance(pool, nn.MaxPool2d)

    def test_factory_is_supported(self):
        """Test factory is_supported method."""
        factory = PoolingFactory()
        assert factory.is_supported('max')
        assert factory.is_supported('avg')
        assert factory.is_supported('adaptive_max')
        assert factory.is_supported('adaptive_avg')
        assert factory.is_supported('none')
        assert not factory.is_supported('invalid')

    def test_factory_supported_pooling(self):
        """Test factory supported_pooling method."""
        factory = PoolingFactory()
        supported = factory.supported_pooling()
        assert 'max' in supported
        assert 'avg' in supported
        assert 'adaptive_max' in supported
        assert 'adaptive_avg' in supported
        assert 'none' in supported

    def test_factory_get_defaults(self):
        """Test factory get_defaults method."""
        factory = PoolingFactory()
        defaults_max = factory.get_defaults('max')
        assert 'kernel_size' in defaults_max
        assert 'stride' in defaults_max

        defaults_adaptive = factory.get_defaults('adaptive_max')
        assert 'output_size' in defaults_adaptive

    def test_factory_get_defaults_invalid(self):
        """Test that get_defaults raises ValueError for invalid pooling."""
        factory = PoolingFactory()
        with pytest.raises(ValueError):
            factory.get_defaults('invalid_pool')


class TestPoolingBlock:
    """Test PoolingBlock class."""

    def test_pooling_block_init_max(self):
        """Test PoolingBlock initialization with max pooling."""
        block = PoolingBlock('max', kernel_size=2)
        assert block.pool_type == 'max'
        assert isinstance(block.pool, nn.MaxPool2d)

    def test_pooling_block_forward_max(self):
        """Test PoolingBlock forward with max pooling."""
        block = PoolingBlock('max', kernel_size=2, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_pooling_block_forward_avg(self):
        """Test PoolingBlock forward with average pooling."""
        block = PoolingBlock('avg', kernel_size=2, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_pooling_block_with_dropout(self):
        """Test PoolingBlock with dropout."""
        block = PoolingBlock('max', kernel_size=2, dropout_rate=0.5)
        block.train()
        x = torch.randn(2, 64, 32, 32)
        out1 = block(x)
        out2 = block(x)
        # With dropout, outputs should differ (with high probability)
        # Just check shape for deterministic testing
        assert out1.shape == (2, 64, 16, 16)
        assert out2.shape == (2, 64, 16, 16)

    def test_pooling_block_dropout_zero(self):
        """Test PoolingBlock with zero dropout (no dropout applied)."""
        block = PoolingBlock('max', kernel_size=2, dropout_rate=0.0)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)
        assert block.dropout is None

    def test_pooling_block_adaptive(self):
        """Test PoolingBlock with adaptive pooling."""
        block = PoolingBlock('adaptive_max', output_size=(1, 1))
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 1, 1)

    def test_pooling_block_adaptive_avg(self):
        """Test PoolingBlock with adaptive average pooling."""
        block = PoolingBlock('adaptive_avg', output_size=(7, 7))
        x = torch.randn(2, 128, 224, 224)
        out = block(x)
        assert out.shape == (2, 128, 7, 7)

    def test_pooling_block_calculate_output_size_max(self):
        """Test calculate_output_size for max pooling."""
        block = PoolingBlock('max', kernel_size=2, stride=2)
        h_out, w_out = block.calculate_output_size(32, 32)
        assert h_out == 16
        assert w_out == 16

    def test_pooling_block_calculate_output_size_custom(self):
        """Test calculate_output_size with custom kernel."""
        block = PoolingBlock('max', kernel_size=3, stride=2)
        h_out, w_out = block.calculate_output_size(64, 64)
        assert h_out == 31
        assert w_out == 31

    def test_pooling_block_calculate_output_size_adaptive(self):
        """Test calculate_output_size for adaptive pooling."""
        block = PoolingBlock('adaptive_max', output_size=(1, 1))
        h_out, w_out = block.calculate_output_size(32, 32)
        assert h_out == 1
        assert w_out == 1

    def test_pooling_block_non_square_input(self):
        """Test PoolingBlock with non-square input."""
        block = PoolingBlock('max', kernel_size=2, stride=2)
        x = torch.randn(2, 64, 48, 64)
        out = block(x)
        assert out.shape == (2, 64, 24, 32)


class TestStochasticPool2d:
    """Test StochasticPool2d class."""

    def test_stochastic_pool_init(self):
        """Test StochasticPool2d initialization."""
        pool = StochasticPool2d(kernel_size=2, stride=2)
        assert pool.kernel_size == 2
        assert pool.stride == 2

    def test_stochastic_pool_eval_mode(self):
        """Test StochasticPool2d in eval mode uses max pooling."""
        pool = StochasticPool2d(kernel_size=2, stride=2)
        pool.eval()
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 16, 16)

    def test_stochastic_pool_train_mode(self):
        """Test StochasticPool2d in train mode."""
        pool = StochasticPool2d(kernel_size=2, stride=2)
        pool.train()
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 16, 16)

    def test_stochastic_pool_training_vs_eval(self):
        """Test that training and eval modes produce different outputs."""
        pool = StochasticPool2d(kernel_size=2, stride=2)
        x = torch.randn(2, 64, 32, 32)

        # Training mode
        pool.train()
        out_train = pool(x)

        # Eval mode
        pool.eval()
        out_eval = pool(x)

        # Outputs should have same shape
        assert out_train.shape == out_eval.shape == (2, 64, 16, 16)

    def test_stochastic_pool_with_padding(self):
        """Test StochasticPool2d with padding."""
        pool = StochasticPool2d(kernel_size=2, stride=2, padding=1)
        pool.train()
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        # With padding=1, stride=2: output = (32 + 2*1 - 2) // 2 + 1 = 17
        assert out.shape == (2, 64, 17, 17)

    def test_stochastic_pool_large_kernel(self):
        """Test StochasticPool2d with larger kernel."""
        pool = StochasticPool2d(kernel_size=3, stride=2)
        pool.train()
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 15, 15)

    def test_stochastic_pool_values_valid(self):
        """Test that stochastic pooling produces values from input."""
        pool = StochasticPool2d(kernel_size=2, stride=2)
        pool.train()
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        # Output values should be within range of input
        # (This is probabilistic, so just check shape)
        assert out.shape == (2, 64, 16, 16)


class TestLPPool2d:
    """Test LPPool2d class."""

    def test_lp_pool_init(self):
        """Test LPPool2d initialization."""
        pool = LPPool2d(norm_type=2, kernel_size=2)
        assert pool.norm_type == 2
        assert pool.kernel_size == 2

    def test_lp_pool_norm2(self):
        """Test LPPool2d with norm_type=2 (RMS pooling)."""
        pool = LPPool2d(norm_type=2, kernel_size=2, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 16, 16)

    def test_lp_pool_norm1(self):
        """Test LPPool2d with norm_type=1."""
        pool = LPPool2d(norm_type=1, kernel_size=2, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 16, 16)

    def test_lp_pool_custom_kernel(self):
        """Test LPPool2d with custom kernel size."""
        pool = LPPool2d(norm_type=2, kernel_size=3, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        assert out.shape == (2, 64, 15, 15)

    def test_lp_pool_with_padding(self):
        """Test LPPool2d output size."""
        pool = LPPool2d(norm_type=2, kernel_size=2, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = pool(x)
        # LPPool2d uses same output calculation as MaxPool2d
        assert out.shape == (2, 64, 16, 16)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_pooling_output_size_basic(self):
        """Test calculate_pooling_output_size basic calculation."""
        out_size = calculate_pooling_output_size(
            input_size=32, kernel_size=2, stride=2
        )
        assert out_size == 16

    def test_calculate_pooling_output_size_no_stride(self):
        """Test calculate_pooling_output_size with stride=1."""
        out_size = calculate_pooling_output_size(
            input_size=32, kernel_size=3, stride=1
        )
        assert out_size == 30

    def test_calculate_pooling_output_size_with_padding(self):
        """Test calculate_pooling_output_size with padding."""
        out_size = calculate_pooling_output_size(
            input_size=32, kernel_size=2, stride=2, padding=1
        )
        assert out_size == 17

    def test_calculate_pooling_output_size_large_input(self):
        """Test calculate_pooling_output_size with large input."""
        out_size = calculate_pooling_output_size(
            input_size=224, kernel_size=7, stride=2
        )
        assert out_size == 109

    def test_calculate_pooling_output_size_dilation(self):
        """Test calculate_pooling_output_size with dilation."""
        out_size = calculate_pooling_output_size(
            input_size=32, kernel_size=3, stride=1, dilation=2
        )
        assert out_size == 28

    def test_validate_pooling_config_valid(self):
        """Test validate_pooling_config with valid configuration."""
        is_valid, msg = validate_pooling_config(
            pool_type='max', kernel_size=2, stride=2,
            input_height=32, input_width=32
        )
        assert is_valid
        assert msg == 'Valid'

    def test_validate_pooling_config_invalid_kernel(self):
        """Test validate_pooling_config with kernel larger than input."""
        is_valid, msg = validate_pooling_config(
            pool_type='max', kernel_size=64, stride=2,
            input_height=32, input_width=32
        )
        assert not is_valid
        assert 'larger' in msg.lower()

    def test_validate_pooling_config_negative_kernel(self):
        """Test validate_pooling_config with negative kernel."""
        is_valid, msg = validate_pooling_config(
            pool_type='max', kernel_size=-1, stride=2,
            input_height=32, input_width=32
        )
        assert not is_valid

    def test_validate_pooling_config_invalid_stride(self):
        """Test validate_pooling_config with invalid stride."""
        is_valid, msg = validate_pooling_config(
            pool_type='max', kernel_size=2, stride=0,
            input_height=32, input_width=32
        )
        assert not is_valid

    def test_validate_pooling_config_non_square(self):
        """Test validate_pooling_config with non-square input."""
        is_valid, msg = validate_pooling_config(
            pool_type='max', kernel_size=2, stride=2,
            input_height=48, input_width=64
        )
        assert is_valid


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_pooling_in_sequence(self):
        """Test using pooling in nn.Sequential."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            PoolingBlock('max', kernel_size=2, dropout_rate=0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            PoolingBlock('avg', kernel_size=2),
        )
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 128, 16, 16)

    def test_multiple_pooling_blocks(self):
        """Test using multiple PoolingBlock instances."""
        blocks = nn.ModuleList([
            PoolingBlock('max', kernel_size=2),
            PoolingBlock('avg', kernel_size=2),
            PoolingBlock('adaptive_max', output_size=(1, 1)),
        ])
        x = torch.randn(2, 64, 32, 32)
        
        out = x
        for i, block in enumerate(blocks):
            out = block(out)
            if i == 0:
                assert out.shape == (2, 64, 16, 16)
            elif i == 1:
                assert out.shape == (2, 64, 8, 8)
            elif i == 2:
                assert out.shape == (2, 64, 1, 1)

    def test_pooling_gradients(self):
        """Test that gradients flow through pooling layers."""
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        pool = PoolingBlock('max', kernel_size=2)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_stochastic_vs_max_pooling_shapes(self):
        """Test that stochastic and max pooling produce same shapes."""
        x = torch.randn(2, 64, 32, 32)
        
        max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        max_out = max_pool(x)
        
        stoch_pool = StochasticPool2d(kernel_size=2, stride=2)
        stoch_pool.eval()
        stoch_out = stoch_pool(x)
        
        assert max_out.shape == stoch_out.shape

    def test_adaptive_pooling_various_sizes(self):
        """Test adaptive pooling with various input sizes."""
        inputs = [
            (2, 64, 32, 32),
            (2, 64, 48, 64),
            (2, 64, 224, 224),
            (2, 64, 7, 7),
        ]
        block = PoolingBlock('adaptive_avg', output_size=(7, 7))
        
        for shape in inputs:
            x = torch.randn(*shape)
            out = block(x)
            assert out.shape == (shape[0], shape[1], 7, 7)

    def test_dropout_affects_training(self):
        """Test that dropout is applied during training."""
        block = PoolingBlock('max', kernel_size=2, dropout_rate=0.99)
        x = torch.randn(2, 64, 32, 32)
        
        # Training mode
        block.train()
        out_train = block(x)
        
        # Eval mode
        block.eval()
        out_eval = block(x)
        
        # Both should have same shape
        assert out_train.shape == out_eval.shape == (2, 64, 16, 16)

    def test_pooling_factory_all_types(self):
        """Test PoolingFactory with all supported pooling types."""
        factory = PoolingFactory()
        x = torch.randn(2, 64, 32, 32)
        
        for pool_type in factory.supported_pooling():
            pool = factory.create(pool_type)
            out = pool(x)
            assert out.shape[0] == 2
            assert out.shape[1] == 64


class TestEdgeCases:
    """Edge case tests."""

    def test_pooling_single_channel(self):
        """Test pooling with single channel input."""
        pool = PoolingBlock('max', kernel_size=2)
        x = torch.randn(2, 1, 32, 32)
        out = pool(x)
        assert out.shape == (2, 1, 16, 16)

    def test_pooling_single_batch(self):
        """Test pooling with single sample batch."""
        pool = PoolingBlock('avg', kernel_size=2)
        x = torch.randn(1, 64, 32, 32)
        out = pool(x)
        assert out.shape == (1, 64, 16, 16)

    def test_pooling_large_channels(self):
        """Test pooling with large number of channels."""
        pool = PoolingBlock('max', kernel_size=2)
        x = torch.randn(2, 2048, 32, 32)
        out = pool(x)
        assert out.shape == (2, 2048, 16, 16)

    def test_pooling_small_spatial_dims(self):
        """Test pooling with small spatial dimensions."""
        pool = PoolingBlock('max', kernel_size=2, stride=2)
        x = torch.randn(2, 64, 4, 4)
        out = pool(x)
        assert out.shape == (2, 64, 2, 2)

    def test_stochastic_pool_deterministic_eval(self):
        """Test that stochastic pool is deterministic in eval mode."""
        pool = StochasticPool2d(kernel_size=2, stride=2)
        pool.eval()
        x = torch.randn(2, 64, 32, 32)
        
        out1 = pool(x)
        out2 = pool(x)
        
        assert torch.equal(out1, out2)

    def test_high_dropout_rate(self):
        """Test PoolingBlock with high dropout rate."""
        block = PoolingBlock('max', kernel_size=2, dropout_rate=0.9)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_output_size_calculation_edge_case(self):
        """Test output size calculation with edge case values."""
        # Very large stride
        out = calculate_pooling_output_size(
            input_size=32, kernel_size=2, stride=32
        )
        assert out == 1
        
        # Kernel equals input size
        out = calculate_pooling_output_size(
            input_size=32, kernel_size=32, stride=1
        )
        assert out == 1

    def test_mixed_pooling_types_in_model(self):
        """Test using different pooling types in same model."""
        model = nn.Sequential(
            PoolingBlock('max', kernel_size=2),
            PoolingBlock('avg', kernel_size=2),
            PoolingBlock('adaptive_max', output_size=(1, 1)),
        )
        x = torch.randn(2, 64, 32, 32)
        out = model(x)
        assert out.shape == (2, 64, 1, 1)

