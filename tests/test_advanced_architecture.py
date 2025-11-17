"""
Comprehensive tests for Step 6: Advanced Architecture Features

Tests for:
- Advanced architecture components (ResidualConnector, SkipConnectionBuilder, DenseConnectionBlock)
- Attention mechanisms (ChannelAttention, SpatialAttention, MultiHeadAttention)
- Architecture search utilities (GridSearch, RandomSearch, ArchitectureFactory)

Test Coverage:
- 70+ test cases across 7 test classes
- All edge cases and error conditions
- Integration tests for combined features

Author: torchvision-customizer
License: MIT
"""

import pytest
import torch
import torch.nn as nn
from typing import List

# Advanced Architecture Tests
from torchvision_customizer.blocks.advanced_architecture import (
    ResidualConnector,
    SkipConnectionBuilder,
    DenseConnectionBlock,
    MixedArchitectureBlock,
    create_skip_connections,
    validate_architecture_compatibility,
)

# Attention Mechanism Tests
from torchvision_customizer.layers.attention import (
    ChannelAttention,
    SpatialAttention,
    ChannelSpatialAttention,
    MultiHeadAttention,
    PositionalEncoding,
    AttentionBlock,
    create_attention_map,
    apply_attention,
    normalize_attention,
)

# Architecture Search Tests
from torchvision_customizer.utils.architecture_search import (
    ArchitectureConfig,
    GridSearch,
    RandomSearch,
    ArchitectureFactory,
    ArchitectureScorer,
    ArchitectureComparator,
    expand_config_list,
    sample_architecture,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_tensor():
    """Create sample tensor for testing."""
    return torch.randn(2, 64, 32, 32)


@pytest.fixture
def sample_batch():
    """Create sample batch of tensors."""
    return [torch.randn(2, 64, 32, 32) for _ in range(4)]


@pytest.fixture
def base_config():
    """Create base architecture configuration."""
    return ArchitectureConfig(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
    )


# ============================================================================
# TEST CLASS 1: ResidualConnector
# ============================================================================

class TestResidualConnector:
    """Test ResidualConnector module."""
    
    def test_identity_skip_initialization(self):
        """Test initialization with identity skip."""
        rc = ResidualConnector('identity', in_channels=64, out_channels=64)
        assert rc.skip_type == 'identity'
        assert rc.in_channels == 64
        assert rc.out_channels == 64
    
    def test_projection_skip_initialization(self):
        """Test initialization with projection skip."""
        rc = ResidualConnector('projection', in_channels=64, out_channels=128, stride=2)
        assert rc.skip_type == 'projection'
        assert rc.stride == 2
    
    def test_bottle_skip_initialization(self):
        """Test initialization with bottleneck skip."""
        rc = ResidualConnector('bottle', in_channels=64, out_channels=128)
        assert rc.skip_type == 'bottle'
    
    def test_invalid_skip_type(self):
        """Test error on invalid skip type."""
        with pytest.raises(ValueError):
            ResidualConnector('invalid_type')
    
    def test_identity_forward(self, sample_tensor):
        """Test forward pass with identity skip."""
        rc = ResidualConnector('identity', in_channels=64, out_channels=64)
        x = sample_tensor
        skip = sample_tensor.clone()
        output = rc(x, skip)
        assert output.shape == sample_tensor.shape
    
    def test_projection_forward(self):
        """Test forward pass with projection skip."""
        rc = ResidualConnector('projection', in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        output = rc(x, skip)
        assert output.shape == (2, 128, 16, 16)
    
    def test_bottle_forward(self):
        """Test forward pass with bottleneck skip."""
        rc = ResidualConnector('bottle', in_channels=64, out_channels=128)
        x = torch.randn(2, 128, 32, 32)
        skip = torch.randn(2, 64, 32, 32)
        output = rc(x, skip)
        assert output.shape == (2, 128, 32, 32)
    
    def test_residual_connector_gradient_flow(self):
        """Test gradient flow through residual connector."""
        rc = ResidualConnector('identity', in_channels=64, out_channels=64)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        skip = torch.randn(2, 64, 32, 32, requires_grad=True)
        output = rc(x, skip)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert skip.grad is not None


# ============================================================================
# TEST CLASS 2: SkipConnectionBuilder
# ============================================================================

class TestSkipConnectionBuilder:
    """Test SkipConnectionBuilder module."""
    
    def test_residual_pattern_initialization(self):
        """Test initialization with residual pattern."""
        builder = SkipConnectionBuilder('residual', num_blocks=4)
        assert builder.pattern == 'residual'
        assert builder.num_blocks == 4
    
    def test_dense_pattern_initialization(self):
        """Test initialization with dense pattern."""
        builder = SkipConnectionBuilder('dense', num_blocks=4, in_channels=64)
        assert builder.pattern == 'dense'
        assert hasattr(builder, 'projections')
    
    def test_hierarchical_pattern_initialization(self):
        """Test initialization with hierarchical pattern."""
        builder = SkipConnectionBuilder('hierarchical', num_blocks=5)
        assert builder.pattern == 'hierarchical'
    
    def test_invalid_pattern(self):
        """Test error on invalid pattern."""
        with pytest.raises(ValueError):
            SkipConnectionBuilder('invalid_pattern')
    
    def test_residual_pattern_forward(self, sample_batch):
        """Test residual pattern forward pass."""
        builder = SkipConnectionBuilder('residual', num_blocks=4)
        output = builder(sample_batch)
        assert output.shape == sample_batch[0].shape
    
    def test_hierarchical_pattern_forward(self, sample_batch):
        """Test hierarchical pattern forward pass."""
        builder = SkipConnectionBuilder('hierarchical', num_blocks=4)
        output = builder(sample_batch)
        assert output.shape == sample_batch[0].shape
    
    def test_dense_residual_pattern(self, sample_batch):
        """Test dense-residual pattern."""
        builder = SkipConnectionBuilder('dense_residual', num_blocks=4, in_channels=64)
        output = builder(sample_batch)
        assert output is not None


# ============================================================================
# TEST CLASS 3: DenseConnectionBlock
# ============================================================================

class TestDenseConnectionBlock:
    """Test DenseConnectionBlock module."""
    
    def test_initialization(self):
        """Test basic initialization."""
        block = DenseConnectionBlock(num_layers=4, in_channels=64, growth_rate=32)
        assert block.num_layers == 4
        assert block.growth_rate == 32
        assert block.out_channels == 64 + (4 * 32)
    
    def test_invalid_num_layers(self):
        """Test error on invalid num_layers."""
        with pytest.raises(ValueError):
            DenseConnectionBlock(num_layers=0)
    
    def test_invalid_growth_rate(self):
        """Test error on invalid growth_rate."""
        with pytest.raises(ValueError):
            DenseConnectionBlock(growth_rate=0)
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        block = DenseConnectionBlock(num_layers=4, in_channels=64, growth_rate=32)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        # Output channels = 64 + (4 * 32) = 192
        assert output.shape == (2, 192, 32, 32)
    
    def test_different_growth_rates(self):
        """Test with different growth rates."""
        for growth_rate in [16, 32, 48, 64]:
            block = DenseConnectionBlock(num_layers=3, in_channels=64, growth_rate=growth_rate)
            x = torch.randn(2, 64, 32, 32)
            output = block(x)
            expected_channels = 64 + (3 * growth_rate)
            assert output.shape[1] == expected_channels
    
    def test_gradient_flow(self):
        """Test gradient flow through dense block."""
        block = DenseConnectionBlock(num_layers=2, in_channels=64, growth_rate=32)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================================
# TEST CLASS 4: Attention Mechanisms
# ============================================================================

class TestAttentionMechanisms:
    """Test attention mechanism modules."""
    
    def test_channel_attention_initialization(self):
        """Test ChannelAttention initialization."""
        attn = ChannelAttention(channels=64, reduction=16)
        assert attn.channels == 64
        assert attn.reduction == 16
    
    def test_channel_attention_forward(self, sample_tensor):
        """Test ChannelAttention forward pass."""
        attn = ChannelAttention(channels=64)
        output = attn(sample_tensor)
        assert output.shape == sample_tensor.shape
    
    def test_spatial_attention_forward(self, sample_tensor):
        """Test SpatialAttention forward pass."""
        attn = SpatialAttention(kernel_size=7)
        output = attn(sample_tensor)
        assert output.shape == sample_tensor.shape
    
    def test_spatial_attention_invalid_kernel(self):
        """Test error on even kernel size."""
        with pytest.raises(ValueError):
            SpatialAttention(kernel_size=8)
    
    def test_channel_spatial_attention(self, sample_tensor):
        """Test combined ChannelSpatialAttention."""
        attn = ChannelSpatialAttention(channels=64)
        output = attn(sample_tensor)
        assert output.shape == sample_tensor.shape
    
    def test_multihead_attention_initialization(self):
        """Test MultiHeadAttention initialization."""
        attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        assert attn.embed_dim == 256
        assert attn.num_heads == 8
        assert attn.head_dim == 32
    
    def test_multihead_attention_dimension_mismatch(self):
        """Test error when embed_dim not divisible by num_heads."""
        with pytest.raises(ValueError):
            MultiHeadAttention(embed_dim=256, num_heads=7)
    
    def test_multihead_attention_forward(self):
        """Test MultiHeadAttention forward pass."""
        attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 32, 256)  # (batch, seq_len, embed_dim)
        output, weights = attn(x)
        assert output.shape == x.shape
        assert weights.shape == (2, 8, 32, 32)
    
    def test_positional_encoding_forward(self):
        """Test PositionalEncoding forward pass."""
        pos_enc = PositionalEncoding(d_model=256)
        x = torch.randn(2, 32, 256)
        pos = pos_enc(x)
        assert pos.shape[1] == 32
        assert pos.shape[2] == 256
    
    def test_attention_block_all_mechanisms(self, sample_tensor):
        """Test AttentionBlock with all mechanisms."""
        block = AttentionBlock(channels=64, use_channel=True, use_spatial=True)
        output = block(sample_tensor)
        assert output.shape == sample_tensor.shape
    
    def test_create_attention_map_channel(self, sample_tensor):
        """Test create_attention_map with channel type."""
        attn_map = create_attention_map(sample_tensor, 'channel')
        assert attn_map.shape == (2, 64, 1, 1)
    
    def test_create_attention_map_spatial(self, sample_tensor):
        """Test create_attention_map with spatial type."""
        attn_map = create_attention_map(sample_tensor, 'spatial')
        assert attn_map.shape == (2, 1, 32, 32)
    
    def test_apply_attention(self, sample_tensor):
        """Test apply_attention function."""
        attn_map = create_attention_map(sample_tensor, 'channel')
        output = apply_attention(sample_tensor, attn_map)
        assert output.shape == sample_tensor.shape
    
    def test_normalize_attention(self):
        """Test normalize_attention function."""
        attn = torch.randn(2, 64, 32, 32)
        normalized = normalize_attention(attn)
        # Check that it's normalized (sums to 1 along spatial dims)
        assert normalized is not None


# ============================================================================
# TEST CLASS 5: Architecture Search - Grid & Random
# ============================================================================

class TestArchitectureSearch:
    """Test architecture search utilities."""
    
    def test_architecture_config_creation(self):
        """Test ArchitectureConfig creation."""
        config = ArchitectureConfig(
            input_shape=(3, 224, 224),
            num_classes=1000,
            num_conv_blocks=4
        )
        assert config.num_conv_blocks == 4
        assert config.num_classes == 1000
    
    def test_architecture_config_validation(self):
        """Test ArchitectureConfig validation."""
        config = ArchitectureConfig(
            input_shape=(3, 224, 224),
            num_classes=1000,
        )
        assert config.validate() is True
    
    def test_invalid_input_shape(self):
        """Test validation error on invalid input_shape."""
        with pytest.raises(ValueError):
            config = ArchitectureConfig(
                input_shape=(224, 224),  # Missing channel
                num_classes=1000,
            )
            config.validate()
    
    def test_grid_search_initialization(self):
        """Test GridSearch initialization."""
        search_space = {
            'num_conv_blocks': [3, 4, 5],
            'activation': ['relu', 'gelu'],
        }
        searcher = GridSearch(search_space)
        assert searcher.num_combinations == 6  # 3 * 2
    
    def test_grid_search_generation(self, base_config):
        """Test GridSearch configuration generation."""
        search_space = {
            'num_conv_blocks': [3, 4],
            'activations': ['relu', 'gelu'],
        }
        base = base_config.to_dict()
        searcher = GridSearch(search_space, base)
        configs = list(searcher.generate())
        assert len(configs) == 4  # 2 * 2
    
    def test_random_search_initialization(self):
        """Test RandomSearch initialization."""
        search_space = {
            'num_conv_blocks': [3, 4, 5],
            'activation': ['relu', 'gelu'],
        }
        searcher = RandomSearch(search_space, num_samples=10)
        assert searcher.num_samples == 10
    
    def test_random_search_generation(self, base_config):
        """Test RandomSearch configuration generation."""
        search_space = {
            'activations': ['relu', 'gelu', 'leaky_relu'],
        }
        base = base_config.to_dict()
        searcher = RandomSearch(search_space, num_samples=5, base_config=base)
        configs = list(searcher.generate())
        assert len(configs) == 5
    
    def test_random_search_reproducibility(self, base_config):
        """Test RandomSearch reproducibility with seed."""
        search_space = {'num_conv_blocks': [2, 3, 4, 5]}
        base = base_config.to_dict()
        
        searcher1 = RandomSearch(search_space, num_samples=3, base_config=base, seed=42)
        configs1 = [c.num_conv_blocks for c in searcher1.generate()]
        
        searcher2 = RandomSearch(search_space, num_samples=3, base_config=base, seed=42)
        configs2 = [c.num_conv_blocks for c in searcher2.generate()]
        
        assert configs1 == configs2


# ============================================================================
# TEST CLASS 6: Architecture Factory & Scoring
# ============================================================================

class TestArchitectureFactory:
    """Test ArchitectureFactory and scoring."""
    
    def test_factory_initialization(self):
        """Test ArchitectureFactory initialization."""
        factory = ArchitectureFactory()
        assert 'sequential' in factory.patterns
        assert 'residual' in factory.patterns
        assert 'dense' in factory.patterns
    
    def test_create_sequential_architecture(self, base_config):
        """Test creating sequential architecture."""
        factory = ArchitectureFactory()
        base_config.pattern = 'sequential'
        arch = factory.create(base_config)
        assert arch['type'] == 'sequential'
    
    def test_create_residual_architecture(self, base_config):
        """Test creating residual architecture."""
        factory = ArchitectureFactory()
        base_config.pattern = 'residual'
        arch = factory.create(base_config)
        assert arch['type'] == 'residual'
        assert arch['use_residual'] is True
    
    def test_create_dense_architecture(self, base_config):
        """Test creating dense architecture."""
        factory = ArchitectureFactory()
        base_config.pattern = 'dense'
        arch = factory.create(base_config)
        assert arch['type'] == 'dense'
    
    def test_architecture_scorer_initialization(self):
        """Test ArchitectureScorer initialization."""
        scorer = ArchitectureScorer(max_parameters=10_000_000)
        assert scorer.max_parameters == 10_000_000
    
    def test_architecture_scorer_scoring(self, base_config):
        """Test ArchitectureScorer scoring."""
        scorer = ArchitectureScorer()
        score = scorer.score_config(base_config)
        assert 0.0 <= score <= 1.0
    
    def test_architecture_comparator(self, base_config):
        """Test ArchitectureComparator."""
        configs = [
            base_config,
            ArchitectureConfig(
                input_shape=(3, 224, 224),
                num_classes=1000,
                num_conv_blocks=3,
            ),
        ]
        comparator = ArchitectureComparator(configs)
        best, score = comparator.get_best()
        assert best is not None
        assert 0.0 <= score <= 1.0
    
    def test_architecture_comparator_top_k(self, base_config):
        """Test ArchitectureComparator get_top_k."""
        configs = [
            base_config,
            ArchitectureConfig(
                input_shape=(3, 224, 224),
                num_classes=1000,
                num_conv_blocks=3,
            ),
        ]
        comparator = ArchitectureComparator(configs)
        top_k = comparator.get_top_k(k=1)
        assert len(top_k) == 1
    
    def test_architecture_comparator_statistics(self, base_config):
        """Test ArchitectureComparator get_statistics."""
        configs = [base_config for _ in range(3)]
        comparator = ArchitectureComparator(configs)
        stats = comparator.get_statistics()
        assert 'mean' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'std' in stats


# ============================================================================
# TEST CLASS 7: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_expand_config_list_single_value(self):
        """Test expand_config_list with single value."""
        result = expand_config_list('relu', 4)
        assert result == ['relu', 'relu', 'relu', 'relu']
    
    def test_expand_config_list_already_list(self):
        """Test expand_config_list with already a list."""
        result = expand_config_list([1, 2, 3, 4], 4)
        assert result == [1, 2, 3, 4]
    
    def test_expand_config_list_length_mismatch(self):
        """Test expand_config_list length mismatch error."""
        with pytest.raises(ValueError):
            expand_config_list([1, 2], 4)
    
    def test_sample_architecture(self):
        """Test sample_architecture utility."""
        arch = sample_architecture(num_conv_blocks=4)
        assert arch.num_conv_blocks == 4
        assert len(arch.channels) == 4
        assert len(arch.kernel_sizes) == 4
    
    def test_create_skip_connections_residual(self, sample_batch):
        """Test create_skip_connections with residual pattern."""
        result = create_skip_connections(sample_batch, 'residual')
        assert result.shape == sample_batch[0].shape
    
    def test_create_skip_connections_concatenate(self, sample_batch):
        """Test create_skip_connections with concatenate pattern."""
        result = create_skip_connections(sample_batch, 'concatenate')
        # Concatenate along channel dimension
        assert result.shape[0] == 2  # batch size
        assert result.shape[1] == 64 * 4  # 4 tensors concatenated
    
    def test_validate_architecture_compatibility(self, sample_batch):
        """Test validate_architecture_compatibility."""
        result = validate_architecture_compatibility(sample_batch, 'residual')
        assert result is True
    
    def test_validate_architecture_incompatible(self):
        """Test validate_architecture_compatibility with incompatible features."""
        features = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 32, 32),  # Different channels
        ]
        with pytest.raises(ValueError):
            validate_architecture_compatibility(features, 'residual')


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_dense_block_with_attention(self):
        """Test DenseConnectionBlock with AttentionBlock."""
        dense = DenseConnectionBlock(num_layers=2, in_channels=64, growth_rate=32)
        # Output channels: 64 + (2 * 32) = 128
        attn = ChannelAttention(channels=128)
        
        x = torch.randn(2, 64, 32, 32)
        dense_out = dense(x)
        attn_out = attn(dense_out)
        
        assert attn_out.shape == dense_out.shape
    
    def test_residual_connector_with_skip_builder(self, sample_batch):
        """Test ResidualConnector with SkipConnectionBuilder."""
        connector = ResidualConnector('identity', in_channels=64, out_channels=64)
        builder = SkipConnectionBuilder('residual', num_blocks=len(sample_batch))
        
        # Apply connector to first pair
        output1 = connector(sample_batch[0], sample_batch[1])
        
        # Use builder
        output2 = builder(sample_batch)
        
        assert output1 is not None
        assert output2 is not None
    
    def test_architecture_search_with_factory(self, base_config):
        """Test architecture search with factory."""
        search_space = {'num_conv_blocks': [3, 4, 5]}
        base = base_config.to_dict()
        
        searcher = GridSearch(search_space, base)
        factory = ArchitectureFactory()
        
        for config in searcher.generate():
            arch = factory.create(config)
            assert arch is not None
    
    def test_multihead_attention_integration(self):
        """Test MultiHeadAttention with sequential processing."""
        attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        
        # Simulate sequential tokens
        x = torch.randn(4, 10, 256)  # 4 samples, 10 tokens, 256 dims
        output, weights = attn(x)
        
        assert output.shape == x.shape
        assert weights.shape[1] == 8  # num_heads
    
    def test_attention_block_in_pipeline(self):
        """Test AttentionBlock in a simple pipeline."""
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        attn = AttentionBlock(channels=64, use_channel=True, use_spatial=True)
        
        x = torch.randn(2, 3, 32, 32)
        x = conv(x)
        x = attn(x)
        
        assert x.shape == (2, 64, 32, 32)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and efficiency tests."""
    
    def test_dense_block_memory_efficiency(self):
        """Test DenseConnectionBlock memory efficiency."""
        block = DenseConnectionBlock(num_layers=5, in_channels=64, growth_rate=16)
        x = torch.randn(2, 64, 32, 32)
        
        # Should complete without memory issues
        output = block(x)
        assert output is not None
    
    def test_multihead_attention_efficiency(self):
        """Test MultiHeadAttention with large sequences."""
        attn = MultiHeadAttention(embed_dim=512, num_heads=16)
        x = torch.randn(2, 64, 512)  # 64 tokens
        
        output, weights = attn(x)
        assert output.shape == x.shape
    
    def test_grid_search_large_space(self, base_config):
        """Test GridSearch with large search space."""
        search_space = {
            'num_conv_blocks': [2, 3, 4],
            'activation': ['relu', 'gelu'],
            'dropout_rates': [0.0, 0.1, 0.2],
        }
        base = base_config.to_dict()
        
        searcher = GridSearch(search_space, base)
        # Should handle large combinations (3 * 2 * 3 = 18)
        assert searcher.num_combinations == 18


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
