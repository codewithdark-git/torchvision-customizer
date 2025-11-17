"""
Step 6: Advanced Architecture Features - Examples

Comprehensive examples demonstrating:
- Advanced architecture components (residual, skip, dense connections)
- Attention mechanisms (channel, spatial, multi-head)
- Architecture search (grid, random, factory, scoring)

All examples are production-ready and fully executable.

Author: torchvision-customizer
License: MIT
"""

import torch
import torch.nn as nn
from torchvision_customizer.blocks.advanced_architecture import (
    ResidualConnector,
    SkipConnectionBuilder,
    DenseConnectionBlock,
    MixedArchitectureBlock,
)
from torchvision_customizer.layers.attention import (
    ChannelAttention,
    SpatialAttention,
    ChannelSpatialAttention,
    MultiHeadAttention,
    PositionalEncoding,
    AttentionBlock,
)
from torchvision_customizer.utils.architecture_search import (
    ArchitectureConfig,
    GridSearch,
    RandomSearch,
    ArchitectureFactory,
    ArchitectureScorer,
    ArchitectureComparator,
    sample_architecture,
)


# ============================================================================
# EXAMPLE 1: Residual Connector with Different Skip Patterns
# ============================================================================

def example_1_residual_connector():
    """
    Demonstrate ResidualConnector with different skip connection types.
    
    Shows how to use identity, projection, and bottleneck skip connections
    for adding residual paths between arbitrary layers.
    """
    print("=" * 70)
    print("EXAMPLE 1: Residual Connector - Different Skip Patterns")
    print("=" * 70)
    
    # Identity skip (same dimensions)
    print("\n1. Identity Skip Connection (same dimensions):")
    rc_identity = ResidualConnector('identity', in_channels=64, out_channels=64)
    x = torch.randn(2, 64, 32, 32)
    skip = torch.randn(2, 64, 32, 32)
    output = rc_identity(x, skip)
    print(f"   Input shape: {x.shape}")
    print(f"   Skip shape: {skip.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Projection skip (different dimensions with stride)
    print("\n2. Projection Skip Connection (downsampling):")
    rc_proj = ResidualConnector('projection', in_channels=64, out_channels=128, stride=2)
    x = torch.randn(2, 128, 16, 16)
    skip = torch.randn(2, 64, 32, 32)
    output = rc_proj(x, skip)
    print(f"   Input shape: {x.shape}")
    print(f"   Skip shape (before projection): {skip.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Bottleneck skip (efficient with 1x1->3x3->1x1 pattern)
    print("\n3. Bottleneck Skip Connection (efficient):")
    rc_bottle = ResidualConnector('bottle', in_channels=64, out_channels=128)
    x = torch.randn(2, 128, 32, 32)
    skip = torch.randn(2, 64, 32, 32)
    output = rc_bottle(x, skip)
    print(f"   Input shape: {x.shape}")
    print(f"   Skip shape: {skip.shape}")
    print(f"   Output shape: {output.shape} (bottleneck applied)")
    
    print()


# ============================================================================
# EXAMPLE 2: Skip Connection Patterns for Multi-Layer Blocks
# ============================================================================

def example_2_skip_connection_patterns():
    """
    Demonstrate different skip connection patterns for combining multiple features.
    
    Shows residual, dense, hierarchical, and hybrid patterns for connecting
    multiple layers in a network.
    """
    print("=" * 70)
    print("EXAMPLE 2: Skip Connection Patterns")
    print("=" * 70)
    
    # Create sample features from multiple layers
    features = [torch.randn(2, 64, 32, 32) for _ in range(4)]
    print(f"Input: 4 feature maps of shape {features[0].shape}\n")
    
    # Residual pattern: simple accumulation
    print("1. Residual Pattern (accumulation):")
    builder_res = SkipConnectionBuilder('residual', num_blocks=4, in_channels=64)
    output_res = builder_res(features)
    print(f"   Output shape: {output_res.shape} (all features summed)")
    
    # Hierarchical pattern: skip at multiple scales
    print("\n2. Hierarchical Pattern (multi-scale skips):")
    builder_hier = SkipConnectionBuilder('hierarchical', num_blocks=4, in_channels=64)
    output_hier = builder_hier(features)
    print(f"   Output shape: {output_hier.shape} (every 2nd layer connected)")
    
    # Dense-residual pattern: combination of both
    print("\n3. Dense-Residual Pattern (hybrid):")
    builder_dense_res = SkipConnectionBuilder('dense_residual', num_blocks=4, in_channels=64)
    output_dense_res = builder_dense_res(features)
    print(f"   Output shape: {output_dense_res.shape} (combined pattern)")
    
    print()


# ============================================================================
# EXAMPLE 3: Dense Connection Block (DenseNet-style)
# ============================================================================

def example_3_dense_connection_block():
    """
    Demonstrate DenseConnectionBlock for dense feature propagation.
    
    Shows how dense connections allow gradient flow directly to all previous layers,
    improving gradient propagation and parameter efficiency.
    """
    print("=" * 70)
    print("EXAMPLE 3: Dense Connection Block")
    print("=" * 70)
    
    # Create dense block with 4 layers and growth rate of 32
    print("\nDense Block Configuration:")
    print("  - Number of layers: 4")
    print("  - Input channels: 64")
    print("  - Growth rate: 32 channels per layer")
    
    dense_block = DenseConnectionBlock(
        num_layers=4,
        in_channels=64,
        growth_rate=32,
        kernel_size=3,
        bottleneck_ratio=4,
    )
    
    x = torch.randn(2, 64, 32, 32)
    output = dense_block(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output channels: {output.shape[1]} (64 + 4*32 = 192)")
    print(f"Output shape: {output.shape}")
    
    # Show parameter count
    total_params = sum(p.numel() for p in dense_block.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Compare with standard sequential blocks
    print("\nComparison with Sequential Blocks:")
    sequential = nn.Sequential(
        nn.Conv2d(64, 96, 3, padding=1),
        nn.Conv2d(96, 128, 3, padding=1),
        nn.Conv2d(128, 160, 3, padding=1),
        nn.Conv2d(160, 192, 3, padding=1),
    )
    seq_params = sum(p.numel() for p in sequential.parameters())
    print(f"Sequential parameters: {seq_params:,}")
    print(f"Efficiency gain: {(1 - total_params/seq_params)*100:.1f}% fewer parameters")
    
    print()


# ============================================================================
# EXAMPLE 4: Channel Attention Mechanism
# ============================================================================

def example_4_channel_attention():
    """
    Demonstrate Channel Attention Mechanism (Squeeze-and-Excitation).
    
    Shows how channel attention learns to reweight channels based on their
    importance for the task.
    """
    print("=" * 70)
    print("EXAMPLE 4: Channel Attention Mechanism")
    print("=" * 70)
    
    # Create channel attention with different reduction ratios
    print("\nChannel Attention with different reduction ratios:\n")
    
    x = torch.randn(2, 256, 32, 32)
    print(f"Input shape: {x.shape}\n")
    
    for reduction in [4, 8, 16, 32]:
        attn = ChannelAttention(channels=256, reduction=reduction)
        output = attn(x)
        print(f"Reduction ratio: 1/{reduction}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in attn.parameters()):,}")
    
    # Visualize attention weights
    print("\nAttention weights visualization:")
    attn = ChannelAttention(channels=64)
    x = torch.randn(1, 64, 16, 16)
    output = attn(x)
    
    # The attn module internally computes weights - we can see the effect
    print(f"  Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  (Channel attention selectively amplifies important channels)")
    
    print()


# ============================================================================
# EXAMPLE 5: Spatial Attention Mechanism
# ============================================================================

def example_5_spatial_attention():
    """
    Demonstrate Spatial Attention Mechanism.
    
    Shows how spatial attention learns to focus on important regions of
    the feature maps.
    """
    print("=" * 70)
    print("EXAMPLE 5: Spatial Attention Mechanism")
    print("=" * 70)
    
    x = torch.randn(2, 64, 32, 32)
    print(f"Input shape: {x.shape}\n")
    
    # Try different kernel sizes
    print("Spatial attention with different kernel sizes:\n")
    
    for kernel_size in [3, 5, 7, 9]:
        attn = SpatialAttention(kernel_size=kernel_size)
        output = attn(x)
        print(f"Kernel size: {kernel_size}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in attn.parameters()):,}")
    
    print("\nKey insight: Spatial attention focuses on 'where' to attend")
    print("  - Larger kernels capture broader context")
    print("  - Smaller kernels focus on local patterns")
    
    print()


# ============================================================================
# EXAMPLE 6: Combined Channel and Spatial Attention
# ============================================================================

def example_6_combined_attention():
    """
    Demonstrate combined Channel-Spatial Attention.
    
    Shows how combining both mechanisms provides comprehensive feature
    recalibration.
    """
    print("=" * 70)
    print("EXAMPLE 6: Combined Channel-Spatial Attention")
    print("=" * 70)
    
    x = torch.randn(4, 128, 32, 32)
    print(f"Input shape: {x.shape}\n")
    
    # Individual mechanisms
    print("1. Channel Attention Only:")
    channel_attn = ChannelAttention(channels=128)
    output1 = channel_attn(x)
    print(f"   Output shape: {output1.shape}")
    print(f"   Parameters: {sum(p.numel() for p in channel_attn.parameters()):,}")
    
    print("\n2. Spatial Attention Only:")
    spatial_attn = SpatialAttention(kernel_size=7)
    output2 = spatial_attn(x)
    print(f"   Output shape: {output2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in spatial_attn.parameters()):,}")
    
    print("\n3. Combined Channel-Spatial Attention:")
    combined_attn = ChannelSpatialAttention(channels=128, spatial_kernel=7)
    output3 = combined_attn(x)
    print(f"   Output shape: {output3.shape}")
    print(f"   Parameters: {sum(p.numel() for p in combined_attn.parameters()):,}")
    
    print("\nBenefits of combined approach:")
    print("  - 'What' to attend (channel) + 'Where' to attend (spatial)")
    print("  - Better feature representation learning")
    print("  - Marginal parameter overhead")
    
    print()


# ============================================================================
# EXAMPLE 7: Multi-Head Attention
# ============================================================================

def example_7_multihead_attention():
    """
    Demonstrate Multi-Head Self-Attention mechanism.
    
    Shows how multiple attention heads learn different representation aspects.
    """
    print("=" * 70)
    print("EXAMPLE 7: Multi-Head Self-Attention")
    print("=" * 70)
    
    # Create attention layer with 8 heads
    attn = MultiHeadAttention(embed_dim=256, num_heads=8)
    
    # Input: batch of 4, sequence length 32, embedding dimension 256
    x = torch.randn(4, 32, 256)
    print(f"Input shape: {x.shape}")
    print(f"  - Batch size: 4")
    print(f"  - Sequence length: 32")
    print(f"  - Embedding dimension: 256")
    print(f"  - Number of heads: 8")
    print(f"  - Head dimension: {256//8}\n")
    
    # Forward pass
    output, attention_weights = attn(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  (batch, num_heads, seq_len, seq_len)")
    print(f"\nParameters: {sum(p.numel() for p in attn.parameters()):,}\n")
    
    # Show that each head attends differently
    print("Multi-head behavior:")
    print(f"  - Head 0 mean attention: {attention_weights[0, 0].mean():.4f}")
    print(f"  - Head 1 mean attention: {attention_weights[0, 1].mean():.4f}")
    print(f"  - Head 7 mean attention: {attention_weights[0, 7].mean():.4f}")
    print(f"  (Different heads learn different attention patterns)")
    
    print()


# ============================================================================
# EXAMPLE 8: Positional Encoding for Transformers
# ============================================================================

def example_8_positional_encoding():
    """
    Demonstrate Positional Encoding for sequence models.
    
    Shows how positional encodings provide position information to the model.
    """
    print("=" * 70)
    print("EXAMPLE 8: Positional Encoding")
    print("=" * 70)
    
    # Create positional encoding
    pos_enc = PositionalEncoding(d_model=256, max_len=5000)
    
    # Create sample embeddings
    x = torch.randn(2, 64, 256)  # (batch, seq_len, d_model)
    print(f"Input embeddings shape: {x.shape}\n")
    
    # Get positional encodings
    pe = pos_enc(x)
    print(f"Positional encoding shape: {pe.shape}")
    print(f"Positional encoding added to embeddings: x + pe")
    
    # Show encoding properties
    print(f"\nPositional encoding properties:")
    print(f"  - D-model: 256")
    print(f"  - Max sequence length: 5000")
    print(f"  - Even indices: sin(pos/10000^(2i/d))")
    print(f"  - Odd indices: cos(pos/10000^(2i/d))")
    print(f"  - Provides absolute position information to transformer")
    
    print()


# ============================================================================
# EXAMPLE 9: Grid Search for Architecture Discovery
# ============================================================================

def example_9_grid_search():
    """
    Demonstrate Grid Search for exploring architecture space.
    
    Shows systematic exploration of all combinations of hyperparameters.
    """
    print("=" * 70)
    print("EXAMPLE 9: Grid Search Architecture Exploration")
    print("=" * 70)
    
    # Define search space
    search_space = {
        'num_conv_blocks': [2, 3, 4],
        'activations': ['relu', 'gelu'],
        'dropout_rates': [0.0, 0.1],
    }
    
    # Create base configuration
    base_config = ArchitectureConfig(
        input_shape=(3, 224, 224),
        num_classes=1000,
    ).to_dict()
    
    # Run grid search
    searcher = GridSearch(search_space, base_config)
    print(f"Search space: {search_space}")
    print(f"Total combinations: {searcher.num_combinations}\n")
    
    print("First 5 configurations:")
    for idx, config in enumerate(searcher.generate_with_index(), 1):
        if idx <= 5:
            idx, cfg = config
            print(f"  {idx}. Blocks={cfg.num_conv_blocks}, Activation={cfg.activations}, "
                  f"Dropout={cfg.dropout_rates}")
        else:
            break
    
    print(f"  ... and {searcher.num_combinations - 5} more")
    
    print()


# ============================================================================
# EXAMPLE 10: Random Search for Efficient Exploration
# ============================================================================

def example_10_random_search():
    """
    Demonstrate Random Search for efficient architecture exploration.
    
    Shows random sampling from search space for diversity.
    """
    print("=" * 70)
    print("EXAMPLE 10: Random Search Architecture Discovery")
    print("=" * 70)
    
    # Define search space
    search_space = {
        'num_conv_blocks': [2, 3, 4, 5, 6],
        'activations': ['relu', 'gelu', 'leaky_relu', 'silu'],
        'dropout_rates': [0.0, 0.1, 0.2, 0.3],
    }
    
    # Create base configuration
    base_config = ArchitectureConfig(
        input_shape=(3, 224, 224),
        num_classes=1000,
    ).to_dict()
    
    # Run random search
    searcher = RandomSearch(search_space, num_samples=10, base_config=base_config, seed=42)
    print(f"Search space combinations: 5 * 4 * 4 = 80")
    print(f"Random samples: 10\n")
    
    print("Sampled configurations:")
    for idx, config in enumerate(searcher.generate_with_index(), 1):
        idx, cfg = config
        print(f"  {idx:2d}. Blocks={cfg.num_conv_blocks}, Activation={cfg.activations}, "
              f"Dropout={cfg.dropout_rates}")
    
    print()


# ============================================================================
# EXAMPLE 11: Architecture Factory and Scoring
# ============================================================================

def example_11_architecture_factory():
    """
    Demonstrate Architecture Factory for creating models from patterns.
    
    Shows how to generate architectures from predefined patterns.
    """
    print("=" * 70)
    print("EXAMPLE 11: Architecture Factory")
    print("=" * 70)
    
    factory = ArchitectureFactory()
    
    print("Available patterns:")
    for name in factory.patterns.keys():
        print(f"  - {name}")
    
    # Create different architecture types
    base_config = ArchitectureConfig(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
    )
    
    print("\nGenerated architectures:\n")
    
    patterns = ['sequential', 'residual', 'dense', 'mixed']
    for pattern in patterns:
        base_config.pattern = pattern
        arch = factory.create(base_config)
        print(f"{pattern.upper()}:")
        print(f"  Type: {arch['type']}")
        for key, value in arch.items():
            if key != 'type':
                print(f"  {key}: {value}")
        print()


# ============================================================================
# EXAMPLE 12: Architecture Scoring and Comparison
# ============================================================================

def example_12_architecture_comparison():
    """
    Demonstrate Architecture Scoring and Comparison.
    
    Shows how to evaluate and compare different architectures.
    """
    print("=" * 70)
    print("EXAMPLE 12: Architecture Scoring and Comparison")
    print("=" * 70)
    
    # Create multiple configurations
    configs = [
        ArchitectureConfig(
            input_shape=(3, 224, 224),
            num_classes=1000,
            num_conv_blocks=2,
        ),
        ArchitectureConfig(
            input_shape=(3, 224, 224),
            num_classes=1000,
            num_conv_blocks=4,
        ),
        ArchitectureConfig(
            input_shape=(3, 224, 224),
            num_classes=1000,
            num_conv_blocks=6,
        ),
    ]
    
    print("Configurations:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.num_conv_blocks} blocks")
    
    # Create scorer
    scorer = ArchitectureScorer(max_parameters=50_000_000)
    print(f"\nScoringConstraints:")
    print(f"  Max parameters: 50,000,000")
    
    # Compare architectures
    comparator = ArchitectureComparator(configs, scorer)
    
    print(f"\nScores:")
    for i, (cfg, score) in enumerate(zip(configs, comparator.scores), 1):
        print(f"  {i}. {cfg.num_conv_blocks} blocks: {score:.4f}")
    
    # Get statistics
    stats = comparator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Std: {stats['std']:.4f}")
    
    # Get best
    best_cfg, best_score = comparator.get_best()
    print(f"\nBest configuration: {best_cfg.num_conv_blocks} blocks (score: {best_score:.4f})")
    
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("=" * 70)
    print("STEP 6: ADVANCED ARCHITECTURE FEATURES - EXAMPLES".center(70))
    print("=" * 70)
    print()
    
    # Run all examples
    example_1_residual_connector()
    example_2_skip_connection_patterns()
    example_3_dense_connection_block()
    example_4_channel_attention()
    example_5_spatial_attention()
    example_6_combined_attention()
    example_7_multihead_attention()
    example_8_positional_encoding()
    example_9_grid_search()
    example_10_random_search()
    example_11_architecture_factory()
    example_12_architecture_comparison()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
