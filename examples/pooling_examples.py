"""Examples demonstrating flexible pooling options in torchvision-customizer.

This module provides practical examples for:
    - Using different pooling types
    - Combining pooling with dropout
    - Integrating pooling into models
    - Calculating output sizes
    - Stochastic pooling for regularization
    - Multi-scale pooling strategies
"""

import torch
import torch.nn as nn
from torchvision_customizer.layers import (
    get_pooling,
    PoolingBlock,
    PoolingFactory,
    StochasticPool2d,
    LPPool2d,
    calculate_pooling_output_size,
    validate_pooling_config,
)


def example_1_basic_pooling():
    """Example 1: Basic pooling operations."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Pooling Operations")
    print("="*70)
    
    x = torch.randn(2, 64, 32, 32)
    print(f"Input shape: {x.shape}")
    
    # Max pooling
    max_pool = get_pooling('max', kernel_size=2, stride=2)
    max_out = max_pool(x)
    print(f"Max pooling output: {max_out.shape}")
    
    # Average pooling
    avg_pool = get_pooling('avg', kernel_size=2, stride=2)
    avg_out = avg_pool(x)
    print(f"Average pooling output: {avg_out.shape}")
    
    # Adaptive max pooling
    adaptive_max = get_pooling('adaptive_max', output_size=(7, 7))
    adaptive_max_out = adaptive_max(x)
    print(f"Adaptive max pooling (7x7): {adaptive_max_out.shape}")
    
    # Adaptive average pooling
    adaptive_avg = get_pooling('adaptive_avg', output_size=(1, 1))
    adaptive_avg_out = adaptive_avg(x)
    print(f"Adaptive average pooling (1x1): {adaptive_avg_out.shape}")


def example_2_pooling_block_with_dropout():
    """Example 2: PoolingBlock with dropout regularization."""
    print("\n" + "="*70)
    print("EXAMPLE 2: PoolingBlock with Dropout")
    print("="*70)
    
    # Create pooling blocks with different dropout rates
    pool_no_dropout = PoolingBlock('max', kernel_size=2, dropout_rate=0.0)
    pool_light_dropout = PoolingBlock('max', kernel_size=2, dropout_rate=0.2)
    pool_heavy_dropout = PoolingBlock('avg', kernel_size=2, dropout_rate=0.5)
    
    x = torch.randn(2, 64, 32, 32)
    print(f"Input shape: {x.shape}")
    
    out_no_dropout = pool_no_dropout(x)
    print(f"No dropout output: {out_no_dropout.shape}, dropout layer: {pool_no_dropout.dropout}")
    
    pool_light_dropout.train()
    out_light = pool_light_dropout(x)
    print(f"Light dropout (0.2) output: {out_light.shape}, dropout: {type(pool_light_dropout.dropout).__name__}")
    
    pool_heavy_dropout.train()
    out_heavy = pool_heavy_dropout(x)
    print(f"Heavy dropout (0.5) output: {out_heavy.shape}, dropout: {type(pool_heavy_dropout.dropout).__name__}")


def example_3_stochastic_pooling():
    """Example 3: Stochastic pooling for regularization."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Stochastic Pooling")
    print("="*70)
    
    x = torch.randn(2, 64, 32, 32)
    print(f"Input shape: {x.shape}")
    
    # Stochastic pooling
    stoch_pool = StochasticPool2d(kernel_size=2, stride=2)
    
    # Training mode (stochastic)
    stoch_pool.train()
    out_train_1 = stoch_pool(x)
    out_train_2 = stoch_pool(x)
    print(f"Training mode output 1: {out_train_1.shape}")
    print(f"Training mode output 2: {out_train_2.shape}")
    print(f"Outputs differ (stochastic): {not torch.equal(out_train_1, out_train_2)}")
    
    # Eval mode (deterministic - uses max pooling)
    stoch_pool.eval()
    out_eval_1 = stoch_pool(x)
    out_eval_2 = stoch_pool(x)
    print(f"Eval mode output 1: {out_eval_1.shape}")
    print(f"Eval mode output 2: {out_eval_2.shape}")
    print(f"Outputs identical (deterministic): {torch.equal(out_eval_1, out_eval_2)}")


def example_4_lp_pooling():
    """Example 4: L-p norm pooling (RMS pooling)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: L-p Norm Pooling (RMS)")
    print("="*70)
    
    x = torch.randn(2, 64, 32, 32)
    print(f"Input shape: {x.shape}")
    
    # L2 norm (RMS) pooling
    lp_pool = LPPool2d(norm_type=2, kernel_size=2, stride=2)
    lp_out = lp_pool(x)
    print(f"L2 norm pooling (RMS) output: {lp_out.shape}")
    
    # L1 norm pooling
    lp_pool_l1 = LPPool2d(norm_type=1, kernel_size=2, stride=2)
    lp_out_l1 = lp_pool_l1(x)
    print(f"L1 norm pooling output: {lp_out_l1.shape}")


def example_5_pooling_factory():
    """Example 5: Using PoolingFactory."""
    print("\n" + "="*70)
    print("EXAMPLE 5: PoolingFactory")
    print("="*70)
    
    factory = PoolingFactory()
    x = torch.randn(2, 64, 32, 32)
    print(f"Input shape: {x.shape}")
    
    # List supported pooling types
    supported = factory.supported_pooling()
    print(f"Supported pooling types: {supported}")
    
    # Create and use different pooling types
    for pool_type in ['max', 'avg', 'adaptive_max', 'none']:
        if pool_type == 'adaptive_max':
            pool = factory.create(pool_type, output_size=(1, 1))
        else:
            pool = factory.create(pool_type, kernel_size=2, stride=2)
        out = pool(x)
        print(f"  {pool_type}: {out.shape}")


def example_6_output_size_calculation():
    """Example 6: Calculating output sizes."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Output Size Calculation")
    print("="*70)
    
    # Calculate output sizes for different configurations
    configs = [
        (32, 2, 2, 0, "32x32 input, 2x2 kernel, stride 2, no padding"),
        (64, 3, 2, 1, "64x64 input, 3x3 kernel, stride 2, padding 1"),
        (224, 7, 2, 0, "224x224 input, 7x7 kernel, stride 2, no padding"),
        (32, 2, 1, 0, "32x32 input, 2x2 kernel, stride 1, no padding"),
    ]
    
    for input_size, kernel, stride, padding, desc in configs:
        output_size = calculate_pooling_output_size(
            input_size=input_size,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        print(f"{desc}: {input_size} → {output_size}")


def example_7_pooling_configuration_validation():
    """Example 7: Validating pooling configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Pooling Configuration Validation")
    print("="*70)
    
    # Test valid configuration
    is_valid, msg = validate_pooling_config(
        pool_type='max', kernel_size=2, stride=2,
        input_height=32, input_width=32
    )
    print(f"Config (2x2 pool, 32x32 input): {msg}")
    
    # Test invalid configuration (kernel too large)
    is_valid, msg = validate_pooling_config(
        pool_type='max', kernel_size=64, stride=2,
        input_height=32, input_width=32
    )
    print(f"Config (64x64 pool, 32x32 input): {msg}")
    
    # Test edge case (small output)
    is_valid, msg = validate_pooling_config(
        pool_type='max', kernel_size=3, stride=3,
        input_height=4, input_width=4
    )
    print(f"Config (3x3 pool, stride 3, 4x4 input): {msg}")


def example_8_pooling_in_cnn():
    """Example 8: Pooling in complete CNN architecture."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Pooling in CNN Architecture")
    print("="*70)
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                PoolingBlock('max', kernel_size=2, dropout_rate=0.1),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                PoolingBlock('avg', kernel_size=2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                PoolingBlock('adaptive_avg', output_size=(1, 1)),
            )
            self.classifier = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = SimpleCNN()
    model.eval()
    
    x = torch.randn(2, 3, 64, 64)
    print(f"Input shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


def example_9_adaptive_pooling_comparison():
    """Example 9: Comparing different adaptive pooling sizes."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Adaptive Pooling Comparison")
    print("="*70)
    
    # Test with variable input sizes
    input_sizes = [
        (32, 32),
        (48, 64),
        (224, 224),
        (112, 224),
    ]
    
    adaptive_max = PoolingBlock('adaptive_max', output_size=(7, 7))
    
    for h, w in input_sizes:
        x = torch.randn(2, 64, h, w)
        out = adaptive_max(x)
        h_out, w_out = adaptive_max.calculate_output_size(h, w)
        print(f"Input: {h}x{w} → Output: {out.shape}, Calculated: ({h_out}, {w_out})")


def example_10_multi_stage_pooling():
    """Example 10: Multi-stage pooling with progressive size reduction."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Multi-Stage Pooling")
    print("="*70)
    
    class MultiStagePool(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool1 = PoolingBlock('max', kernel_size=2, stride=2)
            self.pool2 = PoolingBlock('avg', kernel_size=2, stride=2)
            self.pool3 = PoolingBlock('adaptive_max', output_size=(1, 1))
        
        def forward(self, x):
            sizes = [x.shape]
            x = self.pool1(x)
            sizes.append(x.shape)
            x = self.pool2(x)
            sizes.append(x.shape)
            x = self.pool3(x)
            sizes.append(x.shape)
            return x, sizes
    
    model = MultiStagePool()
    model.eval()
    
    x = torch.randn(2, 128, 64, 64)
    out, sizes = model(x)
    
    print(f"Input shape: {x.shape}")
    for i, size in enumerate(sizes):
        print(f"  After stage {i}: {size}")
    print(f"Final output shape: {out.shape}")


def example_11_pooling_memory_efficiency():
    """Example 11: Comparing memory usage of different pooling types."""
    print("\n" + "="*70)
    print("EXAMPLE 11: Memory Efficiency of Pooling Types")
    print("="*70)
    
    x = torch.randn(16, 256, 32, 32)
    print(f"Input shape: {x.shape}")
    print(f"Input memory: {x.element_size() * x.nelement() / 1024 / 1024:.2f} MB")
    
    pooling_types = [
        ('max', {}),
        ('avg', {}),
        ('adaptive_max', {'output_size': (1, 1)}),
        ('stochastic', {}),
    ]
    
    for pool_type, kwargs in pooling_types:
        if pool_type == 'stochastic':
            pool = StochasticPool2d(kernel_size=2, stride=2)
            pool.eval()
        else:
            pool = get_pooling(pool_type, kernel_size=2, stride=2, **kwargs)
        
        out = pool(x)
        memory_mb = out.element_size() * out.nelement() / 1024 / 1024
        reduction = (1 - out.numel() / x.numel()) * 100
        print(f"{pool_type}: {out.shape} ({memory_mb:.2f} MB, {reduction:.1f}% reduction)")


def example_12_custom_pooling_block():
    """Example 12: Creating custom pooling combinations."""
    print("\n" + "="*70)
    print("EXAMPLE 12: Custom Pooling Combinations")
    print("="*70)
    
    class DualPooling(nn.Module):
        """Combines max and average pooling."""
        def __init__(self, kernel_size=2, stride=2):
            super().__init__()
            self.max_pool = get_pooling('max', kernel_size=kernel_size, stride=stride)
            self.avg_pool = get_pooling('avg', kernel_size=kernel_size, stride=stride)
        
        def forward(self, x):
            max_out = self.max_pool(x)
            avg_out = self.avg_pool(x)
            # Concatenate along channel dimension
            return torch.cat([max_out, avg_out], dim=1)
    
    model = DualPooling(kernel_size=2, stride=2)
    x = torch.randn(2, 64, 32, 32)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape} (channels doubled by concatenation)")
    print(f"Max pool shape: (2, 64, 16, 16)")
    print(f"Avg pool shape: (2, 64, 16, 16)")
    print(f"Concatenated shape: {out.shape}")


def main():
    """Run all examples."""
    examples = [
        example_1_basic_pooling,
        example_2_pooling_block_with_dropout,
        example_3_stochastic_pooling,
        example_4_lp_pooling,
        example_5_pooling_factory,
        example_6_output_size_calculation,
        example_7_pooling_configuration_validation,
        example_8_pooling_in_cnn,
        example_9_adaptive_pooling_comparison,
        example_10_multi_stage_pooling,
        example_11_pooling_memory_efficiency,
        example_12_custom_pooling_block,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {str(e)}")
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
