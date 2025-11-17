"""Model summary and introspection utilities - Usage Examples.

This file demonstrates all features of the utilities module including:
- Model summaries
- Parameter analysis
- Memory estimation
- Architecture validation
- FLOPs calculation
"""

import torch
from torchvision_customizer import CustomCNN
from torchvision_customizer.utils import (
    calculate_output_shape,
    count_parameters_by_type,
    estimate_model_size,
    get_layer_summary,
    get_memory_usage,
    get_model_flops,
    predict_memory_usage,
    print_model_summary,
    validate_architecture,
    validate_channel_progression,
    validate_spatial_dimensions,
)


def example_1_basic_model_summary():
    """Example 1: Print complete model summary."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Model Summary")
    print("=" * 70)
    
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=5
    )
    
    summary = print_model_summary(
        model,
        input_shape=(1, 3, 224, 224),
        verbose=True
    )


def example_2_calculate_output_shape():
    """Example 2: Calculate output shape for different batch sizes."""
    print("\n" + "=" * 70)
    print("Example 2: Calculate Output Shapes")
    print("=" * 70)
    
    model = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=10
    )
    
    print("\nInput shape effects on output:")
    for batch_size in [1, 2, 4, 8, 16]:
        result = calculate_output_shape(
            model,
            input_shape=(batch_size, 3, 32, 32)
        )
        print(f"  Batch size {batch_size:2d}: Output shape {result['output_shape']}")


def example_3_layer_summary():
    """Example 3: Get layer-by-layer summary."""
    print("\n" + "=" * 70)
    print("Example 3: Layer-by-Layer Summary")
    print("=" * 70)
    
    model = CustomCNN(
        input_shape=(3, 64, 64),
        num_classes=10,
        num_conv_blocks=2
    )
    
    layers = get_layer_summary(
        model,
        input_shape=(2, 3, 64, 64)
    )
    
    print(f"\nTotal layers: {len(layers)}\n")
    print(f"{'Layer Name':<50} {'Type':<20} {'Output Shape':<20}")
    print("-" * 90)
    
    for layer in layers[:10]:  # Show first 10 layers
        name = layer['name'][-40:] if len(layer['name']) > 40 else layer['name']
        layer_type = layer['type']
        output_shape = str(layer['output_shape'])
        print(f"{name:<50} {layer_type:<20} {output_shape:<20}")


def example_4_parameter_analysis():
    """Example 4: Analyze parameters by layer type."""
    print("\n" + "=" * 70)
    print("Example 4: Parameter Analysis by Type")
    print("=" * 70)
    
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4
    )
    
    param_count = count_parameters_by_type(model)
    
    print(f"\n{'Layer Type':<20} {'Count':<10} {'Parameters':<20} {'Trainable':<20}")
    print("-" * 70)
    
    total_params = 0
    total_trainable = 0
    
    for layer_type, counts in sorted(param_count.items()):
        print(f"{layer_type:<20} {counts['count']:<10} "
              f"{counts['total']:>18,} {counts['trainable']:>18,}")
        total_params += counts['total']
        total_trainable += counts['trainable']
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {'':<10} {total_params:>18,} {total_trainable:>18,}")


def example_5_memory_usage():
    """Example 5: Estimate memory usage."""
    print("\n" + "=" * 70)
    print("Example 5: Memory Usage Estimation")
    print("=" * 70)
    
    configurations = [
        ("CIFAR-10 (Small)", (3, 32, 32), 2),
        ("ImageNet (Medium)", (3, 224, 224), 1),
        ("High-Res (Large)", (3, 512, 512), 1),
    ]
    
    model = CustomCNN(
        input_shape=(3, 32, 32),
        num_classes=1000,
        num_conv_blocks=4
    )
    
    print("\nMemory usage for different input sizes:")
    
    for name, input_shape, batch_size in configurations:
        full_input = (batch_size,) + input_shape
        memory = get_memory_usage(model, full_input)
        
        print(f"\n{name}:")
        print(f"  Input shape: {input_shape}")
        print(f"  Batch size: {batch_size}")
        print(f"  Parameter memory: {memory['parameter_memory_mb']} MB")
        print(f"  Activation memory: {memory['activation_memory_mb']} MB")
        print(f"  Total memory: {memory['total_memory_mb']} MB")


def example_6_flops_estimation():
    """Example 6: Estimate FLOPs."""
    print("\n" + "=" * 70)
    print("Example 6: FLOPs Estimation")
    print("=" * 70)
    
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4
    )
    
    print("\nFLOPs for different batch sizes:")
    
    for batch_size in [1, 4, 8, 16]:
        flops = get_model_flops(
            model,
            input_shape=(batch_size, 3, 224, 224)
        )
        print(f"  Batch size {batch_size:2d}: "
              f"{flops['total_flops']:>15,} FLOPs "
              f"({flops['total_flops_in_billions']} B)")


def example_7_validate_spatial_dimensions():
    """Example 7: Validate spatial dimensions."""
    print("\n" + "=" * 70)
    print("Example 7: Spatial Dimension Validation")
    print("=" * 70)
    
    test_cases = [
        ((3, 224, 224), 4, "Valid - Large input"),
        ((3, 64, 64), 3, "Valid - Medium input"),
        ((3, 32, 32), 6, "Warning - Many pooling ops"),
        ((3, 16, 16), 8, "Invalid - Spatial dimensions too small"),
    ]
    
    for input_shape, num_blocks, description in test_cases:
        result = validate_spatial_dimensions(
            input_shape=input_shape,
            num_conv_blocks=num_blocks
        )
        
        print(f"\n{description}:")
        print(f"  Input: {input_shape}, Blocks: {num_blocks}")
        print(f"  Valid: {result.get('valid', 'N/A')}")
        print(f"  Final spatial: {result.get('final_spatial', 'N/A')}")
        
        if result.get('warnings'):
            print(f"  Warnings:")
            for warning in result['warnings']:
                print(f"    - {warning}")
        if 'error' in result:
            print(f"  Error: {result['error']}")


def example_8_validate_channel_progression():
    """Example 8: Validate channel progression."""
    print("\n" + "=" * 70)
    print("Example 8: Channel Progression Validation")
    print("=" * 70)
    
    test_cases = [
        ([64, 128, 256, 512], "Doubling progression"),
        ([32, 64, 128, 256], "Controlled growth"),
        ([64, 128, 256, 256], "Plateau at end"),
        ([512, 256, 128, 64], "Decreasing channels"),
        ([64, 512, 4096, 32768], "Exponential growth"),
    ]
    
    for channels, description in test_cases:
        result = validate_channel_progression(
            channels=channels,
            num_conv_blocks=len(channels)
        )
        
        print(f"\n{description}: {channels}")
        print(f"  Valid: {result.get('valid', 'N/A')}")
        print(f"  Growth rate: {result.get('growth_rate', 'N/A')}")
        print(f"  Min channels: {result.get('min_channels', 'N/A')}")
        print(f"  Max channels: {result.get('max_channels', 'N/A')}")
        
        if result.get('warnings'):
            for warning in result['warnings']:
                print(f"  Warning: {warning}")


def example_9_validate_architecture():
    """Example 9: Complete architecture validation."""
    print("\n" + "=" * 70)
    print("Example 9: Complete Architecture Validation")
    print("=" * 70)
    
    # Valid architecture
    print("\nValidating CIFAR-10 architecture:")
    result = validate_architecture(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_conv_blocks=4,
        channels=[32, 64, 128, 256]
    )
    print(f"  Valid: {result['valid']}")
    print(f"  Errors: {result['errors']}")
    print(f"  Warnings: {result['warnings']}")
    
    # Invalid architecture
    print("\nValidating problematic architecture:")
    result = validate_architecture(
        input_shape=(3, 16, 16),
        num_classes=1000,
        num_conv_blocks=10,
        channels=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    )
    print(f"  Valid: {result['valid']}")
    if result['errors']:
        print(f"  Errors:")
        for error in result['errors']:
            print(f"    - {error}")
    if result['warnings']:
        print(f"  Warnings:")
        for warning in result['warnings']:
            print(f"    - {warning}")


def example_10_estimate_model_size():
    """Example 10: Estimate model size without creating it."""
    print("\n" + "=" * 70)
    print("Example 10: Model Size Estimation")
    print("=" * 70)
    
    configs = [
        ("Small", 2, [32, 64], 10),
        ("Medium", 4, [64, 128, 256, 512], 100),
        ("Large", 5, [64, 128, 256, 512, 512], 1000),
    ]
    
    for name, num_blocks, channels, num_classes in configs:
        estimate = estimate_model_size(
            num_conv_blocks=num_blocks,
            channels=channels,
            num_classes=num_classes
        )
        
        print(f"\n{name} Model:")
        print(f"  Conv params: {estimate['conv_params']:>15,}")
        print(f"  BN params:   {estimate['bn_params']:>15,}")
        print(f"  Linear params: {estimate['linear_params']:>15,}")
        print(f"  Total params: {estimate['total_params']:>15,}")


def example_11_predict_memory_usage():
    """Example 11: Predict memory usage without creating model."""
    print("\n" + "=" * 70)
    print("Example 11: Memory Usage Prediction")
    print("=" * 70)
    
    print("\nMemory requirements for training:")
    
    configs = [
        ("CIFAR-10", 4, [64, 128, 256, 512], 10),
        ("ImageNet", 5, [64, 128, 256, 512, 512], 1000),
    ]
    
    for name, num_blocks, channels, num_classes in configs:
        memory = predict_memory_usage(
            num_conv_blocks=num_blocks,
            channels=channels,
            num_classes=num_classes,
            batch_size=32
        )
        
        print(f"\n{name} (batch_size=32):")
        print(f"  Parameters: {memory['parameter_memory_mb']:.2f} MB")
        print(f"  Activations: {memory['activation_memory_mb']:.2f} MB")
        print(f"  Gradients: {memory['gradient_memory_mb']:.2f} MB")
        print(f"  Total (training): {memory['total_training_memory_mb']:.2f} MB")
        print(f"  Total (inference): {memory['inference_memory_mb']:.2f} MB")


def example_12_comprehensive_analysis():
    """Example 12: Comprehensive model analysis."""
    print("\n" + "=" * 70)
    print("Example 12: Comprehensive Model Analysis")
    print("=" * 70)
    
    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
        channels=[64, 128, 256, 512]
    )
    
    print("\n1. Model Configuration:")
    config = model.get_config()
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n2. Complete Summary:")
    summary = print_model_summary(
        model,
        input_shape=(1, 3, 224, 224),
        verbose=False
    )
    
    print("\n3. Architecture Validation:")
    validation = validate_architecture(
        input_shape=(3, 224, 224),
        num_classes=1000,
        num_conv_blocks=4,
        channels=[64, 128, 256, 512]
    )
    print(f"   Valid: {validation['valid']}")
    if validation['warnings']:
        print("   Warnings:")
        for w in validation['warnings']:
            print(f"     - {w}")
    
    print("\n4. Performance Metrics:")
    print(f"   Total parameters: {summary['total_parameters']:,}")
    print(f"   Memory (training): {summary['memory'].get('total_memory_mb')} MB")
    print(f"   FLOPs: {summary['flops'].get('total_flops_in_billions')} B")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# MODEL SUMMARY AND INTROSPECTION UTILITIES - EXAMPLES")
    print("#" * 70)
    
    example_1_basic_model_summary()
    example_2_calculate_output_shape()
    example_3_layer_summary()
    example_4_parameter_analysis()
    example_5_memory_usage()
    example_6_flops_estimation()
    example_7_validate_spatial_dimensions()
    example_8_validate_channel_progression()
    example_9_validate_architecture()
    example_10_estimate_model_size()
    example_11_predict_memory_usage()
    example_12_comprehensive_analysis()
    
    print("\n" + "#" * 70)
    print("# ALL EXAMPLES COMPLETED")
    print("#" * 70 + "\n")
