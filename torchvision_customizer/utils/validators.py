"""Architecture validation utilities.

Provides tools for validating CNN architectures and configurations,
including dimension tracking and compatibility checking.

Example:
    >>> from torchvision_customizer.utils import validate_architecture
    >>> validate_architecture(input_shape=(3, 224, 224), num_conv_blocks=5)
"""

from typing import Dict, List, Optional, Tuple

import torch.nn as nn


def validate_spatial_dimensions(
    input_shape: Tuple[int, int, int],
    num_conv_blocks: int,
    pooling_kernel_size: int = 2,
    pooling_stride: int = 2,
    min_final_size: int = 4,
) -> Dict[str, any]:
    """Validate that spatial dimensions don't become too small.

    Args:
        input_shape: Input shape as (C, H, W).
        num_conv_blocks: Number of convolutional blocks.
        pooling_kernel_size: Pooling kernel size.
        pooling_stride: Pooling stride.
        min_final_size: Minimum acceptable final spatial size.

    Returns:
        Dictionary containing:
            - 'valid': Whether dimensions are valid
            - 'input_spatial': Original spatial dimensions
            - 'final_spatial': Final spatial dimensions after pooling
            - 'num_pooling_ops': Number of pooling operations
            - 'warnings': List of warning messages
    """
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
        return {
            "valid": False,
            "error": f"input_shape must be 3D (C, H, W), got {input_shape}",
        }

    channels, height, width = input_shape

    if height <= 0 or width <= 0:
        return {
            "valid": False,
            "error": f"Height and width must be positive, got ({height}, {width})",
        }

    warnings = []

    # Estimate final spatial size after pooling
    # Each pooling operation reduces by stride, but also limited by kernel
    final_height = height
    final_width = width

    for _ in range(num_conv_blocks):
        # Simulate pooling: output_size = (input_size - kernel_size) // stride + 1
        # Simplified: output_size = input_size // stride
        final_height = (final_height - pooling_kernel_size) // pooling_stride + 1
        final_width = (final_width - pooling_kernel_size) // pooling_stride + 1

    valid = final_height >= min_final_size and final_width >= min_final_size

    if final_height < min_final_size or final_width < min_final_size:
        warnings.append(
            f"Final spatial size ({final_height}x{final_width}) is smaller than "
            f"recommended minimum ({min_final_size}x{min_final_size})"
        )

    if final_height < 1 or final_width < 1:
        warnings.append("Final spatial dimensions reduced to 0 - model may fail")
        valid = False

    return {
        "valid": valid,
        "input_spatial": (height, width),
        "final_spatial": (final_height, final_width),
        "num_pooling_ops": num_conv_blocks,
        "warnings": warnings,
    }


def validate_channel_progression(
    channels: List[int],
    num_conv_blocks: int,
    max_channels: int = 2048,
) -> Dict[str, any]:
    """Validate channel progression configuration.

    Args:
        channels: List of channel sizes per block.
        num_conv_blocks: Expected number of blocks.
        max_channels: Maximum allowed channels per layer.

    Returns:
        Dictionary containing:
            - 'valid': Whether configuration is valid
            - 'warnings': List of warning messages
            - 'growth_rate': Average growth per block
    """
    warnings = []

    # Check length
    if len(channels) != num_conv_blocks:
        return {
            "valid": False,
            "error": f"Channel list length ({len(channels)}) doesn't match "
            f"num_conv_blocks ({num_conv_blocks})",
        }

    # Check channel values
    if not all(isinstance(c, int) and c > 0 for c in channels):
        return {
            "valid": False,
            "error": "All channels must be positive integers",
        }

    # Check for too large channels
    for i, c in enumerate(channels):
        if c > max_channels:
            warnings.append(f"Block {i}: channels ({c}) exceeds max ({max_channels})")

    # Calculate growth rate
    if len(channels) > 1:
        growth_rate = (channels[-1] / channels[0]) ** (1 / (len(channels) - 1))
    else:
        growth_rate = 1.0

    # Warn about unusual growth rates
    if growth_rate > 4:
        warnings.append(
            f"High growth rate ({growth_rate:.2f}) may cause memory issues"
        )
    elif growth_rate < 0.5:
        warnings.append(f"Decreasing channels ({growth_rate:.2f}) is unusual")

    return {
        "valid": True,
        "warnings": warnings,
        "growth_rate": round(growth_rate, 4),
        "min_channels": min(channels),
        "max_channels": max(channels),
        "total_params": sum(c for c in channels),
    }


def validate_architecture(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    num_conv_blocks: int,
    channels: Optional[List[int]] = None,
    pooling_kernel_size: int = 2,
    pooling_stride: int = 2,
) -> Dict[str, any]:
    """Validate complete architecture configuration.

    Args:
        input_shape: Input shape as (C, H, W).
        num_classes: Number of output classes.
        num_conv_blocks: Number of convolutional blocks.
        channels: Channel progression (optional).
        pooling_kernel_size: Pooling kernel size.
        pooling_stride: Pooling stride.

    Returns:
        Dictionary containing validation results for all checks.
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    # Check input shape
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
        results["valid"] = False
        results["errors"].append(f"Invalid input_shape: {input_shape}")
        return results

    # Check num_classes
    if not isinstance(num_classes, int) or num_classes <= 0:
        results["valid"] = False
        results["errors"].append(f"Invalid num_classes: {num_classes}")
        return results

    # Check num_conv_blocks
    if not isinstance(num_conv_blocks, int) or num_conv_blocks <= 0:
        results["valid"] = False
        results["errors"].append(f"Invalid num_conv_blocks: {num_conv_blocks}")
        return results

    # Validate spatial dimensions
    spatial_check = validate_spatial_dimensions(
        input_shape,
        num_conv_blocks,
        pooling_kernel_size,
        pooling_stride,
    )
    results["checks"]["spatial_dimensions"] = spatial_check
    if not spatial_check.get("valid", True):
        results["valid"] = False
        results["errors"].append(spatial_check.get("error", "Spatial dimension error"))
    results["warnings"].extend(spatial_check.get("warnings", []))

    # Validate channels if provided
    if channels is not None:
        channel_check = validate_channel_progression(channels, num_conv_blocks)
        results["checks"]["channels"] = channel_check
        if not channel_check.get("valid", True):
            results["valid"] = False
            results["errors"].append(channel_check.get("error", "Channel error"))
        results["warnings"].extend(channel_check.get("warnings", []))

    return results


def estimate_model_size(
    num_conv_blocks: int,
    channels: List[int],
    num_classes: int,
    include_batch_norm: bool = True,
    include_dropout: bool = True,
) -> Dict[str, any]:
    """Estimate model size without actually creating it.

    Args:
        num_conv_blocks: Number of convolutional blocks.
        channels: Channel progression.
        num_classes: Number of output classes.
        include_batch_norm: Whether batch norm is included.
        include_dropout: Whether dropout is included.

    Returns:
        Dictionary containing estimated parameter counts.
    """
    estimates = {
        "conv_params": 0,
        "bn_params": 0,
        "linear_params": 0,
        "total_params": 0,
    }

    # Estimate conv layers (simplified: 3x3 kernels)
    # Conv2d params = (kernel_h * kernel_w * in_channels + 1) * out_channels
    input_channels = 3  # Assume RGB input
    kernel_size = 3

    for out_channels in channels:
        conv_params = (kernel_size * kernel_size * input_channels + 1) * out_channels
        estimates["conv_params"] += conv_params
        input_channels = out_channels

        # Batch norm params (2 * num_features if trainable)
        if include_batch_norm:
            bn_params = 2 * out_channels
            estimates["bn_params"] += bn_params

    # Estimate linear layer
    # Assume global average pooling reduces to (batch, channels, 1, 1)
    linear_params = (channels[-1] + 1) * num_classes
    estimates["linear_params"] += linear_params

    estimates["total_params"] = (
        estimates["conv_params"] + estimates["bn_params"] + estimates["linear_params"]
    )

    return estimates


def predict_memory_usage(
    num_conv_blocks: int,
    channels: List[int],
    num_classes: int,
    batch_size: int = 32,
    input_height: int = 224,
    input_width: int = 224,
) -> Dict[str, float]:
    """Predict memory usage without running the model.

    Args:
        num_conv_blocks: Number of convolutional blocks.
        channels: Channel progression.
        num_classes: Number of output classes.
        batch_size: Batch size for training.
        input_height: Input height.
        input_width: Input width.

    Returns:
        Dictionary with estimated memory in MB.
    """
    # Bytes per float32 value
    bytes_per_value = 4

    # Estimate parameter memory
    param_estimate = estimate_model_size(num_conv_blocks, channels, num_classes)
    param_memory_mb = param_estimate["total_params"] * bytes_per_value / (1024 * 1024)

    # Estimate activation memory
    # Simplified: sum of all layer outputs during forward pass
    activation_elements = batch_size * 3 * input_height * input_width  # Input

    current_h, current_w = input_height, input_width
    for ch in channels:
        # After conv
        activation_elements += batch_size * ch * current_h * current_w
        # After pooling (2x2)
        current_h = current_h // 2
        current_w = current_w // 2

    activation_memory_mb = activation_elements * bytes_per_value / (1024 * 1024)

    # Estimate gradient memory (same as activation during training)
    gradient_memory_mb = activation_memory_mb

    total_memory_mb = param_memory_mb + activation_memory_mb + gradient_memory_mb

    return {
        "parameter_memory_mb": round(param_memory_mb, 2),
        "activation_memory_mb": round(activation_memory_mb, 2),
        "gradient_memory_mb": round(gradient_memory_mb, 2),
        "total_training_memory_mb": round(total_memory_mb, 2),
        "inference_memory_mb": round(param_memory_mb + activation_memory_mb, 2),
    }
