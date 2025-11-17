"""Model summary and introspection utilities.

Provides comprehensive tools for analyzing PyTorch models including:
- Layer-by-layer output shape tracking
- Parameter distribution analysis
- Memory usage estimation
- Model structure visualization

Example:
    >>> from torchvision_customizer.utils import print_model_summary
    >>> from torchvision_customizer import CustomCNN
    >>> model = CustomCNN(input_shape=(3, 224, 224), num_classes=1000)
    >>> print_model_summary(model, input_shape=(1, 3, 224, 224))
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def calculate_output_shape(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Calculate output shape by forward pass with dummy input.

    Args:
        model: PyTorch model to analyze.
        input_shape: Input tensor shape (including batch dimension).
        device: Device to run the model on ('cpu' or 'cuda').

    Returns:
        Dictionary containing:
            - 'output_shape': Shape of the output tensor
            - 'output_size': Total elements in output
            - 'success': Whether calculation succeeded

    Raises:
        ValueError: If input shape is invalid.
    """
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
        raise ValueError(f"input_shape must be at least 3D (B, C, H, W), got {input_shape}")

    try:
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            dummy_input = torch.randn(*input_shape, device=device)
            output = model(dummy_input)

        return {
            "output_shape": tuple(output.shape),
            "output_size": output.numel(),
            "success": True,
        }
    except Exception as e:
        return {
            "output_shape": None,
            "output_size": None,
            "success": False,
            "error": str(e),
        }


def get_layer_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Get layer-by-layer summary with output shapes.

    Hooks into model layers to capture output shapes during forward pass.

    Args:
        model: PyTorch model to analyze.
        input_shape: Input tensor shape (including batch dimension).
        device: Device to run the model on.

    Returns:
        List of dictionaries containing layer information:
            - 'name': Layer name/path
            - 'type': Layer type (class name)
            - 'output_shape': Output shape of the layer
            - 'parameters': Number of parameters
            - 'trainable': Whether layer is trainable
    """
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
        raise ValueError(f"input_shape must be at least 3D (B, C, H, W), got {input_shape}")

    layers_info = []
    hooks = []

    def get_hook(name: str) -> callable:
        """Create a hook function for a layer."""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
                "output_shape": tuple(output.shape) if isinstance(output, torch.Tensor) else None,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable": any(p.requires_grad for p in module.parameters()),
            }
            layers_info.append(layer_info)

        return hook

    # Register hooks
    for name, module in model.named_modules():
        if name:  # Skip the root module
            hook = module.register_forward_hook(get_hook(name))
            hooks.append(hook)

    # Forward pass
    try:
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            dummy_input = torch.randn(*input_shape, device=device)
            _ = model(dummy_input)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return layers_info


def count_parameters_by_type(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """Count parameters grouped by layer type.

    Args:
        model: PyTorch model to analyze.

    Returns:
        Dictionary mapping layer types to:
            - 'total': Total parameters of this type
            - 'trainable': Trainable parameters
            - 'count': Number of layers of this type
    """
    param_count = {}

    for module in model.modules():
        module_type = module.__class__.__name__

        if module_type not in param_count:
            param_count[module_type] = {
                "total": 0,
                "trainable": 0,
                "count": 0,
            }

        module_params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

        param_count[module_type]["total"] += module_params
        param_count[module_type]["trainable"] += trainable_params
        param_count[module_type]["count"] += 1

    return param_count


def get_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> Dict[str, Union[float, str]]:
    """Estimate model memory usage.

    Args:
        model: PyTorch model to analyze.
        input_shape: Input tensor shape (including batch dimension).
        device: Device to run the model on.

    Returns:
        Dictionary containing:
            - 'parameter_memory_mb': Memory used by parameters
            - 'activation_memory_mb': Estimated activation memory
            - 'total_memory_mb': Total estimated memory
            - 'device': Device used for estimation
    """
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
        raise ValueError(f"input_shape must be at least 3D, got {input_shape}")

    # Calculate parameter memory
    param_size = sum(p.nelement() for p in model.parameters()) * 4 / (1024**2)  # 4 bytes per float32

    # Estimate activation memory through forward pass
    model = model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            dummy_input = torch.randn(*input_shape, device=device)
            output = model(dummy_input)

        activation_size = (dummy_input.nelement() + output.nelement()) * 4 / (1024**2)
        total_size = param_size + activation_size

        return {
            "parameter_memory_mb": round(param_size, 4),
            "activation_memory_mb": round(activation_size, 4),
            "total_memory_mb": round(total_size, 4),
            "device": device,
        }
    except Exception as e:
        return {
            "parameter_memory_mb": round(param_size, 4),
            "activation_memory_mb": None,
            "total_memory_mb": None,
            "device": device,
            "error": str(e),
        }


def get_model_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
) -> Dict[str, Union[int, str]]:
    """Estimate model FLOPs (floating point operations).

    Note: This is an estimation based on standard layer operations.
    Actual FLOPs may vary based on implementation details.

    Args:
        model: PyTorch model to analyze.
        input_shape: Input tensor shape (including batch dimension).

    Returns:
        Dictionary containing:
            - 'total_flops': Total estimated FLOPs
            - 'total_flops_in_billions': Total FLOPs in billions
            - 'success': Whether estimation succeeded
    """
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
        raise ValueError(f"input_shape must be at least 3D, got {input_shape}")

    total_flops = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs: output_height * output_width * kernel_h * kernel_w * in_channels * out_channels
            batch_size = input_shape[0]
            out_channels = module.out_channels
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_size = (input_shape[2] // module.stride[0]) * (input_shape[3] // module.stride[1])
            module_flops = batch_size * output_size * output_size * kernel_ops * out_channels / module.groups
            total_flops += int(module_flops)

        elif isinstance(module, nn.Linear):
            # Linear FLOPs: batch_size * input_features * output_features
            batch_size = input_shape[0]
            module_flops = batch_size * module.in_features * module.out_features
            total_flops += int(module_flops)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # BatchNorm FLOPs: roughly proportional to number of elements
            batch_size = input_shape[0]
            num_features = module.num_features
            spatial_size = 1
            for s in input_shape[2:]:
                spatial_size *= s
            module_flops = batch_size * num_features * spatial_size * 5  # Approximate
            total_flops += int(module_flops)

    return {
        "total_flops": total_flops,
        "total_flops_in_billions": round(total_flops / 1e9, 4),
        "success": True,
    }


def print_model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Print comprehensive model summary.

    Args:
        model: PyTorch model to summarize.
        input_shape: Input tensor shape (including batch dimension).
        verbose: Whether to print detailed information.

    Returns:
        Dictionary containing all summary information.
    """
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
        raise ValueError(f"input_shape must be at least 3D, got {input_shape}")

    summary = {}

    # Basic model info
    summary["model_name"] = model.__class__.__name__
    summary["input_shape"] = input_shape

    # Output shape
    output_info = calculate_output_shape(model, input_shape)
    summary["output_shape"] = output_info.get("output_shape")
    summary["output_size"] = output_info.get("output_size")

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary["total_parameters"] = total_params
    summary["trainable_parameters"] = trainable_params
    summary["frozen_parameters"] = total_params - trainable_params

    # Parameters by type
    summary["parameters_by_type"] = count_parameters_by_type(model)

    # Memory usage
    summary["memory"] = get_memory_usage(model, input_shape)

    # FLOPs estimation
    summary["flops"] = get_model_flops(model, input_shape)

    # Print if verbose
    if verbose:
        print("\n" + "=" * 70)
        print(f"Model Summary: {summary['model_name']}")
        print("=" * 70)
        print(f"Input Shape:  {summary['input_shape']}")
        print(f"Output Shape: {summary['output_shape']}")
        print(f"Output Size:  {summary['output_size']:,} elements")
        print()
        print("Parameters:")
        print(f"  Total:      {summary['total_parameters']:,}")
        print(f"  Trainable:  {summary['trainable_parameters']:,}")
        print(f"  Frozen:     {summary['frozen_parameters']:,}")
        print()
        print("Memory Usage:")
        print(f"  Parameters: {summary['memory'].get('parameter_memory_mb')} MB")
        print(f"  Activations: {summary['memory'].get('activation_memory_mb')} MB")
        print(f"  Total:      {summary['memory'].get('total_memory_mb')} MB")
        print()
        print("FLOPs Estimation:")
        print(f"  Total:      {summary['flops'].get('total_flops'):,}")
        print(f"  Billions:   {summary['flops'].get('total_flops_in_billions')} B")
        print("=" * 70 + "\n")

    return summary
