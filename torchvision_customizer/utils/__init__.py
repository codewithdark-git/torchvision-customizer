"""Utils module: Utilities (summary, validation, FLOPs calculation).

Provides model analysis tools including:
- Model summaries with parameter and memory analysis
- Architecture validation and dimension tracking
- Memory and FLOPs estimation
"""

from torchvision_customizer.utils.model_summary import (
    calculate_output_shape,
    count_parameters_by_type,
    get_layer_summary,
    get_memory_usage,
    get_model_flops,
    print_model_summary,
)
from torchvision_customizer.utils.validators import (
    estimate_model_size,
    predict_memory_usage,
    validate_architecture,
    validate_channel_progression,
    validate_spatial_dimensions,
)
from torchvision_customizer.utils.architecture_search import (
    ArchitectureConfig,
    GridSearch,
    RandomSearch,
    ArchitectureFactory,
    ArchitectureScorer,
    ArchitectureComparator,
    ArchitecturePattern,
    expand_config_list,
    sample_architecture,
)

__all__ = [
    # Model summary functions
    "calculate_output_shape",
    "get_layer_summary",
    "count_parameters_by_type",
    "get_memory_usage",
    "get_model_flops",
    "print_model_summary",
    # Validator functions
    "validate_spatial_dimensions",
    "validate_channel_progression",
    "validate_architecture",
    "estimate_model_size",
    "predict_memory_usage",
    # Step 6: Architecture Search
    "ArchitectureConfig",
    "GridSearch",
    "RandomSearch",
    "ArchitectureFactory",
    "ArchitectureScorer",
    "ArchitectureComparator",
    "ArchitecturePattern",
    "expand_config_list",
    "sample_architecture",
]

