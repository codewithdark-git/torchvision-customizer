"""Command-Line Interface for torchvision-customizer.

Provides CLI tools for:
- Model prototyping and benchmarking
- Recipe validation and building
- Model export (ONNX, TorchScript)
- Architecture exploration

Usage:
    tvc build --yaml recipe.yaml --output model.pt
    tvc benchmark --yaml recipe.yaml --input-size 224
    tvc validate --yaml recipe.yaml
    tvc export --yaml recipe.yaml --format onnx --output model.onnx
    tvc list-backbones
    tvc list-blocks
    tvc create-recipe --template resnet_base --output my_recipe.yaml

Example:
    $ tvc build --yaml my_model.yaml --num-classes 100
    $ tvc benchmark --yaml my_model.yaml --batch-size 32 --device cuda
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='tvc',
        description='torchvision-customizer CLI - Build and prototype custom CNNs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tvc build --yaml my_recipe.yaml
  tvc benchmark --yaml my_recipe.yaml --device cuda
  tvc list-blocks
  tvc create-recipe --template hybrid_resnet_se --output my_model.yaml
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='%(prog)s 2.1.0'
    )
    
    subparsers = parser.add_subparsers(title='commands', dest='command')
    
    # Build command
    build_parser = subparsers.add_parser(
        'build',
        help='Build a model from recipe'
    )
    build_parser.add_argument('--yaml', '-y', required=True, help='Recipe YAML file')
    build_parser.add_argument('--output', '-o', help='Output path for saved model')
    build_parser.add_argument('--num-classes', type=int, help='Override number of classes')
    build_parser.add_argument('--format', choices=['pt', 'onnx', 'torchscript'], default='pt')
    build_parser.set_func = lambda: cmd_build
    build_parser.set_defaults(func=cmd_build)
    
    # Benchmark command
    bench_parser = subparsers.add_parser(
        'benchmark',
        help='Benchmark a model'
    )
    bench_parser.add_argument('--yaml', '-y', required=True, help='Recipe YAML file')
    bench_parser.add_argument('--batch-size', type=int, default=16)
    bench_parser.add_argument('--input-size', type=int, default=224)
    bench_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    bench_parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations')
    bench_parser.add_argument('--iterations', type=int, default=50, help='Benchmark iterations')
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate a recipe file'
    )
    validate_parser.add_argument('--yaml', '-y', required=True, help='Recipe YAML file')
    validate_parser.add_argument('--strict', action='store_true', help='Strict validation')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Export command
    export_parser = subparsers.add_parser(
        'export',
        help='Export model to different formats'
    )
    export_parser.add_argument('--yaml', '-y', required=True, help='Recipe YAML file')
    export_parser.add_argument('--output', '-o', required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['onnx', 'torchscript'], default='onnx')
    export_parser.add_argument('--input-size', type=int, default=224)
    export_parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    export_parser.set_defaults(func=cmd_export)
    
    # List backbones command
    list_backbones_parser = subparsers.add_parser(
        'list-backbones',
        help='List supported torchvision backbones'
    )
    list_backbones_parser.add_argument('--filter', help='Filter by name pattern')
    list_backbones_parser.set_defaults(func=cmd_list_backbones)
    
    # List blocks command
    list_blocks_parser = subparsers.add_parser(
        'list-blocks',
        help='List available building blocks'
    )
    list_blocks_parser.add_argument('--category', help='Filter by category')
    list_blocks_parser.set_defaults(func=cmd_list_blocks)
    
    # Create recipe command
    create_parser = subparsers.add_parser(
        'create-recipe',
        help='Create a recipe from template'
    )
    create_parser.add_argument('--template', '-t', required=True, help='Template name')
    create_parser.add_argument('--output', '-o', required=True, help='Output YAML file')
    create_parser.add_argument('--num-classes', type=int, help='Number of classes')
    create_parser.set_defaults(func=cmd_create_recipe)
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about a model/recipe'
    )
    info_parser.add_argument('--yaml', '-y', required=True, help='Recipe YAML file')
    info_parser.add_argument('--verbose', '-v', action='store_true')
    info_parser.set_defaults(func=cmd_info)
    
    return parser


def cmd_build(args):
    """Build command implementation."""
    from torchvision_customizer.recipe import load_yaml_recipe
    
    print(f"Building model from: {args.yaml}")
    
    kwargs = {}
    if args.num_classes:
        kwargs['num_classes'] = args.num_classes
    
    try:
        model = load_yaml_recipe(args.yaml, **kwargs)
        print(f"Model built successfully!")
        
        # Print summary
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        
        if args.output:
            output_path = Path(args.output)
            
            if args.format == 'pt':
                torch.save(model.state_dict(), output_path)
                print(f"  Saved to: {output_path}")
            elif args.format == 'torchscript':
                scripted = torch.jit.script(model)
                scripted.save(str(output_path))
                print(f"  Saved TorchScript to: {output_path}")
            elif args.format == 'onnx':
                _export_onnx(model, output_path, input_size=224)
                print(f"  Saved ONNX to: {output_path}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_benchmark(args):
    """Benchmark command implementation."""
    from torchvision_customizer.recipe import load_yaml_recipe
    
    print(f"Benchmarking model from: {args.yaml}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Input size: {args.input_size}x{args.input_size}")
    print()
    
    try:
        model = load_yaml_recipe(args.yaml, verbose=False)
        
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        x = torch.randn(args.batch_size, 3, args.input_size, args.input_size).to(device)
        
        # Warmup
        print(f"Warming up ({args.warmup} iterations)...")
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(x)
        
        # Synchronize if CUDA
        if args.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking ({args.iterations} iterations)...")
        times = []
        
        with torch.no_grad():
            for _ in range(args.iterations):
                start = time.perf_counter()
                _ = model(x)
                
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)
        
        # Calculate stats
        mean_time = sum(times) / len(times) * 1000  # ms
        min_time = min(times) * 1000
        max_time = max(times) * 1000
        throughput = args.batch_size / (mean_time / 1000)  # images/sec
        
        print()
        print("Results:")
        print(f"  Mean latency:    {mean_time:.2f} ms")
        print(f"  Min latency:     {min_time:.2f} ms")
        print(f"  Max latency:     {max_time:.2f} ms")
        print(f"  Throughput:      {throughput:.1f} images/sec")
        
        # Memory info
        if args.device == 'cuda':
            memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Peak GPU memory: {memory:.1f} MB")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_validate(args):
    """Validate command implementation."""
    from torchvision_customizer.recipe import load_yaml_config, validate_recipe_config
    
    print(f"Validating: {args.yaml}")
    
    try:
        config = load_yaml_config(args.yaml, validate=False)
        warnings = validate_recipe_config(config, strict=args.strict)
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("Recipe is valid!")
            
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_export(args):
    """Export command implementation."""
    from torchvision_customizer.recipe import load_yaml_recipe
    
    print(f"Exporting model from: {args.yaml}")
    print(f"  Format: {args.format}")
    print(f"  Output: {args.output}")
    
    try:
        model = load_yaml_recipe(args.yaml, verbose=False)
        model.eval()
        
        if args.format == 'onnx':
            _export_onnx(model, args.output, args.input_size, args.opset)
        elif args.format == 'torchscript':
            scripted = torch.jit.script(model)
            scripted.save(args.output)
        
        print("Export successful!")
        
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)


def _export_onnx(model, output_path, input_size=224, opset=11):
    """Export model to ONNX."""
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )


def cmd_list_backbones(args):
    """List backbones command implementation."""
    from torchvision_customizer.hybrid import HybridBuilder
    
    backbones = HybridBuilder.list_backbones()
    
    if args.filter:
        backbones = [b for b in backbones if args.filter.lower() in b.lower()]
    
    print("Supported Backbones:")
    print("-" * 40)
    
    # Group by family
    families = {}
    for b in backbones:
        family = b.split('_')[0].replace('net', 'net_')
        if family not in families:
            families[family] = []
        families[family].append(b)
    
    for family, members in sorted(families.items()):
        print(f"\n{family.title()}:")
        for m in members:
            print(f"  - {m}")


def cmd_list_blocks(args):
    """List blocks command implementation."""
    from torchvision_customizer import registry
    
    if args.category:
        blocks = registry.list_components(args.category)
        print(f"Blocks in category '{args.category}':")
    else:
        print("Available Building Blocks:")
        print("-" * 40)
        
        for category in registry.categories():
            blocks = registry.list_components(category)
            print(f"\n{category.title()}:")
            for block in blocks:
                info = registry.info(block)
                aliases = info.get('aliases', [])
                alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
                print(f"  - {block}{alias_str}")


def cmd_create_recipe(args):
    """Create recipe command implementation."""
    from torchvision_customizer.recipe import create_recipe_from_template, list_templates
    
    templates = list_templates()
    
    if args.template not in templates:
        print(f"Unknown template: {args.template}")
        print(f"Available templates: {', '.join(templates)}")
        sys.exit(1)
    
    overrides = {}
    if args.num_classes:
        overrides['head'] = {'num_classes': args.num_classes}
    
    config = create_recipe_from_template(
        args.template,
        output_path=args.output,
        **overrides
    )
    
    print(f"Created recipe: {args.output}")
    print(f"  Based on template: {args.template}")


def cmd_info(args):
    """Info command implementation."""
    from torchvision_customizer.recipe import load_yaml_config, load_yaml_recipe
    
    print(f"Recipe: {args.yaml}")
    print("=" * 50)
    
    try:
        config = load_yaml_config(args.yaml, validate=False)
        
        # Basic info
        print(f"Name: {config.get('name', 'Unnamed')}")
        print(f"Version: {config.get('version', 'N/A')}")
        print(f"Description: {config.get('description', 'N/A')}")
        
        if 'input_shape' in config:
            print(f"Input shape: {config['input_shape']}")
        
        if 'backbone' in config:
            print(f"\nBackbone: {config['backbone'].get('name', 'N/A')}")
            print(f"  Weights: {config['backbone'].get('weights', 'None')}")
            if 'patches' in config['backbone']:
                print(f"  Patches: {len(config['backbone']['patches'])} modifications")
        
        if 'stages' in config:
            print(f"\nStages: {len(config['stages'])}")
        
        if args.verbose:
            print("\n--- Building model ---")
            model = load_yaml_recipe(args.yaml, verbose=False)
            
            params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\nModel Statistics:")
            print(f"  Total parameters: {params:,}")
            print(f"  Trainable parameters: {trainable:,}")
            
            # Memory estimate
            mem_mb = params * 4 / 1024**2  # 4 bytes per float32
            print(f"  Estimated size: {mem_mb:.1f} MB")
            
            if hasattr(model, 'explain'):
                print(f"\n{model.explain()}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

