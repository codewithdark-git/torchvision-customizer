"""Recipe string parser.

Parses string-based component definitions into structured configurations.
Examples:
    "residual(64) x 2" -> {'pattern': 'residual', 'channels': 64, 'blocks': 2}
    "conv(64, k=7, s=2)" -> {'pattern': 'conv', 'channels': 64, 'kernel_size': 7, 'stride': 2}
    
Shortcuts:
    k -> kernel_size
    s -> stride
    p -> padding
    g -> groups
    e -> expansion
"""

# Parameter shortcuts mapping
PARAM_SHORTCUTS = {
    'k': 'kernel_size',
    's': 'stride',
    'p': 'padding',
    'g': 'groups',
    'e': 'expansion',
    'r': 'reduction',
    'd': 'dropout',
}

import re
import ast
from typing import Any, Dict, List, Optional, Tuple, Union


def parse_definition(def_str: str) -> Dict[str, Any]:
    """Parse a single component definition string.
    
    Format: "name(args) modifiers"
    Example: "residual(64, stride=2) x 3 | downsample"
    
    Returns:
        Dictionary with parsed parameters
    """
    def_str = def_str.strip()
    
    # Defaults
    config = {
        'repeat': 1,
        'modifiers': [],
        'args': [],
        'kwargs': {},
    }
    
    # 1. Parse modifiers (x N, | modifier)
    # Split by pipe first for end modifiers
    parts = def_str.split('|')
    main_part = parts[0].strip()
    
    if len(parts) > 1:
        for mod in parts[1:]:
            config['modifiers'].append(mod.strip())
            
    # Check for repetition (x N)
    if ' x ' in main_part:
        main_part, repeat_part = main_part.split(' x ')
        try:
            config['repeat'] = int(repeat_part.strip())
        except ValueError:
            raise ValueError(f"Invalid repetition count in '{def_str}'")
    
    # 2. Parse name and arguments "name(args)"
    match = re.match(r'^([a-zA-Z0-9_]+)\s*(?:\((.*)\))?$', main_part)
    if not match:
        # Maybe just a name without parens?
        if re.match(r'^[a-zA-Z0-9_]+$', main_part):
            config['name'] = main_part
            return config
        raise ValueError(f"Invalid definition format: '{main_part}'")
    
    name, args_str = match.groups()
    config['name'] = name
    
    if args_str:
        args_str = args_str.strip()
        # Use AST to parse arguments safely
        # We wrap it in a function call to parse args and kwargs
        try:
            tree = ast.parse(f"func({args_str})")
            call = tree.body[0].value
            
            # Positional args
            for arg in call.args:
                if isinstance(arg, ast.Constant):
                    config['args'].append(arg.value)
                elif isinstance(arg, ast.Num): # Python < 3.8
                    config['args'].append(arg.n)
                elif isinstance(arg, ast.Str): # Python < 3.8
                    config['args'].append(arg.s)
                elif isinstance(arg, ast.Name): # e.g. True/False/None
                    if arg.id == 'True': config['args'].append(True)
                    elif arg.id == 'False': config['args'].append(False)
                    elif arg.id == 'None': config['args'].append(None)
                    else: config['args'].append(arg.id) # Treat as string
            
            # Keyword args
            for kw in call.keywords:
                val = kw.value
                # Expand shortcut parameter names (k -> kernel_size, s -> stride, etc.)
                param_name = PARAM_SHORTCUTS.get(kw.arg, kw.arg)
                
                if isinstance(val, ast.Constant):
                    config['kwargs'][param_name] = val.value
                elif isinstance(val, ast.Num):
                    config['kwargs'][param_name] = val.n
                elif isinstance(val, ast.Str):
                    config['kwargs'][param_name] = val.s
                elif isinstance(val, ast.Name):
                    if val.id == 'True': config['kwargs'][param_name] = True
                    elif val.id == 'False': config['kwargs'][param_name] = False
                    elif val.id == 'None': config['kwargs'][param_name] = None
                    else: config['kwargs'][param_name] = val.id
        except Exception as e:
            raise ValueError(f"Failed to parse arguments in '{def_str}': {e}")
            
    return config


def parse_recipe(recipe: 'Recipe') -> Dict[str, Any]:
    """Parse entire recipe into build configuration."""
    
    # Parse Stem
    stem_config = _normalize_config(recipe.stem, "stem")
    
    # Parse Stages
    stages_config = []
    for i, stage in enumerate(recipe.stages):
        stages_config.append(_normalize_config(stage, "stage"))
        
    # Parse Head
    head_config = _normalize_config(recipe.head, "head")
    
    return {
        'stem': stem_config,
        'stages': stages_config,
        'head': head_config,
        'input_shape': recipe.input_shape,
    }


def _normalize_config(item: Union[str, Dict[str, Any]], context: str) -> Dict[str, Any]:
    """Normalize string or dict configuration."""
    if isinstance(item, str):
        parsed = parse_definition(item)
        
        # Transform parsed structure to component config
        config = {'type': parsed['name']}
        
        # Map positional args based on context
        args = parsed['args']
        kwargs = parsed['kwargs']
        
        # Context-specific argument mapping
        if context == 'stem':
            # name(channels, kernel, stride)
            if len(args) > 0: config['channels'] = args[0]
            if len(args) > 1: config['kernel'] = args[1]
            if len(args) > 2: config['stride'] = args[2]
            
        elif context == 'stage':
            # name(channels) x Blocks | modifiers
            if len(args) > 0: config['channels'] = args[0]
            config['blocks'] = parsed['repeat']
            if 'downsample' in parsed['modifiers']:
                config['downsample'] = True
            
            # Store original name as pattern if it's a known pattern
            if parsed['name'] in ['residual', 'bottleneck', 'conv', 'dense', 'depthwise', 
                                  'mbconv', 'fused_mbconv', 'wide_bottleneck']:
                config['pattern'] = parsed['name']
            
        elif context == 'head':
            # linear(classes)
            if len(args) > 0: config['num_classes'] = args[0]
        
        # Merge kwargs
        config.update(kwargs)
        return config
        
    elif isinstance(item, dict):
        return item.copy()
    else:
        raise ValueError(f"Invalid configuration item type: {type(item)}")
