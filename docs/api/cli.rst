Command-Line Interface
======================

.. module:: torchvision_customizer.cli

The CLI (``tvc``) provides command-line tools for model building, benchmarking, and export.

Installation
------------

The CLI is installed automatically with the package:

.. code-block:: bash

   pip install torchvision-customizer
   
   # Verify installation
   tvc --version

Commands
--------

build
^^^^^

Build a model from a YAML recipe.

.. code-block:: bash

   tvc build --yaml model.yaml [options]

Options:

* ``--yaml, -y``: Recipe YAML file (required)
* ``--output, -o``: Output path for saved model
* ``--num-classes``: Override number of classes
* ``--format``: Output format (pt, onnx, torchscript)

Examples:

.. code-block:: bash

   # Build and save model
   tvc build --yaml model.yaml --output model.pt
   
   # Build with custom classes
   tvc build --yaml model.yaml --num-classes 10 --output model.pt
   
   # Build and export to TorchScript
   tvc build --yaml model.yaml --format torchscript --output model.ts

benchmark
^^^^^^^^^

Benchmark model performance.

.. code-block:: bash

   tvc benchmark --yaml model.yaml [options]

Options:

* ``--yaml, -y``: Recipe YAML file (required)
* ``--batch-size``: Batch size (default: 16)
* ``--input-size``: Input image size (default: 224)
* ``--device``: Device (cpu, cuda, mps)
* ``--warmup``: Warmup iterations (default: 5)
* ``--iterations``: Benchmark iterations (default: 50)

Example:

.. code-block:: bash

   tvc benchmark --yaml model.yaml --device cuda --batch-size 32
   
   # Output:
   # Results:
   #   Mean latency:    12.34 ms
   #   Throughput:      2596.2 images/sec
   #   Peak GPU memory: 1024.5 MB

validate
^^^^^^^^

Validate a recipe file.

.. code-block:: bash

   tvc validate --yaml model.yaml [--strict]

Options:

* ``--yaml, -y``: Recipe YAML file (required)
* ``--strict``: Strict validation mode

Example:

.. code-block:: bash

   tvc validate --yaml model.yaml
   # Recipe is valid!
   
   tvc validate --yaml broken.yaml
   # Validation Error: stages must be a list

export
^^^^^^

Export model to different formats.

.. code-block:: bash

   tvc export --yaml model.yaml --output model.onnx [options]

Options:

* ``--yaml, -y``: Recipe YAML file (required)
* ``--output, -o``: Output file path (required)
* ``--format``: Export format (onnx, torchscript)
* ``--input-size``: Input image size (default: 224)
* ``--opset``: ONNX opset version (default: 11)

Examples:

.. code-block:: bash

   # Export to ONNX
   tvc export --yaml model.yaml --format onnx --output model.onnx
   
   # Export with custom input size
   tvc export --yaml model.yaml --format onnx --input-size 384 --output model.onnx
   
   # Export to TorchScript
   tvc export --yaml model.yaml --format torchscript --output model.pt

list-backbones
^^^^^^^^^^^^^^

List all supported torchvision backbones.

.. code-block:: bash

   tvc list-backbones [--filter PATTERN]

Options:

* ``--filter``: Filter by name pattern

Example:

.. code-block:: bash

   tvc list-backbones
   # Resnet:
   #   - resnet18
   #   - resnet34
   #   - resnet50
   #   ...
   
   tvc list-backbones --filter efficient
   # Efficientnet:
   #   - efficientnet_b0
   #   - efficientnet_b1
   #   ...

list-blocks
^^^^^^^^^^^

List all available building blocks.

.. code-block:: bash

   tvc list-blocks [--category CATEGORY]

Options:

* ``--category``: Filter by category (block, attention, regularization, etc.)

Example:

.. code-block:: bash

   tvc list-blocks
   # Block:
   #   - conv
   #   - residual
   #   - mbconv
   # Attention:
   #   - se
   #   - cbam_block
   #   - eca
   
   tvc list-blocks --category attention
   # - cbam_block
   # - channel_attention
   # - eca
   # - se

create-recipe
^^^^^^^^^^^^^

Create a recipe from a template.

.. code-block:: bash

   tvc create-recipe --template NAME --output FILE [options]

Options:

* ``--template, -t``: Template name (required)
* ``--output, -o``: Output YAML file (required)
* ``--num-classes``: Number of classes

Available templates:

* ``resnet_base``: Basic ResNet architecture
* ``efficientnet_base``: EfficientNet-style architecture
* ``hybrid_resnet_se``: ResNet with SE attention (hybrid)

Example:

.. code-block:: bash

   tvc create-recipe --template hybrid_resnet_se --output my_model.yaml
   # Created recipe: my_model.yaml
   # Based on template: hybrid_resnet_se

info
^^^^

Show information about a model/recipe.

.. code-block:: bash

   tvc info --yaml model.yaml [--verbose]

Options:

* ``--yaml, -y``: Recipe YAML file (required)
* ``--verbose, -v``: Show detailed model information

Example:

.. code-block:: bash

   tvc info --yaml model.yaml --verbose
   # Recipe: model.yaml
   # Name: ResNet50-SE-Custom
   # Backbone: resnet50
   #   Weights: IMAGENET1K_V2
   #   Patches: 2 modifications
   #
   # Model Statistics:
   #   Total parameters: 25,557,032
   #   Trainable parameters: 2,048,100

YAML Recipe Format
------------------

Basic Structure
^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Metadata
   name: My Model
   version: "1.0.0"
   
   # Option 1: Hybrid backbone
   backbone:
     name: resnet50
     weights: IMAGENET1K_V2
     patches:
       layer3: {wrap: se}
   
   # Option 2: From scratch
   stem: {type: conv, channels: 64}
   stages:
     - {pattern: residual, channels: 64, blocks: 2}
     - {pattern: residual, channels: 128, blocks: 2, downsample: true}
   
   # Head configuration
   head:
     num_classes: 100
     dropout: 0.3

Macros
^^^^^^

.. code-block:: yaml

   macros:
     attention: se
     dropout: 0.3
   
   backbone:
     name: resnet50
     patches:
       layer3: {wrap: "@attention"}  # Expands to "se"
   
   head:
     dropout: "@dropout"  # Expands to 0.3

Inheritance
^^^^^^^^^^^

.. code-block:: yaml

   # Extend a built-in template
   extends: resnet_base
   
   # Override specific parts
   head:
     num_classes: 10

