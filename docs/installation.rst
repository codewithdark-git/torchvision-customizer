Installation
============

Requirements
------------

* Python 3.11 or higher
* PyTorch 2.5 or higher
* torchvision 0.20 or higher

Basic Installation
------------------

From PyPI (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install torchvision-customizer

From GitHub (Latest)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install git+https://github.com/codewithdark-git/torchvision-customizer.git

From Source (Development)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/codewithdark-git/torchvision-customizer.git
   cd torchvision-customizer
   pip install -e .

Development Installation
------------------------

For contributing or development:

.. code-block:: bash

   git clone https://github.com/codewithdark-git/torchvision-customizer.git
   cd torchvision-customizer
   pip install -e ".[dev]"

This installs additional development dependencies:

* pytest (testing)
* black, isort, flake8 (formatting/linting)
* mypy (type checking)
* sphinx (documentation)

Optional Dependencies
---------------------

For YAML recipes with strict validation:

.. code-block:: bash

   pip install jsonschema

For documentation building:

.. code-block:: bash

   pip install torchvision-customizer[docs]

Verifying Installation
----------------------

.. code-block:: python

   import torchvision_customizer
   print(torchvision_customizer.__version__)
   # 2.1.0

Test the CLI:

.. code-block:: bash

   tvc --version
   # tvc 2.1.0

Quick Test:

.. code-block:: python

   from torchvision_customizer import HybridBuilder, Stem, Stage, Head
   
   # Test Hybrid Builder
   model = HybridBuilder().from_torchvision(
       "resnet18", 
       weights=None,  # Skip download for testing
       num_classes=10
   )
   print(f"Hybrid model: {model.count_parameters():,} params")
   
   # Test Composer
   model = Stem(64) >> Stage(128, blocks=2) >> Head(10)
   print(f"Custom model: {sum(p.numel() for p in model.parameters()):,} params")

GPU Support
-----------

torchvision-customizer works with CUDA out of the box:

.. code-block:: python

   import torch
   from torchvision_customizer import HybridBuilder
   
   model = HybridBuilder().from_torchvision("resnet50", num_classes=10)
   model = model.to("cuda")
   
   x = torch.randn(1, 3, 224, 224).cuda()
   output = model(x)

For Apple Silicon (MPS):

.. code-block:: python

   model = model.to("mps")

Troubleshooting
---------------

**ImportError: No module named 'torchvision_customizer'**
   Make sure you've installed the package: ``pip install torchvision-customizer``

**ModuleNotFoundError: No module named 'torchvision'**
   Install PyTorch and torchvision: ``pip install torch torchvision``

**'tvc' is not recognized as a command**
   Ensure Python's Scripts directory is in your PATH, or use ``python -m torchvision_customizer.cli``

**CUDA out of memory**
   Use a smaller backbone, reduce batch size, or enable gradient checkpointing

**PyYAML not found**
   Install it: ``pip install pyyaml``

Updating
--------

To update to the latest version:

.. code-block:: bash

   pip install --upgrade torchvision-customizer

For development version:

.. code-block:: bash

   cd torchvision-customizer
   git pull
   pip install -e .

Uninstalling
------------

.. code-block:: bash

   pip uninstall torchvision-customizer
