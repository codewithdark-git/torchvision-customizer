Hybrid Module
=============

.. module:: torchvision_customizer.hybrid

The Hybrid module enables customization of pre-trained torchvision models.
This is the star feature of v2.1.

Overview
--------

The Hybrid module provides:

* **HybridBuilder**: Load and customize pre-trained models
* **Weight Utilities**: Smart weight loading with mismatch tolerance
* **Backbone Extraction**: Analyze and decompose model structures

HybridBuilder
-------------

.. autoclass:: torchvision_customizer.hybrid.HybridBuilder
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from torchvision_customizer import HybridBuilder

   builder = HybridBuilder()
   
   # Load ResNet50 with ImageNet weights
   model = builder.from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=100,
   )

Adding Patches
^^^^^^^^^^^^^^

Patches allow you to modify specific layers:

.. code-block:: python

   model = builder.from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       patches={
           "layer3": {"wrap": "se"},           # Wrap with SE attention
           "layer4": {"wrap": "cbam_block"},   # Wrap with CBAM
       },
       num_classes=100,
   )

Patch Operations
^^^^^^^^^^^^^^^^

+-------------+----------------------------------+
| Operation   | Description                      |
+=============+==================================+
| ``wrap``    | Wrap layer with attention/block  |
+-------------+----------------------------------+
| ``inject``  | Inject block after layer         |
+-------------+----------------------------------+
| ``replace`` | Replace layer entirely           |
+-------------+----------------------------------+

Freezing for Fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = builder.from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=10,
       freeze_backbone=True,
       unfreeze_stages=[3],  # Only train last stage + head
   )

   # Later, unfreeze everything
   model.unfreeze_all()

Supported Backbones
-------------------

.. code-block:: python

   from torchvision_customizer import HybridBuilder
   
   # List all supported backbones
   print(HybridBuilder.list_backbones())

**ResNet Family**

* resnet18, resnet34, resnet50, resnet101, resnet152
* wide_resnet50_2, wide_resnet101_2
* resnext50_32x4d, resnext101_32x8d, resnext101_64x4d

**EfficientNet Family**

* efficientnet_b0 through efficientnet_b7
* efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

**ConvNeXt Family**

* convnext_tiny, convnext_small, convnext_base, convnext_large

**MobileNet Family**

* mobilenet_v2
* mobilenet_v3_small, mobilenet_v3_large

**Other**

* VGG (11, 13, 16, 19 with/without BN)
* DenseNet (121, 169, 201, 161)
* Vision Transformer (vit_b_16, vit_b_32, vit_l_16, vit_l_32)
* Swin Transformer (swin_t, swin_s, swin_b)

Weight Utilities
----------------

partial_load
^^^^^^^^^^^^

.. autofunction:: torchvision_customizer.hybrid.partial_load

Load weights with tolerance for shape mismatches:

.. code-block:: python

   from torchvision_customizer import partial_load

   report = partial_load(
       model,
       state_dict,
       ignore_mismatch=True,
       init_new_layers="kaiming",
   )
   
   print(report.summary())
   # Loaded: 95% of parameters
   # Shape Mismatch: 12 parameters
   # Newly Initialized: 5 parameters

transfer_weights
^^^^^^^^^^^^^^^^

.. autofunction:: torchvision_customizer.hybrid.transfer_weights

Transfer weights between models with filtering:

.. code-block:: python

   from torchvision_customizer import transfer_weights

   transfer_weights(
       source=pretrained_model,
       target=custom_model,
       exclude_patterns=['fc', 'classifier'],
   )

Backbone Extraction
-------------------

extract_tiers
^^^^^^^^^^^^^

.. autofunction:: torchvision_customizer.hybrid.extract_tiers

Decompose a model into stem, stages, and head:

.. code-block:: python

   from torchvision.models import resnet50
   from torchvision_customizer import extract_tiers

   model = resnet50()
   tiers = extract_tiers(model, "resnet50")
   
   print(f"Stem: {type(tiers['stem'])}")
   print(f"Stages: {len(tiers['stages'])}")
   print(f"Head: {type(tiers['head'])}")

get_backbone_info
^^^^^^^^^^^^^^^^^

.. autofunction:: torchvision_customizer.hybrid.get_backbone_info

Get structural information about a backbone:

.. code-block:: python

   from torchvision_customizer import get_backbone_info
   from torchvision.models import resnet50

   model = resnet50()
   info = get_backbone_info(model, "resnet50")
   
   print(info.summary())
   # Backbone: resnet50
   # Stem: conv1, bn1, relu, maxpool
   # Stages: layer1, layer2, layer3, layer4
   # Parameters: 25,557,032

HybridModel
-----------

.. autoclass:: torchvision_customizer.hybrid.builder.HybridModel
   :members:
   :undoc-members:
   :show-inheritance:

Methods
^^^^^^^

* ``forward(x)``: Standard forward pass
* ``forward_features(x)``: Extract features before head
* ``get_stage_outputs(x)``: Get intermediate stage outputs (for FPN)
* ``freeze_backbone(unfreeze_stages)``: Freeze backbone layers
* ``unfreeze_all()``: Unfreeze all parameters
* ``count_parameters(trainable_only)``: Count parameters
* ``explain()``: Human-readable model description

Example
^^^^^^^

.. code-block:: python

   model = HybridBuilder().from_torchvision("resnet50", ...)
   
   # Get features for FPN
   features = model.get_stage_outputs(x)
   
   # Freeze for fine-tuning
   model.freeze_backbone(unfreeze_stages=[3])
   
   # Print summary
   print(model.explain())

