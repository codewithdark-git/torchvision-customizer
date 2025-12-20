Hybrid Models (v2.1)
====================

This guide covers the new Hybrid Models feature introduced in v2.1, which allows you to customize pre-trained torchvision models.

What are Hybrid Models?
-----------------------

Hybrid Models combine:

1. **Pre-trained Backbones**: Load any torchvision model with ImageNet weights
2. **Custom Modifications**: Inject attention, replace blocks, modify architecture
3. **Smart Weight Loading**: Preserve as many pre-trained weights as possible

This gives you the best of both worlds: the power of transfer learning with the flexibility of custom architectures.

Getting Started
---------------

.. code-block:: python

   from torchvision_customizer import HybridBuilder

   builder = HybridBuilder()
   
   # Basic: Just change the head
   model = builder.from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=10,
   )

Adding Attention
----------------

The most common use case is adding attention mechanisms to improve feature extraction:

.. code-block:: python

   model = builder.from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       patches={
           "layer3": {"wrap": "se"},           # Squeeze-Excitation
           "layer4": {"wrap": "cbam_block"},   # CBAM (Channel + Spatial)
       },
       num_classes=100,
   )

Available attention blocks:

* ``se`` - Squeeze-and-Excitation
* ``cbam_block`` - Convolutional Block Attention Module
* ``eca`` - Efficient Channel Attention
* ``channel_attention`` - Channel attention only
* ``spatial_attention`` - Spatial attention only

Patch Operations
----------------

There are three types of patch operations:

wrap
^^^^

Wraps the target layer with an attention/block module:

.. code-block:: python

   patches = {
       "layer3": {
           "wrap": {
               "type": "se",
               "params": {"reduction": 16}
           }
       }
   }

The result is: ``original_layer → attention_block``

inject
^^^^^^

Injects a block after the target layer:

.. code-block:: python

   patches = {
       "layer3": {"inject": "eca"}
   }

The result is: ``original_layer → eca_block``

replace
^^^^^^^

Replaces the layer entirely (use with caution):

.. code-block:: python

   patches = {
       "layer1": {"replace": {"type": "conv_bn_act", "params": {"channels": 64}}}
   }

Fine-tuning Strategies
----------------------

Frozen Backbone
^^^^^^^^^^^^^^^

Freeze the backbone and only train the head (fastest training):

.. code-block:: python

   model = builder.from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=10,
       freeze_backbone=True,
   )
   
   # Only the head is trainable
   print(f"Trainable: {model.count_parameters(trainable_only=True):,}")

Partial Unfreezing
^^^^^^^^^^^^^^^^^^

Keep only later stages trainable (recommended for most tasks):

.. code-block:: python

   model = builder.from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=10,
       freeze_backbone=True,
       unfreeze_stages=[2, 3],  # Train layer3 and layer4
   )

Progressive Unfreezing
^^^^^^^^^^^^^^^^^^^^^^

Start frozen, then gradually unfreeze:

.. code-block:: python

   # Start with frozen backbone
   model = builder.from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=10,
       freeze_backbone=True,
   )
   
   # Train for a few epochs...
   
   # Unfreeze last stage
   model.freeze_backbone(unfreeze_stages=[3])
   
   # Train more...
   
   # Finally unfreeze everything
   model.unfreeze_all()

Working with Different Backbones
--------------------------------

ResNet Family
^^^^^^^^^^^^^

.. code-block:: python

   # ResNet-18 (lightweight)
   model = builder.from_torchvision("resnet18", weights="DEFAULT", num_classes=10)
   
   # ResNet-101 (deeper)
   model = builder.from_torchvision("resnet101", weights="IMAGENET1K_V2", num_classes=100)
   
   # Wide ResNet (more channels)
   model = builder.from_torchvision("wide_resnet50_2", weights="IMAGENET1K_V2", num_classes=100)

EfficientNet Family
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # EfficientNet-B0 (smallest)
   model = builder.from_torchvision("efficientnet_b0", weights="IMAGENET1K_V1", num_classes=10)
   
   # EfficientNet-B4 (good balance)
   model = builder.from_torchvision("efficientnet_b4", weights="IMAGENET1K_V1", num_classes=100)
   
   # Note: EfficientNet patches use different layer names
   patches = {
       "features.5": {"wrap": "eca"},  # MBConv block 5
   }

ConvNeXt Family
^^^^^^^^^^^^^^^

.. code-block:: python

   # ConvNeXt Tiny (modern architecture)
   model = builder.from_torchvision(
       "convnext_tiny",
       weights="IMAGENET1K_V1",
       num_classes=10,
   )

MobileNet Family
^^^^^^^^^^^^^^^^

.. code-block:: python

   # MobileNet V3 (mobile-optimized)
   model = builder.from_torchvision(
       "mobilenet_v3_large",
       weights="IMAGENET1K_V1",
       num_classes=10,
   )

Weight Utilities
----------------

Partial Loading
^^^^^^^^^^^^^^^

When customizing models, some weights may not match. Use ``partial_load``:

.. code-block:: python

   from torchvision_customizer import partial_load

   # Load checkpoint with mismatch tolerance
   report = partial_load(
       model,
       checkpoint_state_dict,
       ignore_mismatch=True,
       init_new_layers="kaiming",
   )
   
   print(report.summary())

Weight Transfer
^^^^^^^^^^^^^^^

Transfer weights between different models:

.. code-block:: python

   from torchvision_customizer import transfer_weights

   # Transfer all weights except classifier
   transfer_weights(
       source=pretrained_model,
       target=custom_model,
       exclude_patterns=['fc', 'classifier'],
   )

Extracting Features
-------------------

For tasks like object detection or segmentation:

.. code-block:: python

   model = builder.from_torchvision("resnet50", ...)
   
   # Get intermediate stage outputs (for FPN)
   x = torch.randn(1, 3, 224, 224)
   features = model.get_stage_outputs(x)
   
   # features[0]: stem output
   # features[1]: layer1 output
   # features[2]: layer2 output
   # features[3]: layer3 output
   # features[4]: layer4 output
   
   # Or just get final features (before head)
   final_features = model.forward_features(x)

YAML Recipes for Hybrid Models
------------------------------

Define hybrid models in YAML:

.. code-block:: yaml

   # hybrid_model.yaml
   name: ResNet50-SE-Custom
   
   backbone:
     name: resnet50
     weights: IMAGENET1K_V2
     patches:
       layer3:
         wrap:
           type: se
           params:
             reduction: 16
       layer4:
         wrap: cbam_block
   
   head:
     num_classes: 100
     dropout: 0.3

Load with:

.. code-block:: python

   from torchvision_customizer.recipe import load_yaml_recipe
   model = load_yaml_recipe("hybrid_model.yaml")

Best Practices
--------------

1. **Start with a good backbone**: Use ImageNet V2 weights when available
2. **Match input resolution**: EfficientNet-B4 expects 380x380, not 224x224
3. **Freeze early, unfreeze later**: Start with frozen backbone for stability
4. **Add attention sparingly**: 1-2 attention layers usually suffice
5. **Monitor memory**: Larger backbones need more GPU memory
6. **Use appropriate dropout**: Higher for small datasets, lower for large

Troubleshooting
---------------

**"Unknown backbone"**
   Check ``HybridBuilder.list_backbones()`` for supported names.

**"Target not found"**
   Use the exact layer name from the model. Print ``model.named_children()`` to see available names.

**"Shape mismatch"**
   This is normal when changing architectures. Use ``partial_load`` with ``ignore_mismatch=True``.

**Out of memory**
   Use a smaller backbone, reduce batch size, or enable gradient checkpointing.

