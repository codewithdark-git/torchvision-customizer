Hybrid Model Examples
=====================

This page provides practical examples of using the Hybrid Builder for various tasks.

Example 1: Image Classification with Attention
----------------------------------------------

Fine-tune ResNet50 with SE attention for CIFAR-100:

.. code-block:: python

   import torch
   import torch.nn as nn
   from torchvision_customizer import HybridBuilder

   # Create model
   model = HybridBuilder().from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       patches={
           "layer3": {"wrap": "se"},
           "layer4": {"wrap": "cbam_block"},
       },
       num_classes=100,
       dropout=0.3,
       freeze_backbone=True,
       unfreeze_stages=[3],
   )

   # Training setup
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   
   optimizer = torch.optim.AdamW(
       filter(lambda p: p.requires_grad, model.parameters()),
       lr=1e-4,
       weight_decay=0.01
   )
   
   criterion = nn.CrossEntropyLoss()

   # Training loop
   model.train()
   for epoch in range(10):
       for images, labels in train_loader:
           images, labels = images.to(device), labels.to(device)
           
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()

Example 2: Fine-grained Classification
--------------------------------------

Use EfficientNet-B4 for fine-grained bird classification:

.. code-block:: python

   from torchvision_customizer import HybridBuilder

   # EfficientNet-B4 is good for fine-grained tasks
   model = HybridBuilder().from_torchvision(
       "efficientnet_b4",
       weights="IMAGENET1K_V1",
       num_classes=200,  # CUB-200 birds
       dropout=0.4,  # Higher dropout for small dataset
   )

   # Progressive unfreezing schedule
   phases = [
       (10, []),      # Phase 1: Head only
       (20, [6, 7]),  # Phase 2: Last 2 stages
       (50, None),    # Phase 3: Full fine-tuning
   ]

   for epochs, unfreeze in phases:
       if unfreeze is not None:
           model.freeze_backbone(unfreeze_stages=unfreeze)
       else:
           model.unfreeze_all()
       
       # Train for specified epochs...

Example 3: Medical Image Classification
---------------------------------------

ConvNeXt for chest X-ray classification:

.. code-block:: python

   from torchvision_customizer import HybridBuilder

   model = HybridBuilder().from_torchvision(
       "convnext_base",
       weights="IMAGENET1K_V1",
       num_classes=5,  # 5 disease classes
       dropout=0.5,   # High dropout for medical images
       freeze_backbone=True,
       unfreeze_stages=[2, 3],
   )

   # Important: Medical images often need different preprocessing
   # Make sure to adjust your transforms accordingly

Example 4: Multi-label Classification
-------------------------------------

Custom head for multi-label classification:

.. code-block:: python

   import torch.nn as nn
   from torchvision_customizer import HybridBuilder

   # Build base model
   model = HybridBuilder().from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=80,  # Will replace this
   )

   # Replace head with custom multi-label head
   num_features = model.head[2].in_features  # Get feature dim
   model.head = nn.Sequential(
       nn.AdaptiveAvgPool2d((1, 1)),
       nn.Flatten(1),
       nn.Dropout(0.3),
       nn.Linear(num_features, 80),
       nn.Sigmoid(),  # Multi-label activation
   )

   # Use BCELoss for training
   criterion = nn.BCELoss()

Example 5: Feature Extraction for Object Detection
--------------------------------------------------

Extract features for FPN-based detection:

.. code-block:: python

   from torchvision_customizer import HybridBuilder
   import torch

   # Create backbone
   model = HybridBuilder().from_torchvision(
       "resnet50",
       weights="IMAGENET1K_V2",
       num_classes=1000,  # Doesn't matter, we'll use features
   )

   # Get multi-scale features
   x = torch.randn(1, 3, 800, 800)
   features = model.get_stage_outputs(x)

   # Use for FPN
   # features[1] -> P2 (1/4 resolution)
   # features[2] -> P3 (1/8 resolution)
   # features[3] -> P4 (1/16 resolution)
   # features[4] -> P5 (1/32 resolution)

   for i, f in enumerate(features):
       print(f"Stage {i}: {f.shape}")

Example 6: Loading from Checkpoint
----------------------------------

Resume training from a checkpoint:

.. code-block:: python

   from torchvision_customizer import HybridBuilder, partial_load
   import torch

   # Create model with same architecture
   model = HybridBuilder().from_torchvision(
       "resnet50",
       weights=None,  # Don't load pretrained
       patches={"layer4": {"wrap": "se"}},
       num_classes=100,
   )

   # Load your checkpoint
   checkpoint = torch.load("checkpoint.pt")
   
   # Use partial_load for robustness
   report = partial_load(
       model,
       checkpoint['model_state_dict'],
       ignore_mismatch=True,
   )
   
   print(f"Loaded {report.load_ratio:.1%} of weights")

Example 7: YAML-based Training Pipeline
---------------------------------------

Define everything in YAML:

.. code-block:: yaml

   # train_config.yaml
   name: ResNet50-SE-CIFAR100
   
   backbone:
     name: resnet50
     weights: IMAGENET1K_V2
     patches:
       layer3: {wrap: se}
       layer4: {wrap: cbam_block}
   
   head:
     num_classes: 100
     dropout: 0.3
   
   training:
     optimizer: adamw
     learning_rate: 0.0001
     weight_decay: 0.01
     epochs: 50
     batch_size: 32

.. code-block:: python

   from torchvision_customizer.recipe import load_yaml_config, load_yaml_recipe

   # Load config
   config = load_yaml_config("train_config.yaml")
   
   # Build model
   model = load_yaml_recipe("train_config.yaml")
   
   # Use training hints
   training = config.get('training', {})
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=training.get('learning_rate', 0.001),
       weight_decay=training.get('weight_decay', 0.01),
   )

Example 8: Comparing Different Attention Types
----------------------------------------------

Benchmark different attention mechanisms:

.. code-block:: python

   from torchvision_customizer import HybridBuilder
   import torch
   import time

   builder = HybridBuilder()
   attention_types = ['se', 'eca', 'cbam_block']

   for attn in attention_types:
       model = builder.from_torchvision(
           "resnet50",
           weights="IMAGENET1K_V2",
           patches={"layer4": {"wrap": attn}},
           num_classes=100,
       )
       model.eval()
       
       # Benchmark
       x = torch.randn(16, 3, 224, 224)
       
       with torch.no_grad():
           start = time.time()
           for _ in range(100):
               _ = model(x)
           elapsed = time.time() - start
       
       params = model.count_parameters()
       print(f"{attn}: {elapsed:.2f}s, {params:,} params")

Example 9: Lightweight Mobile Model
-----------------------------------

Optimize for mobile deployment:

.. code-block:: python

   from torchvision_customizer import HybridBuilder

   # Use MobileNet V3 for mobile
   model = HybridBuilder().from_torchvision(
       "mobilenet_v3_small",
       weights="IMAGENET1K_V1",
       num_classes=10,
       dropout=0.2,
   )

   # Export to ONNX for mobile
   import torch
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(
       model,
       dummy_input,
       "mobilenet_custom.onnx",
       opset_version=11,
   )

Example 10: Vision Transformer Hybrid
-------------------------------------

Customize a Vision Transformer:

.. code-block:: python

   from torchvision_customizer import HybridBuilder

   # ViT has different structure
   model = HybridBuilder().from_torchvision(
       "vit_b_16",
       weights="IMAGENET1K_V1",
       num_classes=100,
   )

   # ViT freezing works differently
   # Freeze everything except head and last 2 encoder layers
   for name, param in model.named_parameters():
       if 'head' not in name and 'encoder.layers.10' not in name and 'encoder.layers.11' not in name:
           param.requires_grad = False

