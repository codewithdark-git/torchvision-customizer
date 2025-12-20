Building Blocks
===============

.. module:: torchvision_customizer.blocks

All building blocks are composable ``nn.Module`` subclasses that can be used
in the Composer API, Recipes, or standalone.

Standard Blocks
---------------

ConvBlock
^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.ConvBlock
   :members:

Basic convolutional block with configurable normalization, activation, and pooling.

.. code-block:: python

   from torchvision_customizer.blocks import ConvBlock
   
   block = ConvBlock(
       in_channels=64,
       out_channels=128,
       kernel_size=3,
       activation='relu',
       norm_type='batch',
   )

ResidualBlock
^^^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.ResidualBlock
   :members:

Residual block with skip connection.

.. code-block:: python

   from torchvision_customizer.blocks import ResidualBlock
   
   block = ResidualBlock(in_channels=64, out_channels=64)

SEBlock
^^^^^^^

.. autoclass:: torchvision_customizer.blocks.SEBlock
   :members:

Squeeze-and-Excitation attention block.

.. code-block:: python

   from torchvision_customizer.blocks import SEBlock
   
   se = SEBlock(channels=256, reduction=16)

DepthwiseBlock
^^^^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.DepthwiseBlock
   :members:

Depthwise separable convolution (MobileNet-style).

Advanced Blocks
---------------

InceptionModule
^^^^^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.InceptionModule
   :members:

Multi-branch Inception-style module.

DenseConnectionBlock
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.DenseConnectionBlock
   :members:

DenseNet-style dense connections.

Bottleneck Variants
-------------------

StandardBottleneck
^^^^^^^^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.StandardBottleneck
   :members:

Standard bottleneck block (1x1 → 3x3 → 1x1).

WideBottleneck
^^^^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.WideBottleneck
   :members:

Wide bottleneck with more channels.

GroupedBottleneck
^^^^^^^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.GroupedBottleneck
   :members:

Grouped convolution bottleneck (ResNeXt-style).

v2.1 Blocks (NEW)
-----------------

These blocks were added in v2.1 for modern architectures.

CBAMBlock
^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.CBAMBlock
   :members:

Convolutional Block Attention Module - combines channel and spatial attention.

.. code-block:: python

   from torchvision_customizer.blocks import CBAMBlock
   
   cbam = CBAMBlock(channels=256, reduction=16, spatial_kernel=7)
   out = cbam(x)  # Same shape as input

ECABlock
^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.ECABlock
   :members:

Efficient Channel Attention - lightweight alternative to SE.

.. code-block:: python

   from torchvision_customizer.blocks import ECABlock
   
   eca = ECABlock(channels=512)
   out = eca(x)

DropPath
^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.DropPath
   :members:

Stochastic Depth / Drop Path regularization.

.. code-block:: python

   from torchvision_customizer.blocks import DropPath
   
   drop_path = DropPath(drop_prob=0.2)
   out = x + drop_path(residual)  # Apply to residual branch

Mish
^^^^

.. autoclass:: torchvision_customizer.blocks.Mish
   :members:

Mish activation function: ``x * tanh(softplus(x))``.

.. code-block:: python

   from torchvision_customizer.blocks import Mish
   
   mish = Mish()
   out = mish(x)

GeM
^^^

.. autoclass:: torchvision_customizer.blocks.GeM
   :members:

Generalized Mean Pooling with learnable pooling power.

.. code-block:: python

   from torchvision_customizer.blocks import GeM
   
   gem = GeM(p=3.0, learnable=True)
   out = gem(x)  # Global pooling

MBConv
^^^^^^

.. autoclass:: torchvision_customizer.blocks.MBConv
   :members:

Mobile Inverted Bottleneck (EfficientNet-style).

.. code-block:: python

   from torchvision_customizer.blocks import MBConv
   
   block = MBConv(
       in_channels=32,
       out_channels=64,
       expansion=4,
       stride=2,
       use_se=True,
   )

FusedMBConv
^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.FusedMBConv
   :members:

Fused MBConv (EfficientNetV2-style) - faster training.

.. code-block:: python

   from torchvision_customizer.blocks import FusedMBConv
   
   block = FusedMBConv(
       in_channels=32,
       out_channels=64,
       expansion=4,
       stride=1,
   )

SqueezeExcitation
^^^^^^^^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.SqueezeExcitation
   :members:

Enhanced SE block with configurable activation and gate.

LayerScale
^^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.LayerScale
   :members:

Per-channel learnable scaling (used in Vision Transformers).

ConvBNAct
^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.ConvBNAct
   :members:

Convenience block: Conv2d + BatchNorm + Activation.

CoordConv
^^^^^^^^^

.. autoclass:: torchvision_customizer.blocks.CoordConv
   :members:

Coordinate Convolution - adds coordinate channels to input.

Block Reference Table
---------------------

+---------------------+------------------+-------------------------+
| Block               | Registry Name    | Use Case                |
+=====================+==================+=========================+
| ConvBlock           | ``conv``         | Basic convolution       |
+---------------------+------------------+-------------------------+
| ResidualBlock       | ``residual``     | Skip connections        |
+---------------------+------------------+-------------------------+
| SEBlock             | ``se``           | Channel attention       |
+---------------------+------------------+-------------------------+
| CBAMBlock           | ``cbam_block``   | Channel+Spatial attn    |
+---------------------+------------------+-------------------------+
| ECABlock            | ``eca``          | Efficient channel attn  |
+---------------------+------------------+-------------------------+
| DropPath            | ``drop_path``    | Stochastic depth        |
+---------------------+------------------+-------------------------+
| MBConv              | ``mbconv``       | EfficientNet blocks     |
+---------------------+------------------+-------------------------+
| FusedMBConv         | ``fused_mbconv`` | EfficientNetV2 blocks   |
+---------------------+------------------+-------------------------+
| GeM                 | ``gem``          | Retrieval/pooling       |
+---------------------+------------------+-------------------------+
| Bottleneck          | ``bottleneck``   | ResNet-50+ blocks       |
+---------------------+------------------+-------------------------+

Using Blocks with Registry
--------------------------

.. code-block:: python

   from torchvision_customizer import registry
   
   # Get block class
   CBAMBlock = registry.get('cbam_block')
   block = CBAMBlock(channels=256)
   
   # Or instantiate directly
   block = registry.get('eca', channels=512)
   
   # List all blocks
   print(registry.list('block'))
   print(registry.list('attention'))
