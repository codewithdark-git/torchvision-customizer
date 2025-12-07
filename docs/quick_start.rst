Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install torchvision-customizer

Hello World (ResNet-50)
-----------------------

.. code-block:: python

   from torchvision_customizer import resnet
   model = resnet(layers=50, num_classes=100)

Hello World (Custom)
--------------------

.. code-block:: python

   from torchvision_customizer import Stem, Stage, Head
   
   model = (Stem(64) 
            >> Stage(128, blocks=2, pattern='residual+se') 
            >> Head(10))
