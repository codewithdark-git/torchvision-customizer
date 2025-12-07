Using Templates
===============

Parametric implementations of standard architectures.

Available Templates
-------------------

* **ResNet**: 18, 34, 50, 101, 152 layers
* **VGG**: 11, 13, 16, 19 layers
* **MobileNet**: V1, V2, V3 (Small/Large)
* **DenseNet**: 121, 169, 201, 264 layers
* **EfficientNet**: B0-B7 variants

Customization
-------------

Templates support method chaining for deep customization:

.. code-block:: python

    from torchvision_customizer import Template
    
    model = (Template.resnet(layers=50)
             .replace_activation('swish')
             .replace_norm('group')
             .add_attention('cbam')
             .build(num_classes=10))
