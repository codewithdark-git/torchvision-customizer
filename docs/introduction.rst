Introduction
============

Background
----------

Deep learning research often involves tweaking model architectures—changing normalization, adding attention, modifying block types. Standard libraries like `torchvision` offer pre-built models (e.g., ``resnet50``), but modifying them often requires monkey-patching or copy-pasting code.

Our Approach
------------

**torchvision-customizer** solves this by decomposing architectures into reusable, composable components.

1.  **Registry**: All blocks (Conv, Residual, Bottleneck, SE) are registered components.
2.  **Recipes**: Architectures are defined by declarative strings, not by hardcoded class definitions.
3.  **Composition**: Operators (`>>`) make it easy to chain processing stages together.

Philosophy
----------

*   **No Magic**: Components are standard `nn.Module` subclasses.
*   **No Weights**: This library builds architectures. It does not provide pre-trained weights (load them yourself or train from scratch).
*   **Explicit > Implicit**: Channel dimensions and shapes are handled explicitly where needed, or inferred when safe.
