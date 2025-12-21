---
title: 'torchvision-customizer: A Modular Framework for Building and Customizing CNN Architectures in PyTorch'
tags:
  - Python
  - deep learning
  - convolutional neural networks
  - PyTorch
  - transfer learning
  - computer vision
  - attention mechanisms
authors:
  - name: Ahsan Umar
    orcid: 0009-0005-8823-2686
    affiliation: 1
    corresponding: true
affiliations:
  - name: Independent Researcher, Pakistan
    index: 1
date: 21 December 2025
bibliography: paper/references.bib
---

# Summary

`torchvision-customizer` is an open-source Python library that provides a flexible, modular framework for building, customizing, and fine-tuning convolutional neural network (CNN) architectures in PyTorch. The library addresses a fundamental challenge in deep learning research: the difficulty of modifying pre-trained model architectures without extensive code changes, copy-pasting model source code, or using fragile monkey-patching techniques.

The framework introduces a **three-tier API architecture** designed to accommodate different user needs:

1. **Component Registry** — A centralized catalog of 40+ building blocks for discovery and instantiation
2. **Architecture Recipes** — Declarative YAML configurations with macros, inheritance, and schema validation
3. **Model Composer** — A fluent Python API using the `>>` operator for intuitive model construction

Version 2.1 introduces the **Hybrid Builder**, which enables researchers to load any pre-trained torchvision model and apply targeted modifications—such as injecting attention mechanisms, replacing layers, or modifying block structures—while automatically preserving compatible pre-trained weights. Version 2.1.1 adds a **Training API** for quick experimentation on MNIST, CIFAR-10, and CIFAR-100 datasets.

# Statement of Need

Deep learning research frequently requires architectural experimentation. Researchers may wish to investigate adding attention mechanisms to existing architectures [@hu2018squeeze; @woo2018cbam; @wang2020eca], replacing normalization strategies, or combining components from different architectural families [@he2016deep; @tan2019efficientnet; @liu2022convnet]. However, the current ecosystem presents significant friction for such modifications.

Libraries like **torchvision** [@torchvision2016] provide well-tested implementations with pre-trained weights, but modifying these models—for instance, injecting Squeeze-and-Excitation blocks into a ResNet—requires understanding internal implementation details and manually modifying the forward pass. The **timm** library [@wightman2019timm] offers extensive model variants but focuses on providing ready-to-use architectures rather than enabling user-driven customization.

Current approaches to architectural modification include:

- **Code duplication**: Copying model source code into projects and modifying directly, which creates maintenance burden and makes it difficult to incorporate upstream improvements
- **Monkey-patching**: Runtime modification of model behavior, which is fragile, difficult to reproduce, and error-prone
- **Manual reimplementation**: Writing custom model classes from scratch, requiring substantial boilerplate for channel tracking and component interconnection

`torchvision-customizer` addresses these limitations through a principled approach to model composition and customization, enabling:

- **Block-level customization** of pre-trained models through a patch-based modification system
- **Automatic weight preservation** when modifying architectures, with graceful handling of shape mismatches
- **Declarative model definition** through YAML recipes for reproducible experiments
- **Rapid prototyping** of custom architectures with automatic channel dimension tracking
- **Built-in training utilities** for quick experimentation on standard datasets

# Key Features

## Hybrid Builder

The Hybrid Builder enables customization of pre-trained torchvision models without requiring users to understand internal implementation details. Users specify a backbone name, optional pre-trained weights, and a dictionary of patches describing modifications. The framework handles architecture decomposition, patch application, weight preservation, and head replacement automatically.

Supported backbone families include:

| Family | Models | Reference |
|--------|--------|-----------|
| ResNet | resnet18, 34, 50, 101, 152 | @he2016deep |
| Wide ResNet | wide_resnet50_2, 101_2 | @zagoruyko2016wide |
| ResNeXt | resnext50_32x4d, 101_32x8d | @xie2017aggregated |
| EfficientNet | B0-B7, V2 variants | @tan2019efficientnet; @tan2021efficientnetv2 |
| ConvNeXt | tiny, small, base, large | @liu2022convnet |
| MobileNet | v2, v3_small, v3_large | @howard2017mobilenets; @sandler2018mobilenetv2 |
| VGG | 11, 13, 16, 19 (with/without BN) | @simonyan2014very |
| DenseNet | 121, 169, 201, 161 | @huang2017densely |
| Vision Transformer | vit_b_16, vit_b_32, vit_l_16, vit_l_32 | @dosovitskiy2020image |
| Swin Transformer | swin_t, swin_s, swin_b | @liu2021swin |

## Building Blocks

The library provides over 40 building blocks organized into functional categories:

- **Attention mechanisms**: Squeeze-and-Excitation [@hu2018squeeze], CBAM [@woo2018cbam], and Efficient Channel Attention [@wang2020eca]
- **Convolution blocks**: Standard, depthwise separable [@howard2017mobilenets], and mobile inverted bottleneck (MBConv) variants [@tan2019efficientnet]
- **Residual blocks**: Basic, bottleneck [@he2016deep], wide [@zagoruyko2016wide], and grouped convolution [@xie2017aggregated] variants
- **Regularization**: Stochastic depth (DropPath) [@huang2016deep] for training regularization
- **Activations**: Including Mish [@misra2019mish] and standard PyTorch activations
- **Pooling**: Including Generalized Mean Pooling (GeM) [@radenovic2018fine] with learnable parameters

## Weight Utilities

The framework includes utilities for **partial weight loading** that gracefully handle architectural mismatches. When source and target architectures differ, the utilities:

1. Identify matching parameters by name
2. Verify shape compatibility
3. Load only the compatible subset
4. Initialize new parameters using specified schemes (Kaiming, Xavier, or zero initialization)
5. Produce detailed reports indicating loading success rates and any issues encountered

## Training API

Version 2.1.1 introduces built-in training utilities for quick experimentation:

- **Trainer class** with methods for MNIST, CIFAR-10, CIFAR-100, and custom datasets
- **Optimizers**: Adam, AdamW, SGD with momentum
- **Schedulers**: Cosine annealing, step decay, OneCycle
- **Automatic device selection** (CPU/CUDA)
- **Training metrics** with history tracking and summaries

## YAML Recipes

Models can be defined declaratively using YAML configuration files with:

- **Macro expansion** for reusable values
- **Recipe inheritance** from base configurations
- **Schema validation** for early error detection

## Command-Line Interface

The CLI (`tvc` command) provides access to common operations:

- `tvc build` — Build model from YAML recipe
- `tvc benchmark` — Benchmark model performance
- `tvc validate` — Validate recipe configuration
- `tvc export` — Export to ONNX/TorchScript
- `tvc list-backbones` — List available backbone architectures
- `tvc list-blocks` — List available building blocks

# Comparison with Related Work

| Feature | torchvision | timm | torchvision-customizer |
|---------|-------------|------|------------------------|
| Pre-trained models | ✓ | ✓ | ✓* |
| Block-level customization | — | — | ✓ |
| Attention injection | — | — | ✓ |
| Declarative configuration | — | — | ✓ |
| Fluent composition API | — | — | ✓ |
| Partial weight loading | — | ✓ | ✓ |
| Command-line interface | — | — | ✓ |
| Recipe inheritance | — | — | ✓ |
| Built-in training API | — | — | ✓ |

*Through Hybrid Builder integration with torchvision

The library complements existing tools by providing a customization layer that operates on top of torchvision models. Rather than reimplementing architectures, we leverage existing torchvision implementations and provide mechanisms for targeted modification, ensuring compatibility with the broader PyTorch ecosystem.

# Target Audience

`torchvision-customizer` is designed for:

- **Academic researchers** investigating attention mechanisms, normalization techniques, or novel architectural components who need to inject modifications into established architectures while leveraging pre-trained representations
- **Applied practitioners** developing domain-specific computer vision solutions who need to efficiently adapt pre-trained models with selective layer freezing for effective transfer learning
- **Educators** teaching CNN architecture design who can use the explicit composition syntax to make design choices visible and modifiable
- **AutoML researchers** requiring programmatic architecture generation who can leverage the registry and composer APIs for integration with architecture search algorithms

# Software Availability

`torchvision-customizer` is open-source software released under the MIT license. The source code is hosted on GitHub at [https://github.com/codewithdark-git/torchvision-customizer](https://github.com/codewithdark-git/torchvision-customizer). The repository includes comprehensive documentation, API reference, user guides, and example scripts. Requirements include Python 3.11+, PyTorch 2.5+, and torchvision 0.20+.

# Acknowledgments

We thank the PyTorch development team for creating and maintaining the foundational framework upon which this work builds. We also acknowledge the broader open-source community whose feedback has shaped the library's development.

# References

