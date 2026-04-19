# Integrator Neuron: Dendritic Coincidence Detection as Biological Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

This repository implements the core computational models from:

**"Deductive Inference of Brain Structures and a Unified Framework for AGI"**  
*Cheng Bi*

📄 **Paper DOI**:  10.13140/RG.2.2.26341.56801

---

## Overview

The brain's integrator neurons (cortical L5 pyramidal neurons, cerebellar Purkinje cells) perform a computation structurally analogous to Transformer cross-attention—but with crucial differences that suggest architectural improvements for AI systems.

### Biological → Transformer Mapping

| Biological Component | Transformer Equivalent |
|---------------------|----------------------|
| Q: Spike trains from feature neurons | Q: Query matrix |
| K: Conduction delay matrix | K: Key matrix |
| V: Proximal dendritic inputs | V: Value matrix |
| Distal coincidence detection | Q·K matching |
| Hard threshold (binary vote) | **Softmax (continuous weights)** |
| Soma: α·Σ(votes) + Σ(w·V) | Weighted sum + residual |

### Key Difference: Hard Gating vs Softmax
