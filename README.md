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

### Hippocampal V' Pathway

The full model includes the hippocampal index-content separation architecture:
Raw V → Dentate Gyrus (sparse orthogonalization)
→ CA3 (pattern completion: partial → full sequence index)
→ V' decoder (index → spike pattern)
→ Proximal dendrites of integrator neuron


This implements the paper's core claim: **"The hippocampus stores indices, not content."**

---

## Quick Start

```python
import torch
from integrator_neuron import IntegratorNeuron

# Create neuron with 2 feature inputs, 4 dendritic branches
neuron = IntegratorNeuron(
    num_features=2,
    num_dendritic_branches=4,
    num_proximal_inputs=1
)

# Q: spike trains (batch=1, time_steps=30, features=2)
q_spikes = torch.zeros(1, 30, 2)
q_spikes[0, 5, 0] = 1.0   # Feature A at t=5
q_spikes[0, 10, 1] = 1.0  # Feature B at t=10

# V: proximal input
v_input = torch.ones(1, 1)

# Run neuron
output = neuron(q_spikes, v_input, return_trace=True)
print(f"Output spikes at: {torch.where(output[0] > 0)[0].tolist()}")
