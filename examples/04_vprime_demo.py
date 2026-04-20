"""
Full V' Pathway Demonstration

Shows the complete hippocampal index-content separation architecture:
Raw V → DG → CA3 → V' → Integrator Neuron
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from integrator_neuron_vprime import IntegratorNeuronWithVPrime


def main():
    print("=" * 70)
    print("FULL V' PATHWAY DEMONSTRATION")
    print("=" * 70)
    
    batch_size = 2
    time_steps = 30
    num_features = 2
    num_proximal = 8
    
    # Q spikes
    q_spikes = torch.zeros(batch_size, time_steps, num_features)
    q_spikes[0, 5, 0] = 1.0   # A→B
    q_spikes[0, 10, 1] = 1.0
    q_spikes[1, 5, 1] = 1.0   # B→A
    q_spikes[1, 10, 0] = 1.0
    
    # Raw V inputs (different → different CA3 sequences)
    v_raw = torch.zeros(batch_size, num_proximal)
    v_raw[0, 0:3] = 1.0   # Batch 0: match 'a'
    v_raw[1, 4:7] = 1.0   # Batch 1: match 'd'
    
    # Create neuron
    neuron = IntegratorNeuronWithVPrime(
        num_features=2,
        num_proximal_inputs=num_proximal,
        num_dendritic_branches=1,
        dg_expansion=8,
        threshold=1.5,
        alpha=2.5,
        tau_membrane=5.0,
        v_threshold=1.0,
        v_weight_scale=0.15
    )
    
    with torch.no_grad():
        neuron.delays[0, 0] = 5
        neuron.delays[0, 1] = 0
        neuron.q_weights[0, 0] = 1.0
        neuron.q_weights[0, 1] = 1.0
        neuron.v_weights.data = torch.tensor([0.12, 0.10, 0.08, 0.14, 0.09, 0.11, 0.13, 0.07])
    
    with torch.no_grad():
        output, debug = neuron(q_spikes, v_raw, return_debug_info=True)
    
    print("\n" + "=" * 70)
    print("HIPPOCAMPAL QUERY RESULTS")
    print("=" * 70)
    
    for b in range(batch_size):
        print(f"\nBatch {b}:")
        print(f"  V input: {v_raw[b].tolist()}")
        print(f"  Matched index: '{debug['matched_chars'][b]}'")
        print(f"  Retrieved: {debug['char_seqs'][b]}")
        print(f"  V' pattern: {debug['pattern_descs'][b]}")
    
    print("\n" + "=" * 70)
    print("NEURON OUTPUT")
    print("=" * 70)
    
    for b in range(batch_size):
        spikes = torch.where(output[b] > 0)[0].tolist()
        votes = torch.where(debug['votes'][b] > 0)[0].tolist()
        seq_type = "A→B" if b == 0 else "B→A"
        
        print(f"\nBatch {b} ({seq_type}):")
        print(f"  Dendritic vote: {votes if votes else 'NONE'}")
        print(f"  Output spikes: {spikes if spikes else '[]'}")
        print(f"  Total: {output[b].sum().item():.0f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
    Different V inputs → Different CA3 sequences → Different V' patterns
    → Different outputs (Batch 0 fires, Batch 1 doesn't)
    
    This demonstrates index-content separation:
    - Hippocampus stores indices (a-b-c vs d-e-f)
    - Cortex stores content (V'
