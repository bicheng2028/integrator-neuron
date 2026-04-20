"""
Basic Integrator Neuron Demonstration

This script demonstrates the core functionality of the IntegratorNeuron:
- Q spikes with conduction delays
- Dendritic coincidence detection
- Somatic integration and spike output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from integrator_neuron import IntegratorNeuron, print_comparison_table


def main():
    print("=" * 70)
    print("BASIC INTEGRATOR NEURON DEMONSTRATION")
    print("=" * 70)
    
    print_comparison_table()
    
    # Create neuron
    neuron = IntegratorNeuron(
        num_features=2,
        num_dendritic_branches=1,
        num_proximal_inputs=1,
        threshold=1.5,
        alpha=2.0,
        tau_membrane=5.0,
        v_threshold=1.0,
        refractory_period=2
    )
    
    # Configure delays to detect A→B sequence
    neuron.configure_for_sequence([0, 1], [5, 0])
    
    # Set V weight (weak, so V alone doesn't fire)
    with torch.no_grad():
        neuron.v_weights[0] = 0.3
    
    # Create test sequences
    batch_size = 2
    time_steps = 30
    q_spikes = torch.zeros(batch_size, time_steps, 2)
    
    # Batch 0: A→B
    q_spikes[0, 5, 0] = 1.0
    q_spikes[0, 10, 1] = 1.0
    
    # Batch 1: B→A
    q_spikes[1, 5, 1] = 1.0
    q_spikes[1, 10, 0] = 1.0
    
    v_input = torch.ones(batch_size, 1) * 0.5
    
    # Run
    with torch.no_grad():
        output = neuron(q_spikes, v_input, return_trace=True)
        membrane = neuron(q_spikes, v_input, return_membrane=True)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for b in range(batch_size):
        seq_type = "A→B" if b == 0 else "B→A"
        spikes = torch.where(output[b] > 0)[0].tolist()
        print(f"\nBatch {b} ({seq_type}):")
        print(f"  Output spikes: {spikes if spikes else '[]'}")
        print(f"  Total: {output[b].sum().item():.0f}")
        print(f"  Max membrane: {membrane[b].max().item():.3f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    The neuron fires for A→B but NOT for B→A because:
    - Delays are configured for A→B (A delayed 5, B delayed 0)
    - A→B: A(t=5)+5=10, B(t=10)+0=10 → COINCIDENT → VOTE → SPIKE
    - B→A: B(t=5)+0=5, A(t=10)+5=15 → MISMATCHED → NO VOTE → NO SPIKE
    
    This demonstrates temporal sequence selectivity via conduction delays.
    """)


if __name__ == "__main__":
    main()
