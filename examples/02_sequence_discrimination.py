"""
Sequence Discrimination: A→B vs B→A

Demonstrates that an IntegratorNeuron with fixed delays can discriminate
temporal sequences that a standard weighted-sum neuron cannot.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from integrator_neuron import IntegratorNeuron


def main():
    print("=" * 70)
    print("SEQUENCE DISCRIMINATION: A→B vs B→A")
    print("=" * 70)
    
    neuron = IntegratorNeuron(
        num_features=2,
        num_dendritic_branches=1,
        num_proximal_inputs=1,
        threshold=1.5,
        alpha=2.5,
        tau_membrane=5.0,
        v_threshold=1.0
    )
    
    neuron.configure_for_sequence([0, 1], [5, 0])
    
    with torch.no_grad():
        neuron.v_weights[0] = 0.2  # Weak V
    
    batch_size = 3
    time_steps = 30
    q_spikes = torch.zeros(batch_size, time_steps, 2)
    
    # Batch 0: A→B
    q_spikes[0, 5, 0] = 1.0
    q_spikes[0, 10, 1] = 1.0
    
    # Batch 1: B→A
    q_spikes[1, 5, 1] = 1.0
    q_spikes[1, 10, 0] = 1.0
    
    # Batch 2: Control (no Q spikes)
    
    v_input = torch.ones(batch_size, 1) * 0.2
    
    with torch.no_grad():
        output = neuron(q_spikes, v_input, return_trace=True)
    
    print("\nConfiguration: Delay A=5, Delay B=0")
    print("Threshold: 1.5 (requires both inputs coincident)\n")
    
    results = [
        ("A→B (A at t=5, B at t=10)", 0),
        ("B→A (B at t=5, A at t=10)", 1),
        ("Control (no Q spikes)", 2),
    ]
    
    for name, b in results:
        spikes = torch.where(output[b] > 0)[0].tolist()
        print(f"{name:30} → spikes: {spikes if spikes else '[]':10} (total: {output[b].sum().item():.0f})")
    
    print("\n" + "-" * 40)
    
    # Standard neuron comparison
    print("\nStandard weighted-sum neuron:")
    print(f"  A→B sum: {q_spikes[0].sum().item():.0f} spikes")
    print(f"  B→A sum: {q_spikes[1].sum().item():.0f} spikes")
    print("  → CANNOT distinguish!")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
    The integrator neuron distinguishes A→B from B→A via conduction delays.
    This is the biological equivalent of Q·K matching in Transformers—
    but with hard gating instead of softmax.
    """)


if __name__ == "__main__":
    main()
