"""
Democratic Voting Demonstration

Each dendritic branch acts as an independent coincidence detector.
The soma counts votes (democratic integration) rather than continuous weighting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from integrator_neuron import IntegratorNeuron


def main():
    print("=" * 70)
    print("DEMOCRATIC VOTING IN DENDRITIC INTEGRATION")
    print("=" * 70)
    
    neuron = IntegratorNeuron(
        num_features=4,
        num_dendritic_branches=5,
        num_proximal_inputs=1,
        threshold=1.5,
        alpha=1.0
    )
    
    # Configure branches with different receptive fields
    with torch.no_grad():
        # Branch 0: features 0 and 1
        neuron.q_weights[0, 0] = 1.0
        neuron.q_weights[0, 1] = 1.0
        
        # Branch 1: features 1 and 2
        neuron.q_weights[1, 1] = 1.0
        neuron.q_weights[1, 2] = 1.0
        
        # Branch 2: features 2 and 3
        neuron.q_weights[2, 2] = 1.0
        neuron.q_weights[2, 3] = 1.0
        
        # Branch 3: features 0 and 3
        neuron.q_weights[3, 0] = 1.0
        neuron.q_weights[3, 3] = 1.0
        
        # Branch 4: all features (lower weights)
        neuron.q_weights[4, :] = 0.5
        
        # No delays
        neuron.delays.data.zero_()
        neuron.v_weights[0] = 0.1
    
    # Test patterns
    patterns = [
        ([1, 0, 0, 0], "Only feature 0"),
        ([1, 1, 0, 0], "Features 0,1"),
        ([1, 1, 1, 0], "Features 0,1,2"),
        ([1, 1, 1, 1], "All features"),
    ]
    
    print("\nThreshold = 1.5 (requires coincident input sum ≥ 1.5)\n")
    print("Pattern              | Active Branches | Votes | Excitation")
    print("-" * 60)
    
    batch = 1
    time_steps = 20
    v_input = torch.ones(batch, 1) * 0.1
    
    for pattern, name in patterns:
        q_spikes = torch.zeros(batch, time_steps, 4)
        q_spikes[0, 10, :] = torch.tensor(pattern)
        
        with torch.no_grad():
            delayed = neuron.apply_delays(q_spikes)
            votes = neuron.dendritic_coincidence_detection(delayed, 10)
            active = torch.where(votes[0] > 0)[0].tolist()
            v_count = votes[0].sum().item()
            v_contrib = neuron.proximal_integration(v_input)[0, 0].item()
            total = v_contrib + neuron.alpha * v_count
        
        pattern_str = str(pattern)
        print(f"{name:20} | {str(active):15} | {v_count:.0f}    | {total:.2f}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
    Each branch makes an INDEPENDENT binary decision:
    - Below threshold: ZERO contribution (hard gating)
    - Above threshold: ONE vote (saturating)
    
    The soma simply COUNTS votes (democratic integration).
    This is fundamentally different from softmax, where all tokens
    contribute continuously.
    """)


if __name__ == "__main__":
    main()
