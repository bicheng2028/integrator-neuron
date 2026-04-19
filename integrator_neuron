
"""
Integrator Neuron with Dendritic Coincidence Detection

Biological Mapping to Transformer Cross-Attention:
- Q: Spike trains from feature neurons
- K: Conduction delay matrix (per dendritic branch)
- V: Proximal input spikes

Key Difference from Transformer:
- Hard gating via threshold (not softmax)
- Binary dendritic votes (not continuous weights)
- Temporal leaky integration

Author: Cheng Bi
Paper: "Deductive Inference of Brain Structures and a Unified Framework for AGI"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class IntegratorNeuron(nn.Module):
    """
    Integrator Neuron with Distal Dendritic Coincidence Detection.
    
    This neuron implements:
    - Distal dendrites: Q·K matching via delayed coincidence detection
    - Proximal dendrites: Weighted V input
    - Soma: Leaky integration of votes + V input
    
    Architecture:
        Q_spikes → [Delays (K)] → Coincidence detection → Binary votes
        V_input → [Weights] → Weighted sum
        Votes + V_sum → Leaky integrate-and-fire → Output spikes
    """
    
    def __init__(
        self,
        num_features: int,           # Number of feature neurons (Q dimension)
        num_dendritic_branches: int, # Number of distal dendritic branches
        num_proximal_inputs: int,    # Number of proximal V inputs
        max_delay: int = 20,         # Maximum conduction delay (time steps)
        threshold: float = 0.5,      # Coincidence detection threshold
        alpha: float = 1.0,          # Weight of dendritic votes
        tau_membrane: float = 10.0,  # Membrane time constant (ms)
        v_rest: float = 0.0,         # Resting membrane potential
        v_threshold: float = 1.0,    # Spike threshold
        refractory_period: int = 2,  # Refractory period (time steps)
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_branches = num_dendritic_branches
        self.num_proximal = num_proximal_inputs
        self.max_delay = max_delay
        self.threshold = threshold
        self.alpha = alpha
        self.tau = tau_membrane
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.refractory_period = refractory_period
        
        # K Matrix: Conduction delays for each (feature -> branch)
        self.delays = nn.Parameter(
            torch.randint(0, max_delay, (num_dendritic_branches, num_features)).float()
        )
        
        # Q->dendrite synaptic weights
        self.q_weights = nn.Parameter(
            torch.randn(num_dendritic_branches, num_features) * 0.1
        )
        
        # Proximal V weights
        self.v_weights = nn.Parameter(torch.randn(num_proximal_inputs) * 0.1)
        
    def apply_delays(self, q_spikes: torch.Tensor) -> torch.Tensor:
        """
        Apply conduction delays to Q spikes (K matrix transformation).
        
        Args:
            q_spikes: (batch, time_steps, num_features) binary spike matrix
            
        Returns:
            delayed_spikes: (batch, time_steps, num_branches, num_features)
        """
        batch, T, _ = q_spikes.shape
        device = q_spikes.device
        
        delayed = torch.zeros(
            batch, T, self.num_branches, self.num_features, device=device
        )
        
        for b in range(self.num_branches):
            for f in range(self.num_features):
                delay = int(self.delays[b, f].item())
                if delay > 0 and delay < T:
                    delayed[:, delay:, b, f] = q_spikes[:, :-delay, f]
                elif delay == 0:
                    delayed[:, :, b, f] = q_spikes[:, :, f]
                    
        return delayed
    
    def dendritic_coincidence_detection(
        self, 
        delayed_q: torch.Tensor,
        time_step: int
    ) -> torch.Tensor:
        """
        Detect coincident arrivals on each dendritic branch.
        
        A branch votes "1" if weighted sum of coincident inputs >= threshold.
        
        Args:
            delayed_q: Delayed spike tensor from apply_delays()
            time_step: Current time step
            
        Returns:
            votes: (batch, num_branches) binary votes
        """
        current_spikes = delayed_q[:, time_step, :, :]
        weighted_sum = (current_spikes * self.q_weights).sum(dim=-1)
        votes = (weighted_sum >= self.threshold).float()
        return votes
    
    def proximal_integration(self, v_input: torch.Tensor) -> torch.Tensor:
        """
        Integrate proximal V inputs.
        
        Args:
            v_input: (batch, num_proximal_inputs)
            
        Returns:
            v_contrib: (batch, 1) weighted sum
        """
        return (v_input * self.v_weights).sum(dim=-1, keepdim=True)
    
    def forward(
        self, 
        q_spikes: torch.Tensor,      # (batch, time_steps, num_features)
        v_input: torch.Tensor,       # (batch, num_proximal_inputs)
        inhibition: float = 0.0,
        return_trace: bool = False,
        return_membrane: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the integrator neuron.
        
        Args:
            q_spikes: Spike trains from feature neurons (Q matrix)
            v_input: Proximal input values (V matrix)
            inhibition: Inhibitory input
            return_trace: Return spike train over time
            return_membrane: Return membrane potential trace
            
        Returns:
            Output spikes or membrane potential
        """
        batch, T, _ = q_spikes.shape
        device = q_spikes.device
        
        # Apply delays (K matrix)
        delayed_q = self.apply_delays(q_spikes)
        
        # V contribution (constant over time in basic model)
        v_contrib = self.proximal_integration(v_input)
        
        # Temporal integration
        spike_trace = []
        membrane_trace = []
        membrane = torch.full((batch, 1), self.v_rest, device=device)
        refrac_counter = torch.zeros(batch, 1, device=device)
        
        for t in range(T):
            in_refractory = (refrac_counter > 0).float()
            refrac_counter = torch.clamp(refrac_counter - 1, min=0)
            
            # Dendritic votes
            dendritic_votes = self.dendritic_coincidence_detection(delayed_q, t)
            vote_sum = dendritic_votes.sum(dim=-1, keepdim=True)
            
            # Total excitation
            total_excitation = v_contrib + self.alpha * vote_sum - inhibition
            
            # Leaky integration
            decay = torch.exp(torch.tensor(-1.0 / self.tau, device=device))
            membrane = membrane * decay + total_excitation * (1 - in_refractory)
            
            # Spike generation
            spike = ((membrane >= self.v_threshold) & (in_refractory == 0)).float()
            
            # Reset after spike
            membrane = membrane * (1 - spike) + self.v_rest * spike
            refrac_counter = refrac_counter + spike * self.refractory_period
            
            spike_trace.append(spike)
            membrane_trace.append(membrane.clone())
        
        if return_trace:
            return torch.stack(spike_trace, dim=1).squeeze(-1)
        elif return_membrane:
            return torch.stack(membrane_trace, dim=1).squeeze(-1)
        else:
            return membrane.squeeze(-1)
    
    def configure_for_sequence(self, feature_order: list, delays: list):
        """
        Configure delays to detect a specific temporal sequence.
        
        Args:
            feature_order: List of feature indices in order of firing
            delays: List of delays to align them (should sum to similar arrival)
        
        Example:
            # To detect A→B where A fires 5ms before B:
            neuron.configure_for_sequence([0, 1], [5, 0])
        """
        with torch.no_grad():
            for b in range(min(len(feature_order), self.num_branches)):
                for i, f in enumerate(feature_order):
                    if f < self.num_features and i < len(delays):
                        self.delays[b, f] = delays[i]
                        self.q_weights[b, f] = 1.0


class TransformerCrossAttention(nn.Module):
    """
    Standard Transformer Cross-Attention for comparison.
    
    This shows the structural correspondence with IntegratorNeuron.
    """
    
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn_weights = torch.softmax(scores, dim=-1)  # Softmax: all tokens contribute
        output = torch.matmul(attn_weights, V)
        
        return self.out_proj(output)


# ============================================================================
# Comparison Table Generator
# ============================================================================

def print_comparison_table():
    """Print a comparison table between IntegratorNeuron and Transformer."""
    print("""
┌─────────────────────┬─────────────────────────────────┬────────────────────────────────┐
│ DIMENSION           │ INTEGRATOR NEURON               │ TRANSFORMER ATTENTION          │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Q Matrix            │ Spike trains from feature       │ Query projections              │
│                     │ neurons (binary sequences)      │ (continuous vectors)           │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ K Matrix            │ Conduction delays per branch    │ Key projections                │
│                     │ (temporal offsets)              │ (learned linear transform)     │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ V Matrix            │ Proximal dendritic inputs       │ Value projections              │
│                     │ (weighted sum)                  │ (weighted by attention)        │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Q·K Matching        │ Delayed coincidence detection   │ Dot product                    │
│                     │ (spike_time + delay = arrival)  │ (single-step matrix multiply)  │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Gating              │ HARD threshold                  │ SOFTmax                        │
│                     │ Below threshold → ZERO          │ All tokens contribute          │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Output Integration  │ α·Σ(votes) + Σ(w·V)            │ softmax(Q·K^T/√d)·V           │
│                     │ Binary votes, weighted V        │ Continuous weights             │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Temporal Dynamics   │ Leaky integrate-and-fire        │ Single feedforward step        │
│                     │ Membrane potential evolves      │ No inherent time               │
├─────────────────────┼─────────────────────────────────┼────────────────────────────────┤
│ Sparsity            │ Sparse (only coincident)        │ Dense O(N²)                    │
│                     │ Natural noise filtering         │ Noise accumulates              │
└─────────────────────┴─────────────────────────────────┴────────────────────────────────┘
    """)


if __name__ == "__main__":
    print_comparison_table()
