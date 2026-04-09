import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 🌌 GENESIS-v1: THE MANIFOLD AI ARCHITECTURE
# Author: Soumya Ranjan Das 16 YEAR OLD
# Complexity: O(n) Time | Constant Memory Footprint
# License: Apache 2.0 / 
# =============================================================================

class GenesisV1(nn.Module):
    """
    Genesis v1: An Attention-Free Gated Manifold.
    Implements a latent-graph recurrent manifold of independent nodes 
    to achieve algorithmic reasoning without the O(n^2) quadratic bottleneck.
    """
    def __init__(self, vocab_size, d_model=128, num_nodes=48):
        super(GenesisV1, self).__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.vocab_size = vocab_size
        
        # Latent Projection
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # The Manifold: 48 Independent Specialist Nodes
        # Acts as the persistent working memory state
        self.manifold_nodes = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)
        
        # Gated Update Mechanism (The Consensus Synapses)
        # Filters information flow and maintains logical consistency
        self.gate = nn.Linear(d_model * 2, d_model)
        self.update_layer = nn.Linear(d_model * 2, d_model)
        
        # Readout Head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, hidden=None):
        """
        Forward pass for Genesis v1.
        Input: (Batch_Size, Sequence_Length)
        Output: (Batch_Size, Sequence_Length, Vocab_Size)
        """
        batch_size, seq_len = idx.shape
        x = self.embedding(idx)
        
        # Initialize or fetch the manifold state
        if hidden is None:
            node_states = self.manifold_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            node_states = hidden
        
        logits_list = []
        
        # The Linear O(n) Manifold Scan
        for t in range(seq_len):
            # Token feature broadcasting across the manifold
            current_features = x[:, t, :].unsqueeze(1).expand(-1, self.num_nodes, -1)
            
            # Manifold-Input interaction
            combined_state = torch.cat([current_features, node_states], dim=-1)
            
            # Gated Update Logic (Consensus Formation)
            update_gate = torch.sigmoid(self.gate(combined_state))
            candidate_state = torch.tanh(self.update_layer(combined_state))
            
            # State Evolution
            node_states = update_gate * candidate_state + (1 - update_gate) * node_states
            
            # Consensus Readout (Mean-pool of nodes)
            consensus = node_states.mean(dim=1)
            prediction = self.head(self.norm(consensus))
            logits_list.append(prediction)
            
        return torch.stack(logits_list, dim=1)

    def get_manifold_state(self, batch_size):
        """Returns the initial state of the manifold."""
        return self.manifold_nodes.unsqueeze(0).expand(batch_size, -1, -1)
