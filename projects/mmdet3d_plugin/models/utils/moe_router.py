import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def soft_topk(logits, k, temperature=1.0):
    """
    Differentiable Top-K sampling using Gumbel Softmax trick.
    Args:
        logits: [B, Nq, Num_Levels]
        k: int, number of top experts to select
        temperature: float, controls the sharpness of the distribution
    Returns:
        soft_topk_probs: [B, Nq, Num_Levels] - Probabilities (differentiable)
        topk_indices: [B, Nq, K] - Indices of the top k experts
    """
    # If selecting all levels, just return softmax
    if k == logits.shape[-1]: 
        return F.softmax(logits / temperature, dim=-1), None

    # 1. Add Gumbel noise for exploration during training
    noise = sample_gumbel(logits.shape, device=logits.device)
    perturbed_logits = logits + noise
    
    # 2. Compute Softmax Probabilities
    soft_probs = F.softmax(perturbed_logits / temperature, dim=-1)
    
    # 3. Find the threshold for Top-K
    # topk_vals: [B, Nq, k]
    topk_vals, topk_indices = torch.topk(perturbed_logits, k, dim=-1)
    
    # 4. Generate a differentiable mask
    # Values larger than the k-th largest value get a mask close to 1
    threshold = topk_vals[..., -1].unsqueeze(-1) 
    mask = torch.sigmoid((perturbed_logits - threshold) / temperature)
    
    # 5. Apply mask and re-normalize probabilities
    # We want the sum of the selected k probabilities to be 1
    soft_topk_probs = soft_probs * mask
    soft_topk_probs = soft_topk_probs / (soft_topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
    
    return soft_topk_probs, topk_indices

class StreamMoERouter(nn.Module):
    def __init__(self, embed_dims, num_levels, top_k=2):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels 
        self.top_k = top_k
        
        # Router Network: Maps query embedding to logits for each level
        self.gate = nn.Sequential(
            nn.Linear(embed_dims, 256),
            nn.ReLU(),
            nn.Linear(256, num_levels)
        )

    def forward(self, query_embeds):
        """
        Args:
            query_embeds: (B, Nq, C) 
        Returns:
            routing_weights: (B, Nq, Num_Levels) - Weights for feature fusion
            topk_indices: (B, Nq, K) - Indices of selected levels (for analysis or hard gating)
        """
        # 1. Calculate routing logits
        logits = self.gate(query_embeds) 
        
        # 2. Top-K Gating
        if self.training:
            # Training: Use Soft Top-K with Gumbel noise for differentiability & exploration
            routing_weights, topk_indices = soft_topk(logits, self.top_k, temperature=1.0)
        else:
            # Inference: Use Hard Top-K for deterministic results
            topk_vals, topk_indices = torch.topk(logits, self.top_k, dim=-1)
            
            # Create a hard mask (1 for selected, 0 for others)
            mask = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
            
            # Apply softmax only on selected indices
            routing_weights = F.softmax(logits, dim=-1) * mask
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)

        return routing_weights, topk_indices