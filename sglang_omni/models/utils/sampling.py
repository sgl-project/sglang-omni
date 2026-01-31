# SPDX-License-Identifier: Apache-2.0
"""Common sampling utilities for token generation."""

from __future__ import annotations

import torch


def sample_logits(
    logits: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    suppress_tokens: list[int] | None = None,
    repetition_penalty: float = 1.0,
    generated_tokens: list[int] | None = None,
) -> torch.Tensor:
    """Sample next token from logits with various decoding strategies.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        do_sample: Whether to sample (True) or use greedy decoding (False)
        temperature: Temperature for sampling (higher = more random)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Keep tokens with cumulative probability <= top_p (1.0 = disabled)
        suppress_tokens: List of token IDs to suppress (set to -inf)
        repetition_penalty: Penalty for repeating tokens (1.0 = disabled)
        generated_tokens: Previously generated tokens for repetition penalty
        
    Returns:
        Sampled token IDs of shape (batch_size,)
    """
    logits = logits.clone()
    
    if repetition_penalty != 1.0 and generated_tokens:
        for token_id in set(generated_tokens):
            if token_id < logits.shape[-1]:
                if logits[:, token_id] > 0:
                    logits[:, token_id] = logits[:, token_id] / repetition_penalty
                else:
                    logits[:, token_id] = logits[:, token_id] * repetition_penalty
    
    if suppress_tokens:
        logits[:, suppress_tokens] = float("-inf")
    
    if not do_sample or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    
    logits = logits / max(temperature, 1e-6)
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, topk_idx, topk_vals)
    
    if 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        mask = cumulative > top_p
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    
    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(probs_sum > 0, probs / probs_sum, probs)
    
    if torch.any(probs_sum == 0):
        return torch.argmax(logits, dim=-1)
    
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_k_top_p(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Simple top-k and top-p sampling.
    
    This is a simplified version of sample_logits for cases where
    only top_k and top_p filtering is needed.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        top_k: Keep only top k tokens (None = disabled)
        top_p: Keep tokens with cumulative probability <= top_p (None = disabled)
        
    Returns:
        Sampled token IDs of shape (batch_size,)
    """
    probs = torch.softmax(logits.float(), dim=-1)
    
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, topk_idx, topk_vals)
    
    if top_p is not None and 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        mask = cumulative > top_p
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    
    probs_sum = probs.sum(dim=-1, keepdim=True)
    if torch.any(probs_sum == 0):
        return torch.argmax(logits, dim=-1)
    
    probs = probs / probs_sum
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
