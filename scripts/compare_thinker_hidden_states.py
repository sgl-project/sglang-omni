# SPDX-License-Identifier: Apache-2.0
"""Compare HF vs torch-native thinker hidden states (before lm_head)."""

from __future__ import annotations

import argparse
import gc
import time

import torch
from transformers import AutoTokenizer

from sglang_omni.models.weight_loader import resolve_model_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=str,
                    default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--prompt", type=str,
                    default="Hello, how are you today?")
    p.add_argument("--max-seq-len", type=int, default=128,
                    help="Pad/truncate to this length for reproducibility.")
    return p.parse_args()


@torch.no_grad()
def run_hf_thinker(
    model_path: str, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    device: str, dtype: str,
) -> torch.Tensor:
    """Load HF thinker, run forward, return last hidden state on CPU."""
    from sglang_omni.models.qwen3_omni.components.thinker import (
        Qwen3OmniSplitThinker,
    )

    print("[HF] Loading model ...")
    t0 = time.perf_counter()
    model = Qwen3OmniSplitThinker(model_path, device=device, dtype=dtype)
    print(f"[HF] Loaded in {time.perf_counter() - t0:.1f}s")

    print("[HF] Running forward ...")
    t0 = time.perf_counter()
    out = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        output_hidden_states=True,
    )
    # HF returns CausalLMOutputWithPast; hidden_states[-1] is post-norm
    hf_hidden = out.hidden_states[-1].cpu()
    print(f"[HF] Forward done in {time.perf_counter() - t0:.1f}s, "
          f"shape={hf_hidden.shape}, dtype={hf_hidden.dtype}")

    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return hf_hidden


@torch.no_grad()
def run_torch_thinker(
    model_path: str, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    device: str, dtype: str,
) -> torch.Tensor:
    """Load torch-native thinker, run forward, return last hidden state on CPU."""
    from sglang_omni.models.qwen3_omni.components.torch_thinker import (
        Qwen3OmniTorchThinker,
    )

    print("[Torch] Loading model ...")
    t0 = time.perf_counter()
    model = Qwen3OmniTorchThinker(model_path, device=device, dtype=dtype)
    print(f"[Torch] Loaded in {time.perf_counter() - t0:.1f}s")

    print("[Torch] Running forward ...")
    t0 = time.perf_counter()
    out = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        output_hidden_states=True,
    )
    torch_hidden = out.last_hidden_state.cpu()
    print(f"[Torch] Forward done in {time.perf_counter() - t0:.1f}s, "
          f"shape={torch_hidden.shape}, dtype={torch_hidden.dtype}")

    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return torch_hidden


def compare(hf: torch.Tensor, torch_native: torch.Tensor) -> None:
    """Print numerical comparison of two hidden-state tensors."""
    hf_f = hf.float()
    tn_f = torch_native.float()
    diff = (hf_f - tn_f).abs()
    print("\n===== Comparison (last hidden state, before lm_head) =====")
    print(f"  Shape : {hf.shape}")
    print(f"  Max Δ : {diff.max().item():.6e}")
    print(f"  Mean Δ: {diff.mean().item():.6e}")
    print(f"  Rel Δ : {(diff / (hf_f.abs() + 1e-8)).mean().item():.6e}")
    cos = torch.nn.functional.cosine_similarity(
        hf_f.reshape(1, -1), tn_f.reshape(1, -1)
    )
    print(f"  Cosine: {cos.item():.8f}")

    # Per-token cosine similarity (averaged over seq)
    per_tok = torch.nn.functional.cosine_similarity(
        hf_f.squeeze(0), tn_f.squeeze(0), dim=-1
    )
    print(f"  Per-token cosine: min={per_tok.min():.8f}  "
          f"mean={per_tok.mean():.8f}  max={per_tok.max():.8f}")

    # Also compare logit-space by projecting with a random vector
    # to check if the difference would matter for sampling
    rng = torch.manual_seed(42)
    probe = torch.randn(hf_f.shape[-1], 10)
    hf_proj = hf_f.squeeze(0) @ probe
    tn_proj = tn_f.squeeze(0) @ probe
    proj_diff = (hf_proj - tn_proj).abs()
    print(f"  Random-proj Δ: max={proj_diff.max():.6e}  "
          f"mean={proj_diff.mean():.6e}")


def main() -> None:
    args = parse_args()
    model_path = str(resolve_model_path(args.model_path))

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokens = tokenizer(
        args.prompt, return_tensors="pt",
        padding="max_length", max_length=args.max_seq_len, truncation=True,
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    print(f"Input: {args.prompt!r}")
    print(f"Tokens: {input_ids.shape}, non-pad: {attention_mask.sum().item()}")

    # Run HF first, then torch (sequential to fit in GPU memory)
    hf_hidden = run_hf_thinker(
        model_path, input_ids, attention_mask, args.device, args.dtype)
    torch_hidden = run_torch_thinker(
        model_path, input_ids, attention_mask, args.device, args.dtype)

    compare(hf_hidden, torch_hidden)


if __name__ == "__main__":
    main()
