# SPDX-License-Identifier: Apache-2.0
"""Compare HF vs torch-native thinker hidden states with multimodal inputs.

Runs shared encoders (image + audio) once, then feeds identical encoder
outputs to both thinker implementations and compares hidden states
layer-by-layer.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from sglang_omni.models.weight_loader import resolve_model_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=str,
                    default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--prompt", type=str,
                    default="Describe both the image and the audio content in detail.")
    p.add_argument("--image-path", type=str, default="tests/data/cars.jpg")
    p.add_argument("--audio-path", type=str, default="tests/data/cough.wav")
    p.add_argument("--layers", type=str, default="0,1,2,23,24,47,48",
                    help="Comma-separated layer indices to compare "
                         "(0=embeddings, N=post-norm)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compare two tensors and print metrics. Returns dict of metrics."""
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    cos = F.cosine_similarity(a_f.reshape(1, -1), b_f.reshape(1, -1))
    # Per-token cosine: flatten to [seq, hidden]
    a_sq = a_f.view(-1, a_f.shape[-1])
    b_sq = b_f.view(-1, b_f.shape[-1])
    per_tok = F.cosine_similarity(a_sq, b_sq, dim=-1)
    metrics = {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "cosine": cos.item(),
        "per_tok_min": per_tok.min().item(),
        "per_tok_mean": per_tok.mean().item(),
        "per_tok_max": per_tok.max().item(),
    }
    print(f"  {name}:")
    print(f"    max_diff={metrics['max_diff']:.4e}  "
          f"mean_diff={metrics['mean_diff']:.4e}")
    print(f"    cosine={metrics['cosine']:.8f}")
    print(f"    per_token_cos: min={metrics['per_tok_min']:.6f} "
          f"mean={metrics['per_tok_mean']:.6f} "
          f"max={metrics['per_tok_max']:.6f}")
    return metrics


# ---------------------------------------------------------------------------
# Step 1: Preprocess with HF processor
# ---------------------------------------------------------------------------

@torch.no_grad()
def preprocess(
    model_path: str, prompt: str,
    image_path: str | None, audio_path: str | None,
) -> dict:
    """Tokenize multimodal input via the HF processor."""
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        Qwen3OmniMoeProcessor,
    )
    from sglang_omni.frontends import ensure_audio_list, ensure_image_list

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )

    # Build structured multimodal message
    content: list[dict] = []
    images_raw = [image_path] if image_path else []
    audios_raw = [audio_path] if audio_path else []
    images = ensure_image_list(images_raw)
    audios = ensure_audio_list(audios_raw, target_sr=16000)

    for _ in images:
        content.append({"type": "image"})
    for _ in audios:
        content.append({"type": "audio"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    hf_inputs = processor(
        text=prompt_text,
        images=images or None,
        audio=audios or None,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = hf_inputs["input_ids"]
    att_mask = hf_inputs.get("attention_mask")
    if att_mask is None:
        att_mask = torch.ones_like(input_ids)

    print(f"Prompt: {prompt!r}")
    print(f"Input IDs: {input_ids.shape} "
          f"(non-pad: {att_mask.sum().item()})")
    if "pixel_values" in hf_inputs:
        print(f"Pixel values: {hf_inputs['pixel_values'].shape}")
    if "image_grid_thw" in hf_inputs:
        print(f"Image grid THW: {hf_inputs['image_grid_thw']}")
    if "input_features" in hf_inputs:
        print(f"Input features: {hf_inputs['input_features'].shape}")
    if "feature_attention_mask" in hf_inputs:
        fam = hf_inputs["feature_attention_mask"]
        print(f"Feature attention mask: {fam.shape} "
              f"(valid frames: {fam.sum().item()})")

    return dict(hf_inputs)


# ---------------------------------------------------------------------------
# Step 2: Run shared encoders (torch-native)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_encoders(
    model_path: str, hf_inputs: dict, device: str, dtype: str,
) -> dict:
    """Run torch-native image & audio encoders, return embeddings on CPU."""
    results: dict = {}

    pixel_values = hf_inputs.get("pixel_values")
    image_grid_thw = hf_inputs.get("image_grid_thw")
    if pixel_values is not None:
        from sglang_omni.models.qwen3_omni.components.torch_image_encoder import (
            Qwen3OmniTorchImageEncoder,
        )
        print("\n[Encoder] Loading image encoder ...")
        t0 = time.perf_counter()
        enc = Qwen3OmniTorchImageEncoder(model_path, device=device, dtype=dtype)
        print(f"[Encoder] Image encoder loaded ({time.perf_counter() - t0:.1f}s)")
        out = enc(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        results["image_embeds"] = out["image_embeds"].cpu()
        results["deepstack_visual_embeds"] = [
            e.cpu() for e in out["deepstack_visual_embeds"]
        ]
        results["image_grid_thw"] = out["image_grid_thw"].cpu()
        print(f"  image_embeds: {results['image_embeds'].shape}")
        print(f"  deepstack layers: {len(results['deepstack_visual_embeds'])}")
        del enc, out
        gc.collect()
        torch.cuda.empty_cache()

    input_features = hf_inputs.get("input_features")
    feature_attention_mask = hf_inputs.get("feature_attention_mask")
    if input_features is not None:
        from sglang_omni.models.qwen3_omni.components.torch_audio_encoder import (
            Qwen3OmniTorchAudioEncoder,
        )
        print("[Encoder] Loading audio encoder ...")
        t0 = time.perf_counter()
        enc = Qwen3OmniTorchAudioEncoder(model_path, device=device, dtype=dtype)
        print(f"[Encoder] Audio encoder loaded ({time.perf_counter() - t0:.1f}s)")
        out = enc(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        results["audio_embeds"] = out["audio_embeds"].cpu()
        results["audio_feature_lengths"] = out["audio_feature_lengths"].cpu()
        print(f"  audio_embeds: {results['audio_embeds'].shape}")
        print(f"  audio_feature_lengths: {results['audio_feature_lengths']}")
        del enc, out
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Step 3 & 4: Run thinkers sequentially
# ---------------------------------------------------------------------------

def _build_thinker_kwargs(
    hf_inputs: dict, encoder_out: dict,
) -> dict:
    """Build the common kwargs dict for both thinkers."""
    kwargs: dict = {"output_hidden_states": True}
    if "image_embeds" in encoder_out:
        kwargs["image_embeds"] = encoder_out["image_embeds"]
        kwargs["deepstack_visual_embeds"] = encoder_out["deepstack_visual_embeds"]
    if "image_grid_thw" in encoder_out:
        kwargs["image_grid_thw"] = encoder_out["image_grid_thw"]
    if "audio_embeds" in encoder_out:
        kwargs["audio_embeds"] = encoder_out["audio_embeds"]
    # Pass feature_attention_mask for RoPE computation in both thinkers
    fam = hf_inputs.get("feature_attention_mask")
    if fam is not None:
        kwargs["feature_attention_mask"] = fam
    return kwargs


@torch.no_grad()
def run_hf_thinker(
    model_path: str, hf_inputs: dict, encoder_out: dict,
    device: str, dtype: str,
) -> tuple[torch.Tensor, ...]:
    """Run HF thinker, return hidden states on CPU."""
    from sglang_omni.models.qwen3_omni.components.thinker import (
        Qwen3OmniSplitThinker,
    )

    print("\n[HF] Loading thinker ...")
    t0 = time.perf_counter()
    model = Qwen3OmniSplitThinker(model_path, device=device, dtype=dtype)
    print(f"[HF] Loaded ({time.perf_counter() - t0:.1f}s)")

    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs.get("attention_mask", torch.ones_like(input_ids))
    kwargs = _build_thinker_kwargs(hf_inputs, encoder_out)

    print("[HF] Running forward ...")
    t0 = time.perf_counter()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )
    print(f"[HF] Forward done ({time.perf_counter() - t0:.1f}s)")

    hidden_states = tuple(h.cpu() for h in out.hidden_states)
    print(f"[HF] {len(hidden_states)} hidden states, "
          f"last shape={hidden_states[-1].shape}")

    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return hidden_states


@torch.no_grad()
def run_torch_thinker(
    model_path: str, hf_inputs: dict, encoder_out: dict,
    device: str, dtype: str,
) -> tuple[torch.Tensor, ...]:
    """Run torch-native thinker, return hidden states on CPU."""
    from sglang_omni.models.qwen3_omni.components.torch_thinker import (
        Qwen3OmniTorchThinker,
    )

    print("\n[Torch] Loading thinker ...")
    t0 = time.perf_counter()
    model = Qwen3OmniTorchThinker(model_path, device=device, dtype=dtype)
    print(f"[Torch] Loaded ({time.perf_counter() - t0:.1f}s)")

    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs.get("attention_mask", torch.ones_like(input_ids))
    kwargs = _build_thinker_kwargs(hf_inputs, encoder_out)

    print("[Torch] Running forward ...")
    t0 = time.perf_counter()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )
    print(f"[Torch] Forward done ({time.perf_counter() - t0:.1f}s)")

    hidden_states = tuple(h.cpu() for h in out.hidden_states)
    print(f"[Torch] {len(hidden_states)} hidden states, "
          f"last shape={hidden_states[-1].shape}")

    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return hidden_states


# ---------------------------------------------------------------------------
# Step 5: Compare
# ---------------------------------------------------------------------------

def print_summary(
    layer_metrics: dict[int, dict],
    n_layers: int,
    has_image: bool,
    has_audio: bool,
) -> None:
    """Print a summary table."""
    modalities = []
    if has_image:
        modalities.append("image")
    if has_audio:
        modalities.append("audio")
    mod_str = " + ".join(modalities) if modalities else "text-only"

    print(f"\n{'=' * 60}")
    print(f"=== Summary: HF vs Torch-Native (multimodal: {mod_str}) ===")
    print(f"{'=' * 60}")
    print(f"{'Layer':<20} {'Cosine':>12} {'Max Diff':>12} {'Per-Tok Min':>12}")
    print(f"{'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12}")
    for idx in sorted(layer_metrics.keys()):
        m = layer_metrics[idx]
        label = f"hs[{idx}]"
        if idx == 0:
            label += " (embed)"
        elif idx == n_layers - 1:
            label += " (post-norm)"
        print(f"{label:<20} {m['cosine']:>12.6f} "
              f"{m['max_diff']:>12.4e} {m['per_tok_min']:>12.6f}")
    print()

    # Interpretation
    last_idx = max(layer_metrics.keys())
    last_cos = layer_metrics[last_idx]["cosine"]
    if last_cos > 0.9999:
        verdict = "Negligible difference (bf16 rounding)"
    elif last_cos > 0.99:
        verdict = "Minor implementation difference (acceptable)"
    elif last_cos > 0.9:
        verdict = "Significant difference — investigate"
    else:
        verdict = ("Large divergence — expected for MoE with bf16 "
                    "(expert routing sensitivity)")
    print(f"Diagnosis: {verdict}")


def main() -> None:
    args = parse_args()
    model_path = str(resolve_model_path(args.model_path))
    layer_indices = [int(x) for x in args.layers.split(",")]

    # 1. Preprocess
    print("=" * 60)
    print("Step 1: Preprocessing with HF processor")
    print("=" * 60)
    hf_inputs = preprocess(
        model_path, args.prompt, args.image_path, args.audio_path,
    )

    # 2. Shared encoders
    print(f"\n{'=' * 60}")
    print("Step 2: Running shared encoders (torch-native)")
    print("=" * 60)
    encoder_out = run_encoders(model_path, hf_inputs, args.device, args.dtype)

    # 3. HF thinker
    print(f"\n{'=' * 60}")
    print("Step 3: HF thinker")
    print("=" * 60)
    hf_hidden = run_hf_thinker(
        model_path, hf_inputs, encoder_out, args.device, args.dtype,
    )

    # 4. Torch-native thinker
    print(f"\n{'=' * 60}")
    print("Step 4: Torch-native thinker")
    print("=" * 60)
    torch_hidden = run_torch_thinker(
        model_path, hf_inputs, encoder_out, args.device, args.dtype,
    )

    # 5. Layer-by-layer comparison
    print(f"\n{'=' * 60}")
    print("Step 5: Layer-by-layer comparison")
    print("=" * 60)
    n_hf = len(hf_hidden)
    n_torch = len(torch_hidden)
    print(f"HF hidden states: {n_hf}, Torch hidden states: {n_torch}")
    if n_hf != n_torch:
        print(f"WARNING: different number of hidden states!")

    n_compare = min(n_hf, n_torch)
    layer_metrics: dict[int, dict] = {}
    for idx in layer_indices:
        if idx >= n_compare:
            print(f"  Skipping layer {idx} (only {n_compare} available)")
            continue
        label = f"hs[{idx}]"
        if idx == 0:
            label += " (embeddings)"
        elif idx == n_compare - 1:
            label += " (post-norm)"
        else:
            label += f" (after layer {idx - 1})"
        m = compare(label, hf_hidden[idx], torch_hidden[idx])
        layer_metrics[idx] = m
        print()

    has_image = "image_embeds" in encoder_out
    has_audio = "audio_embeds" in encoder_out
    print_summary(layer_metrics, n_compare, has_image, has_audio)

    # Analyze worst tokens in the final (post-norm) layer
    last_idx = n_compare - 1
    if last_idx in layer_metrics:
        _analyze_worst_tokens(
            hf_hidden[last_idx], torch_hidden[last_idx],
            hf_inputs, n_worst=20,
        )


def _analyze_worst_tokens(
    hf_hs: torch.Tensor, torch_hs: torch.Tensor,
    hf_inputs: dict, n_worst: int = 20,
) -> None:
    """Show which token positions diverge most and what type they are."""
    a = hf_hs.float().view(-1, hf_hs.shape[-1])
    b = torch_hs.float().view(-1, torch_hs.shape[-1])
    per_tok = F.cosine_similarity(a, b, dim=-1)

    input_ids = hf_inputs["input_ids"].squeeze(0)

    # Identify token types by scanning input_ids for known special IDs
    # Qwen3-Omni uses specific token IDs for image/audio placeholders
    # We detect them by looking at high-frequency repeated tokens
    unique, counts = input_ids.unique(return_counts=True)
    # Image tokens: the most repeated (6042 tokens)
    # Audio tokens: ~38 tokens
    sorted_by_count = counts.argsort(descending=True)
    token_type = ["text"] * len(input_ids)

    # Heuristic: tokens repeated > 30 times are likely placeholders
    placeholder_ids: dict[int, str] = {}
    for rank in range(min(5, len(sorted_by_count))):
        tid = unique[sorted_by_count[rank]].item()
        cnt = counts[sorted_by_count[rank]].item()
        if cnt > 30:
            # Larger count = image, smaller = audio
            if not placeholder_ids:
                placeholder_ids[tid] = f"image({cnt}x)"
            elif cnt > 100:
                placeholder_ids[tid] = f"image({cnt}x)"
            else:
                placeholder_ids[tid] = f"audio({cnt}x)"

    for i, tid in enumerate(input_ids.tolist()):
        if tid in placeholder_ids:
            token_type[i] = placeholder_ids[tid].split("(")[0]

    # Find worst tokens
    worst_indices = per_tok.argsort()[:n_worst]

    print(f"\n{'=' * 60}")
    print(f"Worst {n_worst} tokens at post-norm layer:")
    print(f"{'=' * 60}")
    print(f"{'Pos':>6}  {'Type':<8}  {'Cosine':>10}  {'TokenID':>8}")
    print(f"{'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}")

    type_counts: dict[str, int] = {}
    for idx in worst_indices:
        pos = idx.item()
        cos_val = per_tok[pos].item()
        tid = input_ids[pos].item()
        ttype = token_type[pos]
        type_counts[ttype] = type_counts.get(ttype, 0) + 1
        print(f"{pos:>6}  {ttype:<8}  {cos_val:>10.6f}  {tid:>8}")

    # Count by type for all negative-cosine tokens
    neg_mask = per_tok < 0
    n_neg = neg_mask.sum().item()
    print(f"\nTokens with negative cosine: {n_neg}/{len(per_tok)}")
    if n_neg > 0:
        neg_type_counts: dict[str, int] = {}
        for i in neg_mask.nonzero().squeeze(-1).tolist():
            t = token_type[i]
            neg_type_counts[t] = neg_type_counts.get(t, 0) + 1
        for t, c in sorted(neg_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

    # Summary: average cosine by token type
    print(f"\nPer-type cosine at post-norm:")
    for ttype in ["image", "audio", "text"]:
        mask = torch.tensor([1 if token_type[i] == ttype else 0
                             for i in range(len(token_type))], dtype=torch.bool)
        if mask.any():
            type_cos = per_tok[mask]
            print(f"  {ttype:>6}: n={mask.sum():>5}  "
                  f"min={type_cos.min():.6f}  "
                  f"mean={type_cos.mean():.6f}  "
                  f"max={type_cos.max():.6f}")


if __name__ == "__main__":
    main()
