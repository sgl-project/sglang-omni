# SPDX-License-Identifier: Apache-2.0
"""Debug Qwen3-Omni audio preprocessing + encoder outputs."""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import torch

from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.frontends.audio import (
    build_audio_mm_inputs,
    ensure_audio_list,
)
from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    Qwen3OmniAudioEncoder,
)
from sglang_omni.models.qwen3_omni.components.torch_audio_encoder import (
    Qwen3OmniTorchAudioEncoder,
)
from sglang_omni.models.weight_loader import resolve_model_path


def _describe_tensor(name: str, value: torch.Tensor) -> None:
    with torch.no_grad():
        stats = {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        if value.numel():
            stats["min"] = float(value.min().item())
            stats["max"] = float(value.max().item())
            if value.is_floating_point() or value.is_complex():
                stats["mean"] = float(value.mean().item())
                stats["std"] = float(value.std().item())
            else:
                stats["mean"] = None
                stats["std"] = None
        else:
            stats["min"] = math.nan
            stats["max"] = math.nan
            stats["mean"] = math.nan
            stats["std"] = math.nan
    print(f"{name}: {stats}")


def _describe_array(name: str, value: np.ndarray) -> None:
    stats = {
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()) if value.size else math.nan,
        "max": float(value.max()) if value.size else math.nan,
        "mean": float(value.mean()) if value.size else math.nan,
        "std": float(value.std()) if value.size else math.nan,
    }
    print(f"{name}: {stats}")


def _run_encoder(
    name: str,
    encoder: torch.nn.Module,
    mm_inputs: dict[str, Any],
) -> torch.Tensor:
    encoder.eval()
    with torch.inference_mode():
        out = encoder(
            input_features=mm_inputs["input_features"],
            feature_attention_mask=mm_inputs.get("feature_attention_mask"),
            audio_feature_lengths=mm_inputs.get("audio_feature_lengths"),
        )
    audio_embeds = out["audio_embeds"]
    _describe_tensor(f"{name}.audio_embeds", audio_embeds)
    _describe_tensor(f"{name}.audio_feature_lengths", out["audio_feature_lengths"])
    _describe_tensor(f"{name}.audio_output_lengths", out["audio_output_lengths"])
    return audio_embeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        "--model-id",
        dest="model_path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Local model path or Hugging Face model id",
    )
    parser.add_argument("--audio-path", type=str, required=True)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--backend",
        type=str,
        default="both",
        choices=["hf", "torch", "both"],
        help="Which audio encoder to run",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(
        args.model_path, local_files_only=args.local_files_only
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    audios = ensure_audio_list([args.audio_path], target_sr=args.audio_target_sr)
    if not audios:
        raise SystemExit("No audio inputs after ensure_audio_list.")
    if isinstance(audios[0], np.ndarray):
        _describe_array("audio[0]", audios[0])

    prompt_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Describe the audio."}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    hf_inputs = processor(
        text=prompt_text,
        audio=audios,
        add_special_tokens=False,
        return_tensors="pt",
    )
    mm_audio = build_audio_mm_inputs(hf_inputs)

    _describe_tensor("input_features", mm_audio["input_features"])
    if mm_audio.get("feature_attention_mask") is not None:
        _describe_tensor("feature_attention_mask", mm_audio["feature_attention_mask"])
    if mm_audio.get("audio_feature_lengths") is not None:
        _describe_tensor("audio_feature_lengths", mm_audio["audio_feature_lengths"])

    if args.backend in {"hf", "both"}:
        hf_encoder = Qwen3OmniAudioEncoder(
            model_path=str(model_path),
            device=args.device,
            dtype=args.dtype,
        )
        hf_embeds = _run_encoder("hf", hf_encoder, mm_audio)
    else:
        hf_embeds = None

    if args.backend in {"torch", "both"}:
        torch_encoder = Qwen3OmniTorchAudioEncoder(
            model_path=str(model_path),
            device=args.device,
            dtype=args.dtype,
        )
        torch_embeds = _run_encoder("torch", torch_encoder, mm_audio)
    else:
        torch_embeds = None

    if hf_embeds is not None and torch_embeds is not None:
        if hf_embeds.shape != torch_embeds.shape:
            print(
                "Encoder output shapes differ:",
                tuple(hf_embeds.shape),
                tuple(torch_embeds.shape),
            )
        else:
            diff = (hf_embeds - torch_embeds).float()
            _describe_tensor("hf_minus_torch", diff)
            cos = torch.nn.functional.cosine_similarity(
                hf_embeds.flatten().float(), torch_embeds.flatten().float(), dim=0
            )
            print(f"hf_vs_torch_cosine: {float(cos.item()):.6f}")


if __name__ == "__main__":
    main()
