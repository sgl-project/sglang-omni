# SPDX-License-Identifier: Apache-2.0
"""Generate migration-reference audio using a frozen Python MLX source checkout."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import wave
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
from provision import sha256

import sglang_omni
from sglang_omni.models.auk.hf_config import AuKDitConfig, AuKVAEConfig
from sglang_omni.models.auk.mlx.conditioning import AuKMlxConditionEncoder
from sglang_omni.models.auk.mlx.dit import AuKDit
from sglang_omni.models.auk.mlx.flow_matching import (
    AuKFlowMatching,
    AuKSampleItem,
    fuse_hidden_states,
    request_key,
)
from sglang_omni.models.auk.mlx.vae import BigVGANFlowVAE
from sglang_omni.models.auk.reference_encode import build_messages
from sglang_omni.utils.audio import load_audio


def cpu_handoff(tensor: mx.array) -> mx.array:
    """Preserve the frozen pipeline's FP32 CPU tensor transport."""
    return mx.array(
        np.array(tensor.astype(mx.float32) if tensor.dtype == mx.bfloat16 else tensor)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    arguments = parser.parse_args()
    source = json.loads(
        (Path(__file__).parent / "auk_reference/source.json").read_text()
    )
    source_root = Path(sglang_omni.__file__).resolve().parent.parent
    for filename, digest in source["files"].items():
        if sha256(source_root / filename) != digest:
            raise ValueError(f"Reference source differs: {filename}")
        else:
            pass
    if version("mlx") != "0.32.3" or os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError(
            "Reference generation requires mlx==0.32.3 and MLX_ENABLE_TF32=0"
        )
    elif arguments.output_directory.exists():
        raise FileExistsError(arguments.output_directory)
    else:
        arguments.output_directory.mkdir(parents=True)
    requests = [
        json.loads(line)
        for line in arguments.requests.read_text().splitlines()
        if line.strip()
    ]
    request_ids = [request["id"] for request in requests]
    if len(set(request_ids)) != len(request_ids) or any(
        re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id) is None
        for request_id in request_ids
    ):
        raise ValueError(
            "Requests require unique IDs containing letters, digits, _ or -"
        )
    else:
        pass
    for request in requests:
        if (
            "reference_wav" in request
            and sha256(Path(request["reference_wav"])) != request["reference_sha256"]
        ):
            raise ValueError(f"Reference audio differs: {request['id']}")
        else:
            pass
    mx.set_default_device(mx.gpu)
    dtype = {"bfloat16": mx.bfloat16, "float32": mx.float32}[arguments.dtype]
    directory = arguments.model_path
    runtime = json.loads((directory / "config.json").read_text())
    encoder = AuKMlxConditionEncoder(str(directory / "conditioner"), dtype=dtype)
    vae = BigVGANFlowVAE(
        AuKVAEConfig.from_dict(json.loads((directory / "vae/config.json").read_text()))
    )
    vae.load_weights(str(directory / "vae/model.safetensors"), strict=True)
    architecture = asdict(
        AuKDitConfig.from_dict(
            json.loads((directory / "dit/config.json").read_text())["arch"]
        )
    )
    architecture.pop("depth")
    architecture.pop("dropout")
    architecture["latent_dim"] = runtime["latent_dim"]
    weights = mx.load(str(directory / "dit/model.safetensors"))
    flow = AuKFlowMatching(
        AuKDit(**architecture), num_llm_layers=weights["layer_weights"].size
    )
    flow.load_weights(list(weights.items()), strict=True)
    rotary = flow.transformer.rotary_embed.inv_freq.astype(mx.float32)
    flow.transformer.set_dtype(dtype)
    flow.transformer.rotary_embed.inv_freq = rotary
    vae.eval()
    flow.eval()
    mx.eval(vae.parameters(), flow.parameters())
    weights.clear()
    records = []
    for request in requests:
        reference_latents, reference_length, qwen_audio = None, 0, None
        if "reference_wav" in request:
            waveform = load_audio(
                request["reference_wav"],
                source_name="AuK",
                target_sample_rate=runtime["sample_rate"],
            )
            qwen_audio, _ = librosa.load(request["reference_wav"], sr=16000, mono=True)
            latent, lengths = vae.encode(
                mx.array(waveform.reshape(1, 1, -1)),
                mx.array([waveform.size // vae.hop_size * vae.hop_size]),
                key=request_key(request["seed"], stream="reference"),
            )
            reference_latents, reference_length = cpu_handoff(latent[0]), int(
                lengths[0].item()
            )
        else:
            pass
        hidden, mask = encoder.encode_batch(
            [build_messages(request["instruction"], qwen_audio is not None)],
            [qwen_audio],
        )[0]
        conditioning = cpu_handoff(
            fuse_hidden_states(hidden[None], flow.layer_weights, flow.layer_scale)[0]
        )
        del hidden
        frames = max(
            1,
            math.ceil(
                request["seconds"] * runtime["sample_rate"] / runtime["downsample_rate"]
            ),
        )
        sample = AuKSampleItem(
            conditioning=conditioning,
            text_mask=cpu_handoff(mask),
            target_frames=frames,
            ref_latent=reference_latents,
            ref_length=reference_length,
            seed=request["seed"],
        )
        latents = flow.sample(
            sample,
            steps=runtime["nfe"],
            cfg_strength=runtime["cfg_strength"],
            sway_sampling_coef=runtime["sway_sampling_coef"],
            t_grid=runtime["time_grid"] or None,
        )
        waveform = np.array(vae.decode(cpu_handoff(latents)[None])[0]).astype(
            np.float32
        )
        if not np.isfinite(waveform).all():
            raise RuntimeError(f"Nonfinite reference output: {request['id']}")
        else:
            pass
        output_path = arguments.output_directory / f"{request['id']}.wav"
        pcm = np.clip(waveform * np.float32(32768), -32768, 32767).astype("<i2")
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(runtime["sample_rate"])
            output.writeframes(pcm.tobytes())
        records.append(
            {
                "id": request["id"],
                "samples": len(waveform),
                "sha256": sha256(output_path),
            }
        )
        print(request["id"], flush=True)
    report = {
        "reference_revision": source["revision"],
        "dtype": arguments.dtype,
        "mlx_version": version("mlx"),
        "tf32": False,
        "requests_sha256": sha256(arguments.requests),
        "prepared_manifest_sha256": sha256(directory / "manifest.json"),
        "outputs": records,
    }
    (arguments.output_directory / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
