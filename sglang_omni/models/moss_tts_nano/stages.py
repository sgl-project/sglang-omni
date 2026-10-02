# SPDX-License-Identifier: Apache-2.0
"""Stage factory for MOSS-TTS-Nano."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang_omni.models.moss_tts_nano.request_builders import (
    build_moss_tts_nano_request,
)
from sglang_omni.models.weight_loader import resolve_dtype, resolve_model_path
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device


def create_tts_executor(
    model_path: str,
    *,
    device: str | None,
    gpu_id: int | None = None,
    dtype: str,
    attention_backend: Literal["auto", "eager", "sdpa", "flash_attention_2"],
    audio_tokenizer_path: str | None = None,
) -> SimpleScheduler:
    concrete_device = resolve_concrete_device(device, gpu_id)
    torch_dtype = resolve_dtype(dtype)
    assert torch_dtype is not None
    if attention_backend not in {"auto", "eager", "sdpa", "flash_attention_2"}:
        raise ValueError(
            "MOSS-TTS-Nano attention_backend must be eager, sdpa, or flash_attention_2"
        )
    backend = "sdpa" if attention_backend == "auto" else attention_backend
    if backend == "flash_attention_2" and concrete_device.type != "cuda":
        raise ValueError("MOSS-TTS-Nano flash_attention_2 requires CUDA")
    if backend == "flash_attention_2" and torch_dtype is torch.float32:
        raise ValueError("MOSS-TTS-Nano flash_attention_2 requires float16 or bfloat16")

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(
        model_config.auto_map["AutoModelForCausalLM"], model_path
    )
    checkpoint_path = resolve_model_path(model_path) / "pytorch_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"MOSS-TTS-Nano checkpoint weights not found: {checkpoint_path}"
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # note (Codex): Direct loading avoids non-finite outputs with the HF 5.12 loader.
    model = model_class(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=concrete_device, dtype=torch_dtype)
    model._set_attention_implementation(
        backend,
        local_attn_implementation=backend,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    codec_path = audio_tokenizer_path or str(
        model.config.audio_tokenizer_pretrained_name_or_path
    )
    audio_tokenizer = AutoModel.from_pretrained(
        codec_path,
        trust_remote_code=True,
    ).to(concrete_device)
    codec_uses_flash_attention = backend == "flash_attention_2"
    codec_attention_backend = (
        "flash_attention_2" if codec_uses_flash_attention else "sdpa"
    )
    if codec_uses_flash_attention:
        codec_compute_dtype = "bf16" if torch_dtype is torch.bfloat16 else "fp16"
    else:
        codec_compute_dtype = "fp32"
    audio_tokenizer.set_attention_implementation(codec_attention_backend)
    audio_tokenizer.set_compute_dtype(codec_compute_dtype)
    audio_tokenizer.eval()

    def generate(payload: StagePayload) -> StagePayload:
        request = build_moss_tts_nano_request(payload)
        generation = dict(request.generation_kwargs)
        seed = generation.pop("seed", None)
        if seed is not None:
            torch.manual_seed(seed)

        with tempfile.TemporaryDirectory(prefix="sglang-omni-moss-nano-") as tmp:
            temporary_dir = Path(tmp)
            reference = request.ref_audio
            if isinstance(reference, str) and reference.startswith("data:"):
                header, separator, encoded = reference.partition(",")
                if not separator or ";base64" not in header:
                    raise ValueError(
                        "MOSS-TTS-Nano reference data URI must be base64 encoded"
                    )
                reference_file = temporary_dir / "reference.wav"
                reference_file.write_bytes(base64.b64decode(encoded, validate=True))
                reference_path = str(reference_file)
            elif isinstance(reference, bytes):
                reference_file = temporary_dir / "reference.wav"
                reference_file.write_bytes(reference)
                reference_path = str(reference_file)
            elif isinstance(reference, (str, Path)):
                reference_path = str(Path(reference).expanduser())
            else:
                reference_path = None
            mode = (
                "voice_clone"
                if reference_path and not request.ref_text
                else "continuation"
            )
            result = model.inference(
                text=request.text,
                output_audio_path=temporary_dir / "output.wav",
                mode=mode,
                prompt_text=request.ref_text if mode == "continuation" else None,
                prompt_audio_path=reference_path,
                text_tokenizer=tokenizer,
                audio_tokenizer=audio_tokenizer,
                device=concrete_device,
                **generation,
            )

        waveform = torch.as_tensor(result["waveform"], dtype=torch.float32)
        if waveform.numel() == 0:
            raise RuntimeError("MOSS-TTS-Nano generated no audio")
        payload.data = audio_waveform_payload(
            waveform,
            sample_rate=int(result["sample_rate"]),
            modality="audio",
            source_hint="MOSS-TTS-Nano",
            keep_channels=True,
        )
        return payload

    return SimpleScheduler(generate)
