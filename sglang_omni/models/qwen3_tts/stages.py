# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Qwen3-TTS Base pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.request_builders import (
    cleanup_prepared_qwen3_tts_request,
    preprocess_qwen3_tts_payload,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)

_QWEN_TTS_INSTALL_HINT = (
    "Qwen3-TTS support requires the official `qwen-tts` package. "
    "Install `qwen-tts==0.1.1` and its Transformers 4.57.3 requirement "
    "in the serving environment before launching Qwen3-TTS."
)


def load_state(payload: StagePayload) -> Qwen3TTSState:
    return _load_pipeline_state(payload, Qwen3TTSState)


def store_state(payload: StagePayload, state: Qwen3TTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _load_qwen3_tts_tokenizer(
    model_path: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    apply_qwen_tts_transformers_compatibility_patches()
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc

    checkpoint_dir = _resolve_checkpoint(model_path)
    tokenizer_path = os.path.join(checkpoint_dir, "speech_tokenizer")
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": torch_dtype,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    logger.info(f"Loading Qwen3-TTS speech tokenizer from {tokenizer_path} on {device}")
    return Qwen3TTSTokenizer.from_pretrained(tokenizer_path, **kwargs)


def _register_qwen3_tts_hf_config() -> None:
    apply_qwen_tts_transformers_compatibility_patches()
    try:
        from qwen_tts.core.models import Qwen3TTSConfig
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc
    if not hasattr(Qwen3TTSConfig, "_sglang_omni_patched"):
        original_init = Qwen3TTSConfig.__init__

        def _patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            talker_config = getattr(self, "talker_config", None)
            if talker_config is not None:
                self.text_config = talker_config

        Qwen3TTSConfig.__init__ = _patched_init
        Qwen3TTSConfig._sglang_omni_patched = True
    try:
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    except ValueError:
        pass


def _load_qwen3_tts_generate_defaults(checkpoint_dir: str) -> dict[str, Any]:
    import json

    path = os.path.join(checkpoint_dir, "generation_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _compile_qwen3_tts_backbone(model: Any) -> None:
    """Compile decoder blocks while leaving decode-input staging eager."""

    text_model = model.model
    layers = text_model.layers

    from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config

    set_torch_compile_config()
    compile_mode = os.environ.get(
        "SGLANG_TORCH_COMPILE_MODE",
        "max-autotune-no-cudagraphs",
    )
    text_model._compiled_decode_layers = [
        torch.compile(layer, mode=compile_mode) for layer in layers
    ]


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    del model_path
    return SimpleScheduler(
        preprocess_qwen3_tts_payload,
        abort_callback=cleanup_prepared_qwen3_tts_request,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    return Qwen3TtsEngineBuilder(
        attn_implementation=attn_implementation,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


class _Qwen3TTSVocoder(BatchVocoderBase):
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def prepare_item(self, payload: StagePayload) -> tuple[Qwen3TTSState, torch.Tensor]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError("Qwen3-TTS vocoder requires audio_codes from tts_engine")

        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        return state, codes

    async def decode_batch(
        self, items: list[tuple[Qwen3TTSState, torch.Tensor]]
    ) -> list[tuple[Any, int]]:
        wavs, sample_rate = self._tokenizer.decode(
            [{"audio_codes": codes} for _, codes in items]
        )
        if len(wavs) != len(items):
            raise RuntimeError(
                f"Qwen3-TTS speech tokenizer returned {len(wavs)} audios for {len(items)} requests"
            )
        return [(wav, sample_rate) for wav in wavs]

    def store_result(
        self,
        payload: StagePayload,
        state: Qwen3TTSState,
        wav: Any,
        sample_rate: int,
    ) -> StagePayload:
        if wav is None:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")

        if state.ref_code_len:
            total_len = len(state.audio_codes)
            cut = int(state.ref_code_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]
        audio_payload = audio_waveform_payload(wav, source_hint="Qwen3-TTS")
        state.audio_samples = None
        state.sample_rate = int(sample_rate)
        state.audio_codes = None

        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    tokenizer = _load_qwen3_tts_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    return _Qwen3TTSVocoder(tokenizer).build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
