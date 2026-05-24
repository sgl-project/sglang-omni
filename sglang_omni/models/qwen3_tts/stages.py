# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Qwen3-TTS Base pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.request_builders import (
    cleanup_prepared_qwen3_tts_request,
    make_qwen3_tts_scheduler_adapters,
    preprocess_qwen3_tts_payload,
    set_qwen3_tts_preprocessing_context,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)

_QWEN_TTS_INSTALL_HINT = (
    "Qwen3-TTS support requires the official `qwen-tts` package. "
    "Install `qwen-tts==0.1.1` and its Transformers 4.57.3 requirement "
    "in the serving environment before launching Qwen3-TTS."
)


def load_state(payload: StagePayload) -> Qwen3TTSState:
    return Qwen3TTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: Qwen3TTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_qwen3_tts_tokenizer(
    model_path: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
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


def _audio_to_list(audio: Any) -> list[float]:
    if isinstance(audio, torch.Tensor):
        return audio.detach().float().cpu().flatten().tolist()
    try:
        import numpy as np

        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        return array.tolist()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported Qwen3-TTS audio output type: {type(audio)}"
        ) from exc


def _build_usage(state: Qwen3TTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


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
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> Any:
    from qwen_tts import Qwen3TTSModel
    from transformers import AutoProcessor

    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    _register_qwen3_tts_hf_config()
    checkpoint_dir = _resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=8192,
        dtype=dtype,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        mem_fraction_static=0.85,
        max_prefill_tokens=8192,
        max_running_requests=16,
        sampling_backend="pytorch",
        trust_remote_code=True,
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="Qwen3TTSTalker",
    )

    model = model_worker.model_runner.model
    speech_tokenizer = _load_qwen3_tts_tokenizer(
        checkpoint_dir,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.load_speech_tokenizer(speech_tokenizer)
    processor = AutoProcessor.from_pretrained(checkpoint_dir, fix_mistral_regex=True)
    wrapper = Qwen3TTSModel(
        model=model,
        processor=processor,
        generate_defaults=_load_qwen3_tts_generate_defaults(checkpoint_dir),
    )
    set_qwen3_tts_preprocessing_context(model=model, wrapper=wrapper)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_qwen3_tts_scheduler_adapters(
        model=model,
        wrapper=wrapper,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=Qwen3TTSModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=cleanup_prepared_qwen3_tts_request,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    tokenizer = _load_qwen3_tts_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    def _prepare_vocoder_item(
        payload: StagePayload,
    ) -> tuple[Qwen3TTSState, torch.Tensor]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError("Qwen3-TTS vocoder requires audio_codes from tts_engine")

        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        return state, codes

    def _store_vocoder_result(
        payload: StagePayload,
        state: Qwen3TTSState,
        codes: torch.Tensor,
        wav: Any,
        sample_rate: int,
    ) -> StagePayload:
        if wav is None:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")

        if state.ref_code_len:
            total_len = int(codes.shape[0])
            cut = int(state.ref_code_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]
        state.audio_samples = _audio_to_list(wav)
        state.sample_rate = int(sample_rate)
        state.audio_codes = None

        payload = store_state(payload, state)
        audio = state.audio_samples or []
        payload.data["audio_data"] = audio
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _vocode(payload: StagePayload) -> StagePayload:
        state, codes = _prepare_vocoder_item(payload)
        wavs, sample_rate = tokenizer.decode([{"audio_codes": codes}])
        wav = wavs[0] if wavs else None
        return _store_vocoder_result(payload, state, codes, wav, sample_rate)

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        items = [_prepare_vocoder_item(payload) for payload in payloads]
        wavs, sample_rate = tokenizer.decode(
            [{"audio_codes": codes} for _, codes in items]
        )
        if len(wavs) != len(items):
            raise RuntimeError(
                f"Qwen3-TTS speech tokenizer returned {len(wavs)} audios for {len(items)} requests"
            )
        return [
            _store_vocoder_result(payload, state, codes, wav, sample_rate)
            for payload, (state, codes), wav in zip(payloads, items, wavs)
        ]

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
