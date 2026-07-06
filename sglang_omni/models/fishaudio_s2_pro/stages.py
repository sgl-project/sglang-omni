# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Fish Audio S2-Pro TTS pipeline.

Each factory returns a callable (for SimpleScheduler) or an OmniScheduler.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)


def _compile_s2pro_codebook_decoder(model: Any, *, max_batch_size: int) -> None:
    """Compile Fast AR decoder layers while leaving sampling and loop control eager."""
    from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config

    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    set_torch_compile_config()
    compile_mode = os.environ.get(
        "SGLANG_TORCH_COMPILE_MODE",
        "max-autotune-no-cudagraphs",
    )
    audio_decoder = model._audio_decoder
    compiled_forward_kvcached_layers = [
        torch.compile(layer.forward_kvcached, mode=compile_mode)
        for layer in audio_decoder.layers
    ]
    audio_decoder.set_compiled_forward_kvcached_layers(
        compiled_forward_kvcached_layers,
        max_batch_size=max_batch_size,
    )
    logger.info(
        "Compiled %d Fast AR decoder layers (mode=%s, max_batch_size=%d)",
        len(compiled_forward_kvcached_layers),
        compile_mode,
        max_batch_size,
    )


def _resolve_s2pro_model_buffer_bs(model: Any) -> int:
    return min(
        int(model.vq_decode_max_batch_size),
        int(model._audio_decoder.kv_cache_max_batch_size),
    )


def _load_codec(checkpoint_dir: str, device: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    codec_path = os.path.join(checkpoint_dir, "codec.pth")
    import sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac as _dac_mod

    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_dac_mod.__file__))),
        "configs",
    )
    cfg = OmegaConf.load(os.path.join(configs_dir, "modded_dac_vq.yaml"))
    codec = instantiate(cfg)
    state_dict = torch.load(
        codec_path, map_location=device, mmap=True, weights_only=True
    )
    codec.load_state_dict(state_dict, strict=False, assign=True)
    codec.eval().to(device)
    return codec


def load_state(payload: StagePayload) -> S2ProState:
    return _load_pipeline_state(payload, S2ProState)


def store_state(payload: StagePayload, state: S2ProState) -> StagePayload:
    return _store_pipeline_state(payload, state)


# ---------------------------------------------------------------------------
# Preprocessing — returns callable
# ---------------------------------------------------------------------------


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
):
    """Returns a threaded scheduler for CPU-heavy preprocessing."""
    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    checkpoint_dir = _resolve_checkpoint(model_path)

    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = S2ProTokenizerAdapter(tokenizer)
    codec = _load_codec(checkpoint_dir, "cpu")

    def _encode_reference_waveform(audio: torch.Tensor, sr: int) -> torch.Tensor:
        import torchaudio

        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        audios = audio.squeeze(0).unsqueeze(0)
        audio_lengths = torch.tensor([audios.shape[1]], dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()

    def _encode_reference_audio(audio_path: str) -> torch.Tensor:
        import torchaudio

        audio, sr = torchaudio.load(audio_path)
        return _encode_reference_waveform(audio, int(sr))

    def _encode_reference_data(ref_data: dict[str, Any]) -> torch.Tensor | None:
        data = ref_data.get("base64") or ref_data.get("data")
        if data is None and ref_data.get("bytes") is None:
            return None

        from sglang_omni.preprocessing.audio import AudioMediaIO

        audio_io = AudioMediaIO(target_sr=codec.sample_rate)
        if ref_data.get("bytes") is not None:
            audio, sr = audio_io.load_bytes(ref_data["bytes"])
        else:
            audio, sr = audio_io.load_base64(
                ref_data.get("media_type") or "audio/wav", data
            )
        audio_tensor = torch.from_numpy(audio).float().reshape(1, -1)
        return _encode_reference_waveform(audio_tensor, int(sr))

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        num_codebooks = inputs.get("num_codebooks", 10)
        codebook_size = inputs.get("codebook_size", 4096)

        references = None
        raw_refs = inputs.get("references")
        if raw_refs:
            references = []
            for ref_data in raw_refs:
                vq_codes = ref_data.get("vq_codes")
                if vq_codes is not None and not isinstance(vq_codes, torch.Tensor):
                    vq_codes = torch.tensor(vq_codes)
                if vq_codes is None and ref_data.get("audio_path"):
                    vq_codes = _encode_reference_audio(ref_data["audio_path"])
                if vq_codes is None:
                    vq_codes = _encode_reference_data(ref_data)
                references.append(
                    Reference(
                        audio_bytes=b"",
                        text=ref_data.get("text", ""),
                        vq_codes=vq_codes,
                    )
                )

        prompt_data = adapter.build_prompt(
            text=text, references=references, num_codebooks=num_codebooks
        )
        state = S2ProState(
            input_ids=prompt_data["input_ids"],
            vq_mask_tokens=prompt_data["vq_mask_tokens"],
            vq_parts=prompt_data["vq_parts"],
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.8),
            top_k=params.get("top_k", 30),
            repetition_penalty=params.get("repetition_penalty", 1.1),
            seed=params.get("seed"),
        )
        return store_state(payload, state)

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


# ---------------------------------------------------------------------------
# TTS Engine — returns OmniScheduler
# ---------------------------------------------------------------------------


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    top_k: int = 30,
    ras_window: int = 16,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Returns OmniScheduler for the Fish TTS AR engine."""
    del top_k
    from sglang_omni.models.fishaudio_s2_pro.engine_builder import (
        FishS2ProEngineBuilder,
    )

    return FishS2ProEngineBuilder(
        max_new_tokens=max_new_tokens,
        ras_window=ras_window,
    ).build(
        model_path,
        device=device,
        server_args_overrides=server_args_overrides,
    )


# ---------------------------------------------------------------------------
# Vocoder — returns callable
# ---------------------------------------------------------------------------


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 10,
    stream_followup_stride: int = 90,
    stream_overlap_tokens: int | None = 20,
    stream_crossfade_samples: int = 512,
):
    from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import (
        S2ProVocoderScheduler,
    )

    if device is None:
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    checkpoint_dir = _resolve_checkpoint(model_path)
    codec = _load_codec(checkpoint_dir, device)

    return S2ProVocoderScheduler(
        codec,
        device=device,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
