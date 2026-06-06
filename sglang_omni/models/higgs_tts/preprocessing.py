# SPDX-License-Identifier: Apache-2.0
"""Higgs request normalization and reference preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchaudio.functional as F_audio

from sglang_omni.models.higgs_tts.codebook_layout import (
    apply_delay_pattern,
    to_codes_TN,
)
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.reference_audio import reference_audio_cache_key
from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter


@dataclass(frozen=True)
class HiggsPreprocessingConfig:
    num_codebooks: int
    codebook_size: int
    max_ref_audio_sec: int


@dataclass
class HiggsPreparedReference:
    prompt_token_ids: list[int]
    reference_codes_delayed: list[list[int]] | None
    reference_waveform: torch.Tensor | None
    reference_cache_key: str | None
    target_text_for_encoder: str | None
    reference_text_for_encoder: str | None


def normalize_higgs_inputs(raw_inputs: Any) -> dict[str, Any]:
    inputs = raw_inputs or {}
    if isinstance(inputs, str):
        inputs = {"text": inputs}

    raw_refs = inputs.get("references")
    if raw_refs and isinstance(raw_refs, list):
        first = raw_refs[0]
        if isinstance(first, dict):
            inputs = dict(inputs)
            if first.get("text") and not inputs.get("reference_text"):
                inputs["reference_text"] = first["text"]
            if inputs.get("reference_audio") is None:
                if "bytes" in first or "base64" in first or "data" in first:
                    inputs["reference_audio"] = first
                else:
                    inputs["reference_audio"] = first.get("audio_path") or first.get(
                        "path"
                    )
    return inputs


def prepare_higgs_reference(
    inputs: dict[str, Any],
    *,
    adapter: HiggsTokenizerAdapter,
    reference_waveform_cache: Any,
    reference_waveform_cache_lock: Any,
    load_audio_fn: Any,
    config: HiggsPreprocessingConfig,
) -> HiggsPreparedReference:
    text = inputs.get("input") or inputs.get("text") or ""
    reference_text = inputs.get("reference_text") or None
    ref_codes_TN = to_codes_TN(inputs.get("reference_codes"), config.num_codebooks)
    if (
        ref_codes_TN is not None
        and ref_codes_TN.shape[0] > config.max_ref_audio_sec * 75
    ):
        raise ValueError(
            f"reference_codes is too long ({ref_codes_TN.shape[0]} frames); "
            f"cap at {config.max_ref_audio_sec}s of audio "
            f"(~{config.max_ref_audio_sec * 75} frames at 75 Hz)."
        )

    waveform_tensor = None
    reference_cache_key = None
    if ref_codes_TN is None and inputs.get("reference_audio") is not None:
        reference_audio = inputs["reference_audio"]
        reference_cache_key = reference_audio_cache_key(reference_audio)
        with reference_waveform_cache_lock:
            cached_waveform = reference_waveform_cache.get(reference_cache_key)
        if cached_waveform is not None:
            waveform_tensor = cached_waveform.clone()
        if waveform_tensor is None:
            waveform_np, sample_rate = load_audio_fn(reference_audio)
            wav = torch.from_numpy(waveform_np)
            if sample_rate != 24000:
                wav = F_audio.resample(wav, sample_rate, 24000)
            if wav.shape[-1] > config.max_ref_audio_sec * 24000:
                raise ValueError(
                    f"reference_audio is too long "
                    f"({wav.shape[-1] / 24000:.1f}s); "
                    f"cap at {config.max_ref_audio_sec}s."
                )
            waveform_tensor = wav.view(1, 1, -1).contiguous().float()
            with reference_waveform_cache_lock:
                reference_waveform_cache.put(
                    reference_cache_key, waveform_tensor.clone()
                )

    if ref_codes_TN is not None:
        delayed = apply_delay_pattern(ref_codes_TN)
        prompt_ids = adapter.build_prompt(
            text,
            num_ref_tokens=delayed.shape[0],
            reference_text=reference_text,
        )
        ref_codes_delayed: list[list[int]] | None = delayed.tolist()
        target_text_for_encoder = None
        reference_text_for_encoder = None
    elif waveform_tensor is None:
        prompt_ids = adapter.build_prompt(
            text, num_ref_tokens=0, reference_text=reference_text
        )
        ref_codes_delayed = None
        target_text_for_encoder = None
        reference_text_for_encoder = None
    else:
        prompt_ids = []
        ref_codes_delayed = None
        target_text_for_encoder = text
        reference_text_for_encoder = reference_text

    return HiggsPreparedReference(
        prompt_token_ids=prompt_ids,
        reference_codes_delayed=ref_codes_delayed,
        reference_waveform=waveform_tensor,
        reference_cache_key=reference_cache_key,
        target_text_for_encoder=target_text_for_encoder,
        reference_text_for_encoder=reference_text_for_encoder,
    )


def build_higgs_preprocessed_state(
    prepared: HiggsPreparedReference,
    *,
    params: Any,
    config: HiggsPreprocessingConfig,
) -> HiggsTtsState:
    return HiggsTtsState(
        prompt_token_ids=prepared.prompt_token_ids,
        reference_codes_delayed=prepared.reference_codes_delayed,
        reference_waveform=prepared.reference_waveform,
        reference_cache_key=prepared.reference_cache_key,
        target_text=prepared.target_text_for_encoder,
        reference_text=prepared.reference_text_for_encoder,
        num_codebooks=config.num_codebooks,
        codebook_size=config.codebook_size,
        max_new_tokens=int(params.get("max_new_tokens", 2048)),
        temperature=float(params.get("temperature", 1.0)),
        top_p=params.get("top_p"),
        top_k=params.get("top_k"),
        seed=params.get("seed"),
    )


def encode_higgs_reference_audio(
    state: HiggsTtsState,
    *,
    codec: Any,
    adapter: HiggsTokenizerAdapter,
    reference_code_cache: Any,
    num_codebooks: int,
) -> HiggsTtsState:
    waveform = state.reference_waveform
    cached_delayed = reference_code_cache.get(state.reference_cache_key)
    if cached_delayed is not None:
        delayed_rows = cached_delayed.tolist()
    else:
        ref_codes_TN = codec.encode_reference(waveform, sample_rate=24000).to(
            torch.long
        )
        if ref_codes_TN.ndim != 2 or ref_codes_TN.shape[1] != num_codebooks:
            raise ValueError(
                f"codec output must be [T, {num_codebooks}], got "
                f"{tuple(ref_codes_TN.shape)}"
            )
        delayed = apply_delay_pattern(ref_codes_TN)
        delayed_rows = delayed.tolist()
        reference_code_cache.put(
            state.reference_cache_key, delayed.to("cpu", torch.int32)
        )
    state.reference_codes_delayed = delayed_rows
    state.prompt_token_ids = adapter.build_prompt(
        state.target_text or "",
        num_ref_tokens=len(delayed_rows),
        reference_text=state.reference_text,
    )
    state.reference_waveform = None
    state.reference_cache_key = None
    state.target_text = None
    state.reference_text = None
    return state
