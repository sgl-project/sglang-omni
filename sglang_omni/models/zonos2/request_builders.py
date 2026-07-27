# SPDX-License-Identifier: Apache-2.0
"""Map an OpenAI /v1/audio/speech request onto a Zonos2State.

Reuses the MOSS input/reference parsers so ``--ref-format references`` payloads
work unchanged.
"""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts.request_builders import (
    _resolve_optional_text,
    normalize_moss_tts_inputs,
    resolve_moss_reference,
)
from sglang_omni.models.zonos2.components.text_frontend import (
    TTSSamplingParams,
    make_speaker_slot,
)
from sglang_omni.models.zonos2.payload_types import Zonos2State
from sglang_omni.models.zonos2.radix_hash import RADIX_HASH_SPACE, poly_row_hash
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM

_DATA_URI_RE = re.compile(r"^data:[^;,]*;base64,(?P<data>.+)$", re.DOTALL)

_SAMPLING_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repetition_penalty",
)


def ref_audio_to_encoder_input(ref_audio: Any) -> Any:
    """Decode a base64 data-URI reference to raw bytes; pass paths/arrays through."""
    if isinstance(ref_audio, str):
        m = _DATA_URI_RE.match(ref_audio)
        if m is not None:
            return base64.b64decode(m.group("data"))
    return ref_audio


def build_zonos2_state(payload: StagePayload) -> Zonos2State:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = normalize_moss_tts_inputs(inputs)
    ref_audio, ref_text = resolve_moss_reference(references, tts_params)
    language = _resolve_optional_text(
        tts_params.get("language") or params.get("language")
    )

    explicit = tts_params.get("explicit_generation_params")
    explicit = (
        {str(f) for f in explicit}
        if isinstance(explicit, (list, tuple, set))
        else set()
    )
    if "seed" in explicit or tts_params.get("seed") is not None:
        raise ValueError(
            "ZONOS2 does not support seed because sampling uses the shared device RNG"
        )

    gen: dict[str, Any] = {}
    raw_max = params.get("max_new_tokens")
    if raw_max is not None and not isinstance(raw_max, bool):
        gen["max_tokens"] = int(raw_max)
    for sampling_field in _SAMPLING_FIELDS:
        for source in (tts_params, params):
            val = source.get(sampling_field)
            if val is None:
                continue
            # tts_params always wins; params only honored when explicitly whitelisted
            if sampling_field in explicit or source is tts_params:
                gen[sampling_field] = (
                    int(val) if sampling_field == "top_k" else float(val)
                )
            break

    return Zonos2State(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        generation_kwargs=gen,
    )


# ---- AR engine request adapters (OmniScheduler) ----

_AR_SAMPLING_FIELDS = {
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repetition_penalty",
    "repetition_window",
    "repetition_codebooks",
    "max_tokens",
    "ignore_eos",
}


@dataclass
class Zonos2SGLangRequestData(SGLangARRequestData):
    prompt_rows: torch.Tensor | None = None
    speaker_emb: torch.Tensor | None = None
    speaker_position: int = -1
    params: TTSSamplingParams = field(default_factory=TTSSamplingParams)
    output_codes: list = field(default_factory=list)
    rep_hist: list = field(default_factory=list)
    eos_frame: int | None = None
    eos_countdown: int = 0
    generation_step: int = 0
    engine_start_s: float = 0.0
    stream_metadata: dict | None = None
    _stream_emit_idx: int = 0


def build_zonos2_stream_metadata(payload: StagePayload, *, n_codebooks: int):
    """Per-frame stream-chunk metadata, or None when the request is not streaming."""
    params = payload.request.params
    if not isinstance(params, dict) or not params.get("stream"):
        return None
    metadata = {
        "stream": True,
        "modality": "audio_codes",
        "n_codebooks": int(n_codebooks),
    }
    initial = params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM)
    if initial is not None:
        metadata[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = initial
    return metadata


def _marker_token_ids(cfg) -> tuple[int, int]:
    """(clean/noisy background, accurate-mode) text-column ids per the layout."""
    base = (
        cfg.text_vocab - cfg.speaking_rate_num_buckets - cfg.quality_num_buckets - 2 - 1
    )
    cond = cfg.speaking_rate_num_buckets + cfg.quality_num_buckets
    return base + cond + 1, base + cond + 2  # noisy bg (default), accurate


def _marker_row(cfg, tok: int) -> torch.Tensor:
    row = torch.full((1, cfg.n_codebooks + 1), cfg.audio_pad_id, dtype=torch.long)
    row[0, cfg.n_codebooks] = tok
    return row


def build_sglang_zonos2_request(
    payload: StagePayload, *, model: Any
) -> Zonos2SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    cfg = model.config
    state = Zonos2State.from_dict(payload.data)

    rows = torch.as_tensor(state.input_ids, dtype=torch.long)
    speaker_emb = state.speaker_emb
    speaker_position = -1
    if speaker_emb is not None:
        speaker_emb = torch.as_tensor(speaker_emb, dtype=torch.float32)
        bg, acc = _marker_token_ids(cfg)
        rows = torch.cat(
            [
                make_speaker_slot().to(torch.long),
                _marker_row(cfg, bg),
                _marker_row(cfg, acc),
                rows,
            ],
            dim=0,
        )
        speaker_position = 0

    row_keys = poly_row_hash(rows).tolist()
    params = TTSSamplingParams(
        **{k: v for k, v in state.generation_kwargs.items() if k in _AR_SAMPLING_FIELDS}
    )
    max_new = int(params.max_tokens)

    sp = SamplingParams(max_new_tokens=max_new, temperature=0.0)
    sp.normalize(None)
    sp.verify(RADIX_HASH_SPACE + 1)
    # Continuing frames hash into [0, RADIX_HASH_SPACE); the runner emits
    # RADIX_HASH_SPACE as the stop token once the EOS countdown completes.
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=row_keys,
        sampling_params=sp,
        eos_token_ids={RADIX_HASH_SPACE},
        vocab_size=RADIX_HASH_SPACE + 1,
        extra_key=state.speaker_fingerprint,
    )
    req.tokenizer = None

    data = Zonos2SGLangRequestData(
        input_ids=torch.tensor(row_keys, dtype=torch.long),
        max_new_tokens=max_new,
        output_ids=req.output_ids,
        req=req,
        prompt_rows=rows,
        speaker_emb=speaker_emb,
        speaker_position=speaker_position,
        params=params,
        engine_start_s=time.perf_counter(),
    )
    data.stage_payload = payload
    data.stream_metadata = build_zonos2_stream_metadata(
        payload, n_codebooks=cfg.n_codebooks
    )
    return data


def apply_sglang_zonos2_result(
    payload: StagePayload, data: Zonos2SGLangRequestData
) -> StagePayload:
    state = Zonos2State.from_dict(payload.data)
    if data.output_codes:
        state.audio_codes = torch.stack(data.output_codes, dim=0).to(torch.long)
    else:
        state.audio_codes = torch.empty((0, 9), dtype=torch.long)
    state.eos_frame = data.eos_frame
    state.prompt_tokens = (
        int(data.prompt_rows.shape[0]) if data.prompt_rows is not None else 0
    )
    state.completion_tokens = len(data.output_codes)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=state.to_dict()
    )


def make_zonos2_scheduler_adapters(*, model: Any):
    def request_builder(payload: StagePayload) -> Zonos2SGLangRequestData:
        return build_sglang_zonos2_request(payload, model=model)

    def result_adapter(data: Zonos2SGLangRequestData) -> StagePayload:
        try:
            return apply_sglang_zonos2_result(data.stage_payload, data)
        finally:
            # note(ratish): For completed requests, scheduler terminalization—not
            # sampler EOS—is the ownership boundary; finally prevents adapter
            # errors from stranding the row.
            model.reset_request(data.stage_payload.request_id)

    return request_builder, result_adapter
