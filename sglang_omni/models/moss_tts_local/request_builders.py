# SPDX-License-Identifier: Apache-2.0
"""Request adapters for MOSS-TTS Local.

Reuses the Delay preprocessing; the AR handoff differs: stop on ``audio_end``
and collect time-synchronous frames (no delay-pattern de-staggering).
"""

from __future__ import annotations

import time
from typing import Any

import torch

from sglang_omni.models.moss_tts.request_builders import (
    MOSS_TTS_DEFAULT_MAX_NEW_TOKENS,
    MossTTSSGLangRequestData,
    _new_moss_tts_sampling_seed,
    derive_moss_tts_sampling_seed,
    pop_prepared_moss_tts_request,
)
from sglang_omni.proto import StagePayload

# Upstream MossTTSLocal generate() defaults; per-request overrides not yet plumbed.
_LOCAL_SAMPLING = {
    "text_temperature": 1.5,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 1.0,
    "audio_top_p": 0.95,
    "audio_top_k": 50,
    "audio_repetition_penalty": 1.1,
}


def build_sglang_moss_tts_local_request(
    payload: StagePayload,
    *,
    model: Any,
) -> MossTTSSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prepared = pop_prepared_moss_tts_request(payload)
    if prepared is None:
        raise RuntimeError(
            "MOSS-TTS Local AR request builder requires a payload prepared by "
            "preprocess_moss_tts_payload"
        )

    cfg = model.config
    gen_kwargs = prepared.gen_kwargs
    max_new_tokens = int(
        gen_kwargs.get("max_new_tokens", MOSS_TTS_DEFAULT_MAX_NEW_TOKENS)
    )
    audio_end = int(cfg.audio_end_token_id)
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_token_ids=[audio_end],
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(cfg.vocab_size_list[0]))

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prepared.input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids={audio_end},
        vocab_size=int(cfg.vocab_size_list[0]),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None

    data = MossTTSSGLangRequestData(
        input_ids=prepared.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        output_ids=req.output_ids,
        req=req,
        state=prepared.state,
        model_config=cfg,
        prompt_rows=prepared.prompt_rows,
        text_temperature=float(_LOCAL_SAMPLING["text_temperature"]),
        text_top_p=float(_LOCAL_SAMPLING["text_top_p"]),
        text_top_k=int(_LOCAL_SAMPLING["text_top_k"]),
        audio_temperature=float(_LOCAL_SAMPLING["audio_temperature"]),
        audio_top_p=float(_LOCAL_SAMPLING["audio_top_p"]),
        audio_top_k=int(_LOCAL_SAMPLING["audio_top_k"]),
        audio_repetition_penalty=float(_LOCAL_SAMPLING["audio_repetition_penalty"]),
        seed=gen_kwargs.get("seed"),
        sampling_seed=(
            derive_moss_tts_sampling_seed(gen_kwargs["seed"])
            if gen_kwargs.get("seed") is not None
            else _new_moss_tts_sampling_seed()
        ),
        engine_start_s=time.perf_counter(),
    )
    data.input_embeds_are_projected = True
    data.stage_payload = payload
    return data


def apply_sglang_moss_tts_local_result(
    payload: StagePayload,
    data: MossTTSSGLangRequestData,
) -> StagePayload:
    state = data.state
    if data.output_rows:
        gen = torch.stack(data.output_rows, dim=0).to(dtype=torch.long)
        # Drop channel 0 (text); channels 1.. are the time-synchronous RVQ codes.
        state.delayed_audio_codes = gen[:, 1:].detach().cpu()
    else:
        n_vq = (
            int(data.prompt_rows.shape[1] - 1)
            if data.prompt_rows is not None and data.prompt_rows.ndim == 2
            else 0
        )
        state.delayed_audio_codes = torch.empty((0, n_vq), dtype=torch.long)

    state.prompt_tokens = len(data.input_ids) if data.input_ids is not None else 0
    state.completion_tokens = len(data.output_rows)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_moss_tts_local_scheduler_adapters(*, model: Any):
    def request_builder(payload: StagePayload) -> MossTTSSGLangRequestData:
        return build_sglang_moss_tts_local_request(payload, model=model)

    def result_adapter(data: MossTTSSGLangRequestData) -> StagePayload:
        return apply_sglang_moss_tts_local_result(data.stage_payload, data)

    return request_builder, result_adapter
