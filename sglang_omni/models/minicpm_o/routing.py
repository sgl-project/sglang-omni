# SPDX-License-Identifier: Apache-2.0
"""Routing and payload projections for MiniCPM-o stages."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.proto.request import OmniRequest, StagePayload

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
TALKER_STAGE = "talker"
CODE2WAV_STAGE = "code2wav"


def output_modalities(request: OmniRequest) -> set[str] | None:
    metadata = request.metadata or {}
    modalities = metadata.get("output_modalities")
    if modalities is None:
        return None
    else:
        pass
    if isinstance(modalities, str):
        values = (modalities,)
    elif isinstance(modalities, (list, tuple, set)):
        values = modalities
    else:
        return None
    return {str(modality).lower() for modality in values}


def should_generate_audio_output(
    payload_or_request: StagePayload | OmniRequest,
) -> bool:
    request = (
        payload_or_request.request
        if isinstance(payload_or_request, StagePayload)
        else payload_or_request
    )
    modalities = output_modalities(request)
    return modalities is None or "audio" in modalities


def speaker_reference_audio(payload: StagePayload) -> bytes | None:
    """Read an explicit, inline speaker reference from request parameters."""
    from sglang_omni.utils.audio import decode_audio_data_uri
    from sglang_omni.utils.audio_payload import audio_data_uri_from_reference

    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    stage_params = params.get("stage_params") or {}
    sources = (
        stage_params.get(CODE2WAV_STAGE) or {},
        metadata.get("audio_config") or {},
        metadata.get("tts_params") or {},
        params,
    )
    for source in sources:
        for key in ("ref_audio", "prompt_wav"):
            reference = source.get(key)
            if reference is None:
                continue
            else:
                pass
            if isinstance(reference, dict):
                reference = audio_data_uri_from_reference(reference)
            else:
                pass
            if isinstance(reference, bytes):
                if reference:
                    return reference
                else:
                    pass
            elif isinstance(reference, str):
                decoded = decode_audio_data_uri(reference)
                if decoded:
                    return decoded
                else:
                    pass
            else:
                pass
            raise ValueError(
                "MiniCPM-o ref_audio must be inline audio bytes or a base64 data "
                "URI; encode local files before sending"
            )
    return None


def resolve_preprocessing_next_stages(
    request_id: str, output: StagePayload
) -> list[str]:
    """Select encoder branches; request_id is required by the routing interface."""
    state = MiniCPMOPipelineState.from_dict(output.data)
    return [
        *encoder_stages_with_model_inputs(state.encoder_inputs),
        THINKER_STAGE,
    ]


def resolve_thinker_wait_sources(
    request_id: str,
    from_stage: str,
    payload: StagePayload,
) -> list[str] | None:
    """Select fan-in sources; request_id is required by the wait-source interface."""
    if from_stage != "preprocessing":
        return None
    else:
        pass
    state = MiniCPMOPipelineState.from_dict(payload.data)
    return [
        "preprocessing",
        *encoder_stages_with_model_inputs(state.encoder_inputs),
    ]


def project_preprocessing_to_image_encoder(payload: StagePayload) -> StagePayload:
    return project_preprocessing_to_encoder(payload, stage_name=IMAGE_STAGE)


def project_preprocessing_to_audio_encoder(payload: StagePayload) -> StagePayload:
    return project_preprocessing_to_encoder(payload, stage_name=AUDIO_STAGE)


def project_preprocessing_to_thinker(payload: StagePayload) -> StagePayload:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    projected = MiniCPMOPipelineState(
        prompt=dict(state.prompt) if isinstance(state.prompt, dict) else None,
        mm_inputs=dict(state.mm_inputs),
        encoder_inputs=project_encoder_input_metadata(state.encoder_inputs),
        stream_state=dict(state.stream_state),
        speaker_prompt=state.speaker_prompt,
    )
    return payload_with_state(payload, projected)


def project_encoder_to_thinker(payload: StagePayload) -> StagePayload:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    if len(state.encoder_outs) != 1:
        raise ValueError(
            "Expected exactly one encoder output in payload, got "
            f"{sorted(state.encoder_outs)}"
        )
    else:
        pass
    stage_name = next(iter(state.encoder_outs))
    projected = MiniCPMOPipelineState(
        encoder_outs={stage_name: state.encoder_outs[stage_name]}
    )
    return payload_with_state(payload, projected)


def resolve_thinker_next_stages(request_id: str, output: StagePayload) -> list[str]:
    """Select output branches; request_id is required by the routing interface."""
    if should_generate_audio_output(output):
        return [DECODE_STAGE, TALKER_STAGE]
    else:
        pass
    return [DECODE_STAGE]


def resolve_terminal_stages(request: OmniRequest) -> list[str]:
    if should_generate_audio_output(request):
        return [DECODE_STAGE, CODE2WAV_STAGE]
    else:
        pass
    return [DECODE_STAGE]


def project_thinker_to_talker(payload: StagePayload) -> StagePayload:
    """Project prompt ids, output ids, and captured hidden states for speech."""
    state = MiniCPMOPipelineState.from_dict(payload.data)
    thinker_out = state.thinker_out if isinstance(state.thinker_out, dict) else {}
    extra = thinker_out.get("extra_model_outputs") or {}
    projected = MiniCPMOPipelineState(
        prompt=dict(state.prompt) if isinstance(state.prompt, dict) else None,
        thinker_out={
            "output_ids": list(thinker_out.get("output_ids") or []),
            "extra_model_outputs": {
                "hidden_states_seq": extra.get("hidden_states_seq") or [],
            },
        },
        speaker_prompt=state.speaker_prompt,
    )
    return payload_with_state(payload, projected)


def project_talker_to_code2wav(payload: StagePayload) -> StagePayload:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    projected = MiniCPMOPipelineState(
        engine_outputs={TALKER_STAGE: state.engine_outputs.get(TALKER_STAGE) or {}},
        speaker_prompt=state.speaker_prompt,
    )
    return payload_with_state(payload, projected)


def project_thinker_to_decode(payload: StagePayload) -> StagePayload:
    """Keep decode payload focused on text detokenization state."""
    state = MiniCPMOPipelineState.from_dict(payload.data)
    state.thinker_inputs = {}
    state.speaker_prompt = None

    if isinstance(state.thinker_out, dict):
        thinker_out = dict(state.thinker_out)
        thinker_out.pop("extra_model_outputs", None)
        state.thinker_out = thinker_out
    else:
        pass

    if state.engine_outputs:
        engine_outputs = dict(state.engine_outputs)
        thinker_engine_out = engine_outputs.get(THINKER_STAGE)
        if isinstance(thinker_engine_out, dict):
            thinker_engine_out = dict(thinker_engine_out)
            thinker_engine_out.pop("extra_model_outputs", None)
            engine_outputs[THINKER_STAGE] = thinker_engine_out
        else:
            pass
        state.engine_outputs = engine_outputs
    else:
        pass

    return payload_with_state(payload, state)


def project_preprocessing_to_encoder(
    payload: StagePayload,
    *,
    stage_name: str,
) -> StagePayload:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    stage_inputs = state.encoder_inputs.get(stage_name)
    encoder_inputs = (
        {stage_name: dict(stage_inputs)} if isinstance(stage_inputs, dict) else {}
    )
    projected = MiniCPMOPipelineState(encoder_inputs=encoder_inputs)
    return payload_with_state(payload, projected)


def payload_with_state(
    payload: StagePayload, state: MiniCPMOPipelineState
) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def project_encoder_input_metadata(
    encoder_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    projected: dict[str, dict[str, Any]] = {}
    for stage_name, stage_inputs in encoder_inputs.items():
        if not isinstance(stage_inputs, dict):
            continue
        else:
            pass
        stage_metadata: dict[str, Any] = {}
        cache_key = stage_inputs.get("cache_key")
        if cache_key is not None:
            stage_metadata["cache_key"] = cache_key
        else:
            pass
        if has_encoder_model_input(stage_name, stage_inputs):
            stage_metadata["_active"] = True
        else:
            pass
        if stage_metadata:
            projected[stage_name] = stage_metadata
        else:
            pass
    return projected


def encoder_stages_with_model_inputs(
    encoder_inputs: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        stage_name
        for stage_name in (IMAGE_STAGE, AUDIO_STAGE)
        if has_encoder_model_input(stage_name, encoder_inputs.get(stage_name))
    ]


def has_encoder_model_input(stage_name: str, stage_inputs: Any) -> bool:
    if not isinstance(stage_inputs, dict):
        return False
    else:
        pass
    if stage_inputs.get("_active") is not None:
        return stage_inputs.get("_active") is True
    else:
        pass
    if stage_name == IMAGE_STAGE:
        return stage_inputs.get("pixel_values") is not None
    else:
        pass
    if stage_name == AUDIO_STAGE:
        return stage_inputs.get("audio_features") is not None
    else:
        pass
    return False
