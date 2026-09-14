# SPDX-License-Identifier: Apache-2.0
"""Rust router topologies for model CI."""

from __future__ import annotations

import json
from enum import StrEnum

CI_ROUTER_MAX_INFLIGHT = 256
TTS_SERVING_WORKER_BATCH_LIMIT = 32
TTS_SERVING_BATCH_ADMISSION = CI_ROUTER_MAX_INFLIGHT * TTS_SERVING_WORKER_BATCH_LIMIT


class CiRouterTopology(StrEnum):
    ASR = "asr"
    TTS = "tts"
    TTS_SERVING = "tts_serving"
    OMNI_TEXT = "omni_text"
    OMNI_AUDIO = "omni_audio"


def render_router_config(
    *,
    topology: CiRouterTopology,
    router_port: int,
    worker_urls: list[str],
    model_name: str,
) -> str:
    """Render one current-schema router config for a homogeneous CI worker pool."""
    preamble = _router_preamble(topology, router_port)
    worker_blocks = [
        _worker_block(
            topology=topology,
            ordinal=ordinal,
            worker_url=worker_url,
            model_name=model_name,
        )
        for ordinal, worker_url in enumerate(worker_urls, start=1)
    ]
    return f"{preamble.rstrip()}\n\n" + "\n\n".join(worker_blocks) + "\n"


def _router_preamble(topology: CiRouterTopology, router_port: int) -> str:
    strategy = "round_robin" if topology is CiRouterTopology.ASR else "least_requests"
    admission = {
        CiRouterTopology.ASR: (
            f"global = {CI_ROUTER_MAX_INFLIGHT}",
            f"transcription_http = {CI_ROUTER_MAX_INFLIGHT}",
        ),
        CiRouterTopology.TTS: (
            f"global = {CI_ROUTER_MAX_INFLIGHT}",
            f"speech_http = {CI_ROUTER_MAX_INFLIGHT}",
        ),
        CiRouterTopology.TTS_SERVING: (
            f"global = {CI_ROUTER_MAX_INFLIGHT}",
            f"speech_http = {CI_ROUTER_MAX_INFLIGHT}",
            f"speech_batch = {TTS_SERVING_BATCH_ADMISSION}",
            f"speech_websocket = {CI_ROUTER_MAX_INFLIGHT}",
        ),
        CiRouterTopology.OMNI_TEXT: (
            f"global = {CI_ROUTER_MAX_INFLIGHT}",
            f"generation_http = {CI_ROUTER_MAX_INFLIGHT}",
        ),
        CiRouterTopology.OMNI_AUDIO: (
            f"global = {CI_ROUTER_MAX_INFLIGHT}",
            f"generation_http = {CI_ROUTER_MAX_INFLIGHT}",
        ),
    }[topology]
    lines = [
        "schema_version = 1",
        "",
        "[server]",
        f'listen = {_toml_string(f"127.0.0.1:{router_port}")}',
        "",
        "[shutdown]",
        "drain_timeout_ms = 30000",
        "",
        "[logging]",
        'format = "json"',
        'filter = "info"',
        "",
        "[router]",
        f"strategy = {_toml_string(strategy)}",
    ]
    if topology is CiRouterTopology.TTS_SERVING:
        lines.append('voice_owner_worker_id = "tts-serving-1"')
    lines.extend(
        [
            "",
            "[admission]",
            *admission,
            "",
            "[health]",
            "",
        ]
    )
    if topology in {CiRouterTopology.OMNI_TEXT, CiRouterTopology.OMNI_AUDIO}:
        lines.extend(
            [
                "[http_generation]",
                'trust_domain = "local"',
            ]
        )
        return "\n".join(lines)

    routes = {
        CiRouterTopology.ASR: ["transcription"],
        CiRouterTopology.TTS: ["speech"],
        CiRouterTopology.TTS_SERVING: ["speech", "speech_batch"],
    }[topology]
    lines.extend(
        [
            "[http_media]",
            f"routes = {_toml_array(routes)}",
            'trust_domain = "local"',
        ]
    )
    if topology is CiRouterTopology.TTS_SERVING:
        lines.extend(["", "[websocket.speech]", 'trust_domain = "local"'])
    return "\n".join(lines)


def _worker_block(
    *,
    topology: CiRouterTopology,
    ordinal: int,
    worker_url: str,
    model_name: str,
) -> str:
    prefix = {
        CiRouterTopology.ASR: "asr",
        CiRouterTopology.TTS: "tts",
        CiRouterTopology.TTS_SERVING: "tts-serving",
        CiRouterTopology.OMNI_TEXT: "omni",
        CiRouterTopology.OMNI_AUDIO: "omni",
    }[topology]
    worker_id = f"{prefix}-{ordinal}"
    lines = [
        "[[workers]]",
        f"worker_id = {_toml_string(worker_id)}",
        f"base_url = {_toml_string(worker_url.rstrip('/') + '/')}",
        'trust_domain = "local"',
        f"default_model_id = {_toml_string(model_name)}",
    ]
    if topology is CiRouterTopology.TTS_SERVING:
        lines.extend(
            [
                "",
                "[workers.capacity]",
                f"speech_websocket = {CI_ROUTER_MAX_INFLIGHT}",
            ]
        )
    lines.extend(["", _service_profiles(topology, model_name)])
    return "\n".join(lines)


def _service_profiles(topology: CiRouterTopology, model_name: str) -> str:
    model_ids = _toml_array([model_name])
    if topology is CiRouterTopology.ASR:
        return _transcription_profile(
            model_ids,
            task="transcribe",
            formats=["json", "verbose_json", "sse"],
        )
    if topology is CiRouterTopology.TTS:
        return "\n\n".join(
            [
                _speech_profile(
                    service="speech_http",
                    model_ids=model_ids,
                    response_formats=["wav"],
                    stream_modes=["non_streaming"],
                    tasks=["voice_clone"],
                    reference_forms=["direct", "list"],
                    voice_name_policy="uploaded",
                ),
                _speech_profile(
                    service="speech_http",
                    model_ids=model_ids,
                    response_formats=["pcm"],
                    stream_modes=["non_streaming", "streaming"],
                    tasks=["voice_clone"],
                    reference_forms=["direct", "list"],
                    voice_name_policy="uploaded",
                ),
            ]
        )
    if topology is CiRouterTopology.TTS_SERVING:
        profiles = [
            _speech_profile(
                service="speech_http",
                model_ids=model_ids,
                response_formats=["mp3", "opus", "aac", "flac", "wav"],
                stream_modes=["non_streaming"],
                tasks=["text_to_speech", "voice_clone", "voice_design"],
                reference_forms=["none", "direct", "list"],
                voice_name_policy="uploaded",
            ),
            _speech_profile(
                service="speech_http",
                model_ids=model_ids,
                response_formats=["pcm"],
                stream_modes=["non_streaming", "streaming"],
                tasks=["text_to_speech", "voice_clone", "voice_design"],
                reference_forms=["none", "direct", "list"],
                voice_name_policy="uploaded",
            ),
            _speech_batch_profile(model_ids),
            _speech_profile(
                service="speech_websocket",
                model_ids=model_ids,
                response_formats=["pcm"],
                stream_modes=["non_streaming", "streaming"],
                tasks=["text_to_speech", "voice_clone", "voice_design"],
                reference_forms=["none", "direct", "list"],
                voice_name_policy="uploaded",
            ),
        ]
        return "\n\n".join(profiles)
    return _generation_profile(
        model_ids=model_ids,
        audio_output=topology is CiRouterTopology.OMNI_AUDIO,
    )


def _transcription_profile(
    model_ids: str,
    *,
    task: str,
    formats: list[str],
) -> str:
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            'service = "transcription_http"',
            f"model_ids = {model_ids}",
            f"task = {_toml_string(task)}",
            f"response_formats = {_toml_array(formats)}",
            'stream_modes = ["non_streaming", "streaming"]',
        ]
    )


def _speech_profile(
    *,
    service: str,
    model_ids: str,
    response_formats: list[str],
    stream_modes: list[str],
    tasks: list[str],
    reference_forms: list[str],
    voice_name_policy: str,
) -> str:
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            f"service = {_toml_string(service)}",
            f"model_ids = {model_ids}",
            f"response_formats = {_toml_array(response_formats)}",
            f"stream_modes = {_toml_array(stream_modes)}",
            f"tasks = {_toml_array(tasks)}",
            f"reference_forms = {_toml_array(reference_forms)}",
            f"voice_name_policy = {_toml_string(voice_name_policy)}",
        ]
    )


def _speech_batch_profile(model_ids: str) -> str:
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            'service = "speech_batch"',
            f"model_ids = {model_ids}",
            'response_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]',
            'tasks = ["text_to_speech", "voice_clone", "voice_design"]',
            'reference_forms = ["none", "direct", "list"]',
            'voice_name_policy = "uploaded"',
            f"max_batch_size = {TTS_SERVING_WORKER_BATCH_LIMIT}",
        ]
    )


def _generation_profile(*, model_ids: str, audio_output: bool) -> str:
    output_modalities = ["text", "audio"] if audio_output else ["text"]
    audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"] if audio_output else []
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            'service = "generation_http"',
            f"model_ids = {model_ids}",
            'message_content_forms = ["string", "typed_parts"]',
            'media_placements = ["top_level", "typed_parts"]',
            'input_modalities = ["text", "image", "audio", "video"]',
            f"output_modalities = {_toml_array(output_modalities)}",
            f"chat_audio_formats = {_toml_array(audio_formats)}",
            'stream_modes = ["non_streaming", "streaming"]',
        ]
    )


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"
