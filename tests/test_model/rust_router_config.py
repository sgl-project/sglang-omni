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
    generation_streaming: bool = True,
    named_voice: bool = False,
) -> str:
    """Render one current-schema router config for a homogeneous CI worker pool.

    ``named_voice`` describes a TTS pool whose checkpoint serves preset voices
    from the text alone; the speech profiles then advertise ``text_to_speech``
    without a reference instead of ``voice_clone``.
    """
    preamble = router_preamble(topology, router_port)
    worker_blocks = [
        worker_block(
            topology=topology,
            ordinal=ordinal,
            worker_url=worker_url,
            model_name=model_name,
            generation_streaming=generation_streaming,
            named_voice=named_voice,
        )
        for ordinal, worker_url in enumerate(worker_urls, start=1)
    ]
    return f"{preamble.rstrip()}\n\n" + "\n\n".join(worker_blocks) + "\n"


def router_preamble(topology: CiRouterTopology, router_port: int) -> str:
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
        f'listen = {toml_string(f"127.0.0.1:{router_port}")}',
        "",
        "[shutdown]",
        "drain_timeout_ms = 30000",
        "",
        "[logging]",
        'format = "json"',
        'filter = "info"',
        "",
        "[router]",
        f"strategy = {toml_string(strategy)}",
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
            f"routes = {toml_array(routes)}",
            'trust_domain = "local"',
        ]
    )
    if topology is CiRouterTopology.TTS_SERVING:
        lines.extend(["", "[websocket.speech]", 'trust_domain = "local"'])
    return "\n".join(lines)


def worker_block(
    *,
    topology: CiRouterTopology,
    ordinal: int,
    worker_url: str,
    model_name: str,
    generation_streaming: bool,
    named_voice: bool,
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
        f"worker_id = {toml_string(worker_id)}",
        f"base_url = {toml_string(worker_url.rstrip('/') + '/')}",
        'trust_domain = "local"',
        f"default_model_id = {toml_string(model_name)}",
    ]
    if topology is CiRouterTopology.TTS_SERVING:
        lines.extend(
            [
                "",
                "[workers.capacity]",
                f"speech_websocket = {CI_ROUTER_MAX_INFLIGHT}",
            ]
        )
    lines.extend(
        [
            "",
            service_profiles(
                topology,
                model_name,
                named_voice,
                generation_streaming=generation_streaming,
            ),
        ]
    )
    return "\n".join(lines)


def service_profiles(
    topology: CiRouterTopology,
    model_name: str,
    named_voice: bool,
    *,
    generation_streaming: bool,
) -> str:
    model_ids = toml_array([model_name])
    if topology is CiRouterTopology.ASR:
        return transcription_profile(
            model_ids,
            task="transcribe",
            formats=["json", "verbose_json", "sse"],
        )
    if topology is CiRouterTopology.TTS:
        if named_voice:
            tasks = ["text_to_speech"]
            reference_forms = ["none"]
            voice_name_policy = "preset"
        else:
            tasks = ["voice_clone"]
            reference_forms = ["direct", "list"]
            voice_name_policy = "uploaded"
        return "\n\n".join(
            [
                speech_profile(
                    service="speech_http",
                    model_ids=model_ids,
                    response_formats=["wav"],
                    stream_modes=["non_streaming"],
                    tasks=tasks,
                    reference_forms=reference_forms,
                    voice_name_policy=voice_name_policy,
                ),
                speech_profile(
                    service="speech_http",
                    model_ids=model_ids,
                    response_formats=["pcm"],
                    stream_modes=["non_streaming", "streaming"],
                    tasks=tasks,
                    reference_forms=reference_forms,
                    voice_name_policy=voice_name_policy,
                ),
            ]
        )
    if topology is CiRouterTopology.TTS_SERVING:
        profiles = [
            speech_profile(
                service="speech_http",
                model_ids=model_ids,
                response_formats=["mp3", "opus", "aac", "flac", "wav"],
                stream_modes=["non_streaming"],
                tasks=["text_to_speech", "voice_clone", "voice_design"],
                reference_forms=["none", "direct", "list"],
                voice_name_policy="uploaded",
            ),
            speech_profile(
                service="speech_http",
                model_ids=model_ids,
                response_formats=["pcm"],
                stream_modes=["non_streaming", "streaming"],
                tasks=["text_to_speech", "voice_clone", "voice_design"],
                reference_forms=["none", "direct", "list"],
                voice_name_policy="uploaded",
            ),
            speech_batch_profile(model_ids),
            speech_profile(
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
    return generation_profile(
        model_ids=model_ids,
        audio_output=topology is CiRouterTopology.OMNI_AUDIO,
        streaming=generation_streaming,
    )


def transcription_profile(
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
            f"task = {toml_string(task)}",
            f"response_formats = {toml_array(formats)}",
            'stream_modes = ["non_streaming", "streaming"]',
        ]
    )


def speech_profile(
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
            f"service = {toml_string(service)}",
            f"model_ids = {model_ids}",
            f"response_formats = {toml_array(response_formats)}",
            f"stream_modes = {toml_array(stream_modes)}",
            f"tasks = {toml_array(tasks)}",
            f"reference_forms = {toml_array(reference_forms)}",
            f"voice_name_policy = {toml_string(voice_name_policy)}",
        ]
    )


def speech_batch_profile(model_ids: str) -> str:
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


def generation_profile(*, model_ids: str, audio_output: bool, streaming: bool) -> str:
    output_modalities = ["text", "audio"] if audio_output else ["text"]
    audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"] if audio_output else []
    stream_modes = ["non_streaming", "streaming"] if streaming else ["non_streaming"]
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            'service = "generation_http"',
            f"model_ids = {model_ids}",
            'message_content_forms = ["string", "typed_parts"]',
            'media_placements = ["top_level", "typed_parts"]',
            'input_modalities = ["text", "image", "audio", "video"]',
            f"output_modalities = {toml_array(output_modalities)}",
            f"chat_audio_formats = {toml_array(audio_formats)}",
            f"stream_modes = {toml_array(stream_modes)}",
        ]
    )


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def toml_array(values: list[str]) -> str:
    return "[" + ", ".join(toml_string(value) for value in values) + "]"
