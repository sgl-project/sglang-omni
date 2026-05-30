# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from argparse import Namespace

import pytest

from examples.encoder_tp_e2e_probe import (
    _summarize_gpu_process_samples,
    _summarize_gpu_samples,
    build_request,
    extract_response_text,
    summarize_runs,
)


def test_probe_builds_video_audio_request():
    req = build_request(
        Namespace(
            model="qwen3-omni",
            prompt="Describe the media.",
            video="/tmp/video.mp4",
            audio="/tmp/audio.wav",
            video_fps=None,
            video_max_frames=None,
            video_min_pixels=None,
            video_max_pixels=None,
            audio_no_truncation=False,
            max_tokens=64,
        )
    )

    assert req["model"] == "qwen3-omni"
    assert req["messages"] == [{"role": "user", "content": "Describe the media."}]
    assert req["videos"] == ["/tmp/video.mp4"]
    assert req["audios"] == ["/tmp/audio.wav"]
    assert req["modalities"] == ["text"]
    assert req["stream"] is False


def test_probe_rejects_empty_media_request():
    with pytest.raises(ValueError, match="at least one"):
        build_request(
            Namespace(
                model="qwen3-omni",
                prompt="Describe the media.",
                video=None,
                audio=None,
                video_fps=None,
                video_max_frames=None,
                video_min_pixels=None,
                video_max_pixels=None,
                audio_no_truncation=False,
                max_tokens=64,
            )
        )


def test_probe_includes_video_sampling_overrides():
    req = build_request(
        Namespace(
            model="qwen3-omni",
            prompt="Describe the media.",
            video="/tmp/video.mp4",
            audio=None,
            video_fps=2.0,
            video_max_frames=128,
            video_min_pixels=100352,
            video_max_pixels=401408,
            audio_no_truncation=False,
            max_tokens=64,
        )
    )

    assert req["video_fps"] == 2.0
    assert req["video_max_frames"] == 128
    assert req["video_min_pixels"] == 100352
    assert req["video_max_pixels"] == 401408


def test_probe_includes_audio_no_truncation_override():
    req = build_request(
        Namespace(
            model="qwen3-omni",
            prompt="Describe the audio.",
            video=None,
            audio="/tmp/audio.wav",
            video_fps=None,
            video_max_frames=None,
            video_min_pixels=None,
            video_max_pixels=None,
            audio_no_truncation=True,
            max_tokens=64,
        )
    )

    assert req["audio_truncation"] is False


def test_probe_extracts_text_from_chat_completion_shapes():
    assert (
        extract_response_text(
            {"choices": [{"message": {"content": "plain response"}}]}
        )
        == "plain response"
    )
    assert (
        extract_response_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "hello "},
                                {"type": "text", "text": "world"},
                            ]
                        }
                    }
                ]
            }
        )
        == "hello world"
    )
    assert extract_response_text({"error": "boom"}) == ""


def test_probe_summarizes_only_measured_successful_latencies():
    summary = summarize_runs(
        [
            {"warmup": True, "ok": True, "latency_s": 10.0},
            {
                "warmup": False,
                "ok": True,
                "latency_s": 2.0,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
            {
                "warmup": False,
                "ok": True,
                "latency_s": 4.0,
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            },
            {"warmup": False, "ok": False, "latency_s": 100.0},
        ]
    )

    assert summary["total_runs"] == 4
    assert summary["measured_runs"] == 3
    assert summary["successes"] == 2
    assert summary["failures"] == 1
    assert summary["latency_s_mean"] == 3.0
    assert summary["latency_s_p50"] == 3.0
    assert summary["latency_s_p90"] == pytest.approx(3.8)
    assert summary["latency_s_p95"] == pytest.approx(3.9)
    assert summary["latency_s_max"] == 4.0
    assert summary["prompt_tokens_mean"] == 15
    assert summary["total_tokens_max"] == 28
    assert summary["completion_tokens_per_s_mean"] == 2.0


def test_probe_summarizes_gpu_samples_by_peak():
    summary = _summarize_gpu_samples(
        [
            {
                "ok": True,
                "gpus": [
                    {
                        "index": 0,
                        "name": "GPU0",
                        "uuid": "GPU-0",
                        "memory_used_mib": 10,
                        "memory_free_mib": 90,
                        "utilization_gpu_percent": 5,
                    }
                ],
            },
            {
                "ok": True,
                "gpus": [
                    {
                        "index": 0,
                        "name": "GPU0",
                        "uuid": "GPU-0",
                        "memory_used_mib": 15,
                        "memory_free_mib": 85,
                        "utilization_gpu_percent": 70,
                    }
                ],
            },
            {"ok": False, "result": {"error": "nvidia-smi failed"}},
        ]
    )

    assert summary["sample_count"] == 3
    assert summary["gpus"] == [
        {
            "index": 0,
            "name": "GPU0",
            "uuid": "GPU-0",
            "initial_memory_used_mib": 10,
            "initial_memory_free_mib": 90,
            "final_memory_used_mib": 15,
            "final_memory_free_mib": 85,
            "max_memory_used_mib": 15,
            "max_memory_delta_mib": 5,
            "min_memory_free_mib": 85,
            "max_utilization_gpu_percent": 70,
            "samples": 2,
        }
    ]


def test_probe_summarizes_gpu_process_samples_by_peak():
    summary = _summarize_gpu_process_samples(
        [
            {
                "ok": True,
                "processes": [
                    {
                        "pid": 123,
                        "process_name": "python",
                        "gpu_uuid": "GPU-0",
                        "used_memory_mib": 100,
                    },
                    {
                        "pid": 999,
                        "process_name": "[Not Found]",
                        "gpu_uuid": "GPU-1",
                        "used_memory_mib": 25,
                    },
                ],
            },
            {
                "ok": True,
                "processes": [
                    {
                        "pid": 123,
                        "process_name": "python",
                        "gpu_uuid": "GPU-0",
                        "used_memory_mib": 140,
                    },
                    {
                        "pid": 123,
                        "process_name": "python",
                        "gpu_uuid": "GPU-1",
                        "used_memory_mib": 20,
                    },
                ],
            },
            {"ok": False, "processes": []},
        ]
    )

    assert summary["sample_count"] == 3
    assert summary["processes"] == [
        {
            "gpu_uuid": "GPU-0",
            "pid": 123,
            "process_name": "python",
            "initial_used_memory_mib": 100,
            "final_used_memory_mib": 140,
            "max_used_memory_mib": 140,
            "max_memory_delta_mib": 40,
            "samples": 2,
        },
        {
            "gpu_uuid": "GPU-1",
            "pid": 123,
            "process_name": "python",
            "initial_used_memory_mib": 20,
            "final_used_memory_mib": 20,
            "max_used_memory_mib": 20,
            "max_memory_delta_mib": 0,
            "samples": 1,
        },
        {
            "gpu_uuid": "GPU-1",
            "pid": 999,
            "process_name": "[Not Found]",
            "initial_used_memory_mib": 25,
            "final_used_memory_mib": 25,
            "max_used_memory_mib": 25,
            "max_memory_delta_mib": 0,
            "samples": 1,
        },
    ]
