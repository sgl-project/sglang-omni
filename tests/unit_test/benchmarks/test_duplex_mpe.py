# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Duplex-MPE dataset, causal VAD, reactive scheduler, and scoring."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import http.server
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile
import torch
import websockets
from numpy.typing import NDArray
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection

import benchmarks.duplex.mpe.vad as duplex_mpe_vad
from benchmarks.dataset.duplex_mpe import (
    DUPLEX_MPE_DATASET_ID,
    DUPLEX_MPE_DATASET_REVISION,
    DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH,
    DUPLEX_MPE_METADATA_SHA256,
    DuplexMpeScenarioRecord,
    compute_file_sha256,
    download_duplex_mpe_dataset,
    load_duplex_mpe_scenarios,
    resolve_dataset_audio_path,
    resolve_retained_scenario_audio_path,
    validate_duplex_mpe_dataset,
    write_json_atomic,
)
from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.duplex.mpe.metrics import (
    score_scenario_reports,
    validate_scenario_reports,
)
from benchmarks.duplex.mpe.prompts import compute_prompt_sha256
from benchmarks.duplex.mpe.protocol import (
    TARGET_CONTINUATION_OUTCOME,
    TARGET_HIT_OUTCOME,
    TARGET_MISS_OUTCOME,
    TARGET_OVERLONG_OUTCOME,
    AudioFeedAdapter,
    ClockedAudioAdapter,
    ScenarioTurnInput,
    run_scenario_schedule,
)
from benchmarks.duplex.mpe.realtime_session import (
    PlaybackTimelineBuffer,
    RealtimeEndpointSpec,
    RealtimeProtocolMode,
    decode_pcm16_base64_to_float32,
    encode_float32_to_pcm16_base64,
    negotiate_realtime_endpoint,
    run_realtime_scenario,
    validate_accepted_session,
)
from benchmarks.duplex.mpe.semantic import (
    JudgeEndpointConfig,
    build_semantic_tasks,
    judge_tasks_via_http,
    public_judge_metadata,
    transcribe_tasks_via_http,
)
from benchmarks.duplex.mpe.vad import SileroActivityConfig, SileroActivityDetector
from benchmarks.eval.benchmark_duplex_mpe import DuplexMpeEvalConfig
from benchmarks.eval.benchmark_duplex_mpe import main as run_duplex_mpe_cli
from benchmarks.eval.benchmark_duplex_mpe import (
    run_duplex_mpe_evaluation,
    run_generate_phase,
    sanitize_launch_command,
)


class EnvelopeVadModel:
    """Deterministic RMS-envelope SileroVadModel double for offline unit tests."""

    def __init__(self, rms_threshold: float = 0.05) -> None:
        self.rms_threshold: float = rms_threshold
        self.reset_count: int = 0

    def reset_states(self) -> None:
        self.reset_count += 1

    def __call__(self, audio_tensor: torch.Tensor, sample_rate_hz: int) -> torch.Tensor:
        if sample_rate_hz != 16000:
            raise ValueError(f"Expected 16000 Hz Silero input, got {sample_rate_hz}")
        else:
            rms_amplitude = float(torch.sqrt(torch.mean(audio_tensor.float() ** 2)))
            probability = 0.9 if rms_amplitude >= self.rms_threshold else 0.05
            return torch.tensor([probability], dtype=torch.float32)


class ScriptedReplyAdapter:
    """Deterministic AudioFeedAdapter emitting voiced tone intervals on the output clock."""

    def __init__(
        self,
        *,
        input_sample_rate_hz: int,
        output_sample_rate_hz: int,
        voiced_windows_seconds: list[tuple[float, float]],
    ) -> None:
        self.input_sample_rate_hz: int = input_sample_rate_hz
        self.output_sample_rate_hz: int = output_sample_rate_hz
        self.voiced_windows_seconds: list[tuple[float, float]] = voiced_windows_seconds
        self.fed_input_samples: int = 0
        self.emitted_output_samples: int = 0

    def feed(self, input_samples: NDArray[np.float32]) -> NDArray[np.float32]:
        self.fed_input_samples += len(input_samples)
        target_output_total = int(
            round(
                self.fed_input_samples
                * self.output_sample_rate_hz
                / self.input_sample_rate_hz
            )
        )
        chunk_length = target_output_total - self.emitted_output_samples
        output = np.zeros(chunk_length, dtype=np.float32)
        if chunk_length > 0:
            sample_indices = np.arange(
                self.emitted_output_samples, target_output_total, dtype=np.float64
            )
            timestamps = sample_indices / self.output_sample_rate_hz
            voiced_mask = np.zeros(chunk_length, dtype=bool)
            for window_start, window_end in self.voiced_windows_seconds:
                voiced_mask |= (timestamps >= window_start) & (timestamps < window_end)
            tone = (0.35 * np.sin(2.0 * np.pi * 220.0 * timestamps)).astype(np.float32)
            output[voiced_mask] = tone[voiced_mask]
        else:
            pass
        self.emitted_output_samples = target_output_total
        return output


def write_synthetic_wav(
    file_path: Path,
    *,
    duration_seconds: float,
    sample_rate_hz: int = 24000,
    amplitude: float = 0.25,
) -> dict[str, JsonValue]:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(duration_seconds * sample_rate_hz))
    timeline = np.arange(frame_count, dtype=np.float64) / sample_rate_hz
    samples = (amplitude * np.sin(2.0 * np.pi * 260.0 * timeline)).astype(np.float32)
    soundfile.write(
        str(file_path), samples, sample_rate_hz, format="WAV", subtype="PCM_16"
    )
    return {
        "path": "",
        "sha256": compute_file_sha256(file_path),
        "bytes": file_path.stat().st_size,
        "sample_rate": sample_rate_hz,
        "frames": frame_count,
        "channels": 1,
        "duration_s": round(frame_count / sample_rate_hz, 4),
    }


def build_synthetic_dataset(root: Path, *, pair_count: int = 22) -> Path:
    preamble_relative_path = DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH
    preamble_metadata = write_synthetic_wav(
        root / preamble_relative_path, duration_seconds=0.32
    )
    preamble_metadata["path"] = preamble_relative_path

    clip_relative_path = "audio/aa/clip_turn.wav"
    clip_metadata = write_synthetic_wav(
        root / clip_relative_path, duration_seconds=0.32
    )
    clip_metadata["path"] = clip_relative_path

    explicit_rows: list[dict[str, JsonValue]] = []
    implicit_rows: list[dict[str, JsonValue]] = []
    for index in range(pair_count):
        pair_id = f"duplex_mpe_{index:04d}"
        for condition, rows in (
            ("explicit", explicit_rows),
            ("implicit", implicit_rows),
        ):
            rows.append(
                {
                    "scenario_id": f"{pair_id}_{condition}",
                    "pair_id": pair_id,
                    "addressing": condition,
                    "trigger_index": 1,
                    "supporting_indices": [0],
                    "gold_answer": "Nine dollars per person at the taco shop.",
                    "turns": [
                        {
                            "turn_index": 0,
                            "event_type": "N2",
                            "speaker": "A",
                            "text": "Tacos on Fifth are nine dollars each.",
                            "audio_path": clip_relative_path,
                        },
                        {
                            "turn_index": 1,
                            "event_type": "T",
                            "speaker": "B",
                            "text": "Aria, how much are the tacos per person?",
                            "audio_path": clip_relative_path,
                        },
                        {
                            "turn_index": 2,
                            "event_type": "N4_Q",
                            "speaker": "C",
                            "text": "Aria, when does the bus leave?",
                            "audio_path": clip_relative_path,
                        },
                        {
                            "turn_index": 3,
                            "event_type": "N4_R",
                            "speaker": "A",
                            "text": "Never mind Aria, I see it leaves at eight.",
                            "audio_path": clip_relative_path,
                        },
                    ],
                }
            )

    write_json_atomic(root / "dataset.json", {"dataset": "Duplex-MPE"})
    write_json_atomic(root / "manifests" / "explicit.json", explicit_rows)
    write_json_atomic(root / "manifests" / "implicit.json", implicit_rows)
    write_json_atomic(root / "audio_inventory.json", [preamble_metadata, clip_metadata])
    return root


def test_dataset_registry_pinning_and_media_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert DATASETS["duplex-mpe"] == DUPLEX_MPE_DATASET_ID
    assert DATASETS["duplex-mpe-mini"] == f"{DUPLEX_MPE_DATASET_ID}:mini"
    assert set(DUPLEX_MPE_METADATA_SHA256) == {
        "dataset.json",
        "manifests/explicit.json",
        "manifests/implicit.json",
        "audio_inventory.json",
        DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH,
    }

    with pytest.raises(ValueError, match=DUPLEX_MPE_DATASET_REVISION):
        download_dataset(DUPLEX_MPE_DATASET_ID, revision="unpinned-branch")
    with pytest.raises(ValueError, match="Unsupported Duplex-MPE subset"):
        download_dataset(f"{DUPLEX_MPE_DATASET_ID}:unknown_split")

    dataset_root = build_synthetic_dataset(tmp_path / "dataset", pair_count=25)
    for unsafe_path in (
        "../escape.wav",
        "/etc/passwd",
        "audio\\bad.wav",
        "audio/./x.wav",
    ):
        with pytest.raises(ValueError):
            resolve_dataset_audio_path(dataset_root, unsafe_path)

    full_scenarios = load_duplex_mpe_scenarios(dataset_root, addressing="all")
    mini_scenarios = load_duplex_mpe_scenarios(
        dataset_root, addressing="all", mini=True
    )
    assert len(full_scenarios) == 50
    assert len(mini_scenarios) == 40

    validation_summary = validate_duplex_mpe_dataset(
        dataset_root, mini=True, verify_audio_checksums=True
    )
    assert validation_summary["scenarios"] == 40
    assert validation_summary["pairs"] == 20

    monkeypatch.setattr(
        "benchmarks.dataset.duplex_mpe.snapshot_download",
        lambda **kwargs: str(dataset_root),
    )
    with pytest.raises(ValueError, match="SHA-256 digests do not match"):
        download_duplex_mpe_dataset(destination_dir=dataset_root, mini=True, quiet=True)


@pytest.mark.parametrize("sample_rate_hz", [16000, 24000, 22050])
def test_vad_episode_hangover_and_isolated_session_state(
    sample_rate_hz: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pauses under 800 ms merge into one episode, endpoint waits 2000 ms, and detectors own separate RNN state."""
    detector = SileroActivityDetector(
        vad_config=SileroActivityConfig(),
        sample_rate_hz=sample_rate_hz,
        vad_model=EnvelopeVadModel(),
    )

    def make_tone(seconds: float) -> NDArray[np.float32]:
        count = int(round(seconds * sample_rate_hz))
        timeline = np.arange(count, dtype=np.float64) / sample_rate_hz
        return (0.35 * np.sin(2.0 * np.pi * 240.0 * timeline)).astype(np.float32)

    def make_silence(seconds: float) -> NDArray[np.float32]:
        return np.zeros(int(round(seconds * sample_rate_hz)), dtype=np.float32)

    detector.push(make_silence(0.32))
    detector.push(make_tone(0.48))
    detector.push(make_silence(0.48))
    assert detector.is_speaking is True
    assert detector.is_endpoint_confirmed(2000.0) is False

    detector.push(make_tone(0.48))
    detector.push(make_silence(1.00))
    assert len(detector.episodes) == 1
    assert detector.is_endpoint_confirmed(2000.0) is False

    detector.push(make_silence(1.15))
    assert detector.is_endpoint_confirmed(2000.0) is True

    class StatefulCounterModel:
        def __init__(self) -> None:
            self.active_frames: int = 0

        def reset_states(self) -> None:
            self.active_frames = 0

        def __call__(self, audio_tensor: torch.Tensor, rate_hz: int) -> torch.Tensor:
            if float(audio_tensor.abs().mean()) > 0.1:
                self.active_frames += 1
            else:
                self.active_frames = 0
            return torch.tensor(
                [0.9 if self.active_frames >= 4 else 0.05], dtype=torch.float32
            )

    monkeypatch.setattr(duplex_mpe_vad, "load_silero_vad", StatefulCounterModel)
    session_a = SileroActivityDetector(sample_rate_hz=16000)
    session_a.push(np.ones(4 * 512, dtype=np.float32))
    session_b = SileroActivityDetector(sample_rate_hz=16000)
    session_b.push(np.zeros(4 * 512, dtype=np.float32))
    session_a.push(np.ones(3 * 512, dtype=np.float32))
    assert session_a.is_voicing_now is True
    assert session_b.is_voicing_now is False


@pytest.mark.parametrize(
    (
        "voiced_windows",
        "maximum_response_seconds",
        "expected_outcome",
        "expected_eligible",
        "expected_yielded",
    ),
    [
        ([], 60.0, TARGET_MISS_OUTCOME, 0, 0),
        ([(1.70, 2.34), (5.50, 8.60)], 60.0, TARGET_HIT_OUTCOME, 1, 1),
        ([(1.10, 1.84), (5.00, 9.80)], 60.0, TARGET_CONTINUATION_OUTCOME, 1, 0),
        ([(1.70, 6.00)], 2.0, TARGET_OVERLONG_OUTCOME, 0, 0),
    ],
)
def test_reactive_schedule_outcomes_and_n4_yield_boundary(
    voiced_windows: list[tuple[float, float]],
    maximum_response_seconds: float,
    expected_outcome: str,
    expected_eligible: int,
    expected_yielded: int,
) -> None:
    clip = (0.3 * np.sin(2.0 * np.pi * 200.0 * np.arange(16000 // 2) / 16000.0)).astype(
        np.float32
    )
    resolution_clip = (
        0.3 * np.sin(2.0 * np.pi * 200.0 * np.arange(16000 * 5 // 2) / 16000.0)
    ).astype(np.float32)
    turns = [
        ScenarioTurnInput(
            turn_index=0, event_type="N2", speaker="A", waveform_samples=clip
        ),
        ScenarioTurnInput(
            turn_index=1, event_type="T", speaker="B", waveform_samples=clip
        ),
        ScenarioTurnInput(
            turn_index=2, event_type="N4_Q", speaker="C", waveform_samples=clip
        ),
        ScenarioTurnInput(
            turn_index=3,
            event_type="N4_R",
            speaker="A",
            waveform_samples=resolution_clip,
        ),
    ]
    adapter: AudioFeedAdapter = ClockedAudioAdapter(
        ScriptedReplyAdapter(
            input_sample_rate_hz=16000,
            output_sample_rate_hz=24000,
            voiced_windows_seconds=voiced_windows,
        )
    )
    schedule_result = run_scenario_schedule(
        adapter,
        turns,
        vad_model=EnvelopeVadModel(),
        maximum_response_seconds=maximum_response_seconds,
        collect_audio=True,
    )
    report = schedule_result.to_report_dict()
    assert schedule_result.target_outcome == expected_outcome

    scenario_record = DuplexMpeScenarioRecord(
        scenario_id="duplex_mpe_0000_explicit",
        pair_id="duplex_mpe_0000",
        addressing="explicit",
        trigger_index=1,
        supporting_indices=[0],
        gold_answer="Nine dollars.",
        turns=[
            {
                "turn_index": turn.turn_index,
                "event_type": turn.event_type,
                "speaker": turn.speaker,
                "text": "synthetic turn",
                "audio_path": "audio/aa/clip.wav",
            }
            for turn in turns
        ],
    )
    verdicts: list[dict[str, JsonValue]] = []
    if expected_outcome != TARGET_MISS_OUTCOME:
        verdicts.append(
            {
                "scenario_id": "duplex_mpe_0000_explicit",
                "turn_index": 1,
                "kind": "answer",
                "correct": True,
            }
        )
    else:
        pass
    scored = score_scenario_reports(
        {"duplex_mpe_0000_explicit": report},
        {"duplex_mpe_0000_explicit": scenario_record},
        verdicts,
    )
    ay_cell = scored["explicit"]["answering_window_yield"]
    assert isinstance(ay_cell, dict)
    assert ay_cell["denominator"] == expected_eligible
    assert ay_cell["numerator"] == expected_yielded


def test_pcm16_lossless_round_trip_and_capabilities_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain_int16 = np.arange(-32768, 32768, dtype=np.int32).astype("<i2")
    encoded_domain = base64.b64encode(domain_int16.tobytes()).decode("ascii")
    decoded_float = decode_pcm16_base64_to_float32(encoded_domain)
    reencoded = encode_float32_to_pcm16_base64(decoded_float)
    restored_int16 = np.frombuffer(base64.b64decode(reencoded), dtype="<i2")
    np.testing.assert_array_equal(restored_int16, domain_int16)

    original_async_client = httpx.AsyncClient

    def make_404_client(
        *, timeout: float = 30.0, trust_env: bool = False
    ) -> httpx.AsyncClient:
        return original_async_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(404, json={"detail": "not found"})
            ),
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr(httpx, "AsyncClient", make_404_client)

    # note (machengqian): WebSocket capability probe and session connections ignore proxy env vars when trust_env=False.
    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "WS_PROXY",
        "WSS_PROXY",
        "ALL_PROXY",
    ):
        monkeypatch.setenv(proxy_var, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")

    async def verify_ws_fallback() -> RealtimeEndpointSpec:
        async def probe_peer(connection: ServerConnection) -> None:
            await connection.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "event_id": "evt_probe",
                        "session": {
                            "model": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 16000}
                                },
                                "output": {
                                    "format": {"type": "audio/pcm", "rate": 22050}
                                },
                            },
                            "sglang": {
                                "capabilities": {
                                    "interaction": "native",
                                    "native_full_duplex": True,
                                }
                            },
                        },
                    }
                )
            )
            async for raw_frame in connection:
                client_msg = json.loads(raw_frame)
                if client_msg["type"] == "session.close":
                    return
                else:
                    pass

        async with websockets.serve(probe_peer, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await negotiate_realtime_endpoint(f"http://127.0.0.1:{port}")

    negotiated = asyncio.run(verify_ws_fallback())
    assert negotiated.protocol == "native"
    assert negotiated.input_sample_rate_hz == 16000
    assert negotiated.output_sample_rate_hz == 22050
    assert negotiated.reported_model == "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"


@pytest.mark.parametrize(
    ("protocol_mode", "input_rate_hz", "output_rate_hz"),
    [
        ("native", 16000, 24000),
        ("native", 24000, 24000),
        ("native", 16000, 22050),
        ("turn_based", 16000, 24000),
        ("legacy", 16000, 24000),
    ],
)
def test_realtime_websocket_session_contracts_and_drain_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol_mode: RealtimeProtocolMode,
    input_rate_hz: int,
    output_rate_hz: int,
) -> None:
    """Verify /v1/realtime streaming, turn_based truncation, legacy probe, and drain-tail capture."""
    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "WS_PROXY",
        "WSS_PROXY",
        "ALL_PROXY",
    ):
        monkeypatch.setenv(proxy_var, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    dataset_root = build_synthetic_dataset(tmp_path / "ds", pair_count=1)
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit")[0]

    async def exercise_session() -> tuple[dict[str, JsonValue], Path]:
        async def peer_handler(connection: ServerConnection) -> None:
            await connection.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "event_id": "evt_created",
                        "session": {"id": "sess_1"},
                    }
                )
            )
            accumulated_input_seconds = 0.0
            replied_to_target = False
            emitted_n4_cancellation = False
            delta_type = (
                "response.audio.delta"
                if protocol_mode == "legacy"
                else "response.output_audio.delta"
            )
            async for raw_message in connection:
                client_event = json.loads(raw_message)
                event_type = client_event["type"]
                if event_type == "session.update":
                    session_cfg = client_event["session"]
                    echoed_session: dict[str, JsonValue]
                    if not session_cfg:
                        echoed_session = {}
                    elif protocol_mode == "legacy":
                        echoed_session = {
                            "modalities": ["text", "audio"],
                            "turn_detection": {"type": "server_vad"},
                        }
                    else:
                        echoed_session = {
                            "audio": {
                                "input": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": input_rate_hz,
                                    },
                                    "turn_detection": {"type": "server_vad"},
                                },
                                "output": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": output_rate_hz,
                                    }
                                },
                            }
                        }
                    await connection.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "event_id": "evt_updated",
                                "session": echoed_session,
                            }
                        )
                    )
                    if not session_cfg and protocol_mode == "legacy":
                        return
                    else:
                        pass
                elif event_type == "input_audio_buffer.append":
                    pcm_samples = decode_pcm16_base64_to_float32(client_event["audio"])
                    accumulated_input_seconds += len(pcm_samples) / input_rate_hz
                    if not replied_to_target and accumulated_input_seconds >= 2.10:
                        replied_to_target = True
                        tone_count = int(round(0.48 * output_rate_hz))
                        timeline = (
                            np.arange(tone_count, dtype=np.float64) / output_rate_hz
                        )
                        tone = (0.35 * np.sin(2.0 * np.pi * 240.0 * timeline)).astype(
                            np.float32
                        )
                        await connection.send(
                            json.dumps(
                                {
                                    "type": delta_type,
                                    "event_id": "evt_audio",
                                    "delta": encode_float32_to_pcm16_base64(tone),
                                }
                            )
                        )
                    elif (
                        protocol_mode == "turn_based"
                        and not emitted_n4_cancellation
                        and accumulated_input_seconds >= 5.20
                    ):
                        emitted_n4_cancellation = True
                        extra_tone = np.full(
                            int(round(0.40 * output_rate_hz)), 0.25, dtype=np.float32
                        )
                        await connection.send(
                            json.dumps(
                                {
                                    "type": delta_type,
                                    "event_id": "evt_extra",
                                    "delta": encode_float32_to_pcm16_base64(extra_tone),
                                }
                            )
                        )
                        await connection.send(
                            json.dumps(
                                {
                                    "type": "response.done",
                                    "event_id": "evt_cancelled",
                                    "response": {"status": "cancelled"},
                                }
                            )
                        )
                    else:
                        pass
                elif event_type == "sglang.input_audio.end":
                    tail_tone = np.full(
                        int(round(0.10 * output_rate_hz)), 0.20, dtype=np.float32
                    )
                    await connection.send(
                        json.dumps(
                            {
                                "type": delta_type,
                                "event_id": "evt_tail",
                                "delta": encode_float32_to_pcm16_base64(tail_tone),
                            }
                        )
                    )
                    await connection.send(
                        json.dumps(
                            {
                                "type": "sglang.input_audio.drained",
                                "event_id": "evt_drained",
                            }
                        )
                    )
                elif event_type == "session.close":
                    await connection.send(
                        json.dumps({"type": "session.closed", "event_id": "evt_closed"})
                    )
                    return
                else:
                    pass

        async with websockets.serve(peer_handler, "127.0.0.1", 0) as server:
            bound_port = server.sockets[0].getsockname()[1]
            turn_detection = (
                None
                if protocol_mode == "native"
                else {"type": "server_vad", "silence_duration_ms": 500}
            )
            endpoint_spec = RealtimeEndpointSpec(
                websocket_url=f"ws://127.0.0.1:{bound_port}/v1/realtime",
                capabilities_url=f"http://127.0.0.1:{bound_port}/v1/realtime/capabilities",
                protocol=protocol_mode,
                input_sample_rate_hz=input_rate_hz,
                output_sample_rate_hz=output_rate_hz,
                turn_detection=turn_detection,
            )
            wav_path = tmp_path / "out" / f"{scenario.scenario_id}.wav"
            report = await run_realtime_scenario(
                scenario,
                dataset_root=dataset_root,
                endpoint_spec=endpoint_spec,
                output_audio_path=wav_path,
                vad_model=EnvelopeVadModel(),
                realtime_pacing=False,
            )
            return report, wav_path

    report, wav_path = asyncio.run(exercise_session())
    validate_scenario_reports(
        {scenario.scenario_id: report}, {scenario.scenario_id: scenario}
    )
    target_block = report["T"]
    assert isinstance(target_block, dict)
    assert target_block["outcome"] == TARGET_HIT_OUTCOME
    assert report["seed_policy"] == "reference_seed_metadata_only"
    assert report["server_seed_applied"] is False
    if protocol_mode == "turn_based":
        cancelled_samples = report["cancelled_unplayed_samples"]
        assert type(cancelled_samples) is int and cancelled_samples > 0
    else:
        pass
    if protocol_mode != "legacy":
        drain_tail = report["drain_tail_samples"]
        assert type(drain_tail) is int and drain_tail == int(
            round(0.10 * output_rate_hz)
        )
        assert report["assistant_audio_sha256"] == compute_file_sha256(wav_path)
    else:
        pass


def test_semantic_checkpointing_resume_guards_and_cli_completeness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = build_synthetic_dataset(tmp_path / "ds", pair_count=1)
    scenarios = load_duplex_mpe_scenarios(dataset_root, addressing="all")
    scenarios_by_id = {row.scenario_id: row for row in scenarios}

    audio_dir = tmp_path / "run" / "audio"
    audio_dir.mkdir(parents=True)
    reports: dict[str, dict[str, JsonValue]] = {}
    for index, scenario in enumerate(scenarios):
        wav_path = audio_dir / f"{scenario.scenario_id}.wav"
        tone = (
            0.3 * np.sin(2.0 * np.pi * 220.0 * np.arange(24000 * 8) / 24000.0)
        ).astype(np.float32)
        soundfile.write(str(wav_path), tone, 24000, format="WAV", subtype="FLOAT")
        audio_sha256 = compute_file_sha256(wav_path)
        answer_end = 2.60 if index == 0 else 2.10
        reports[scenario.scenario_id] = {
            "protocol_id": "mpeval.unified.v3",
            "aed_config": SileroActivityConfig().to_report_dict(),
            "total_duration_s": 8.0,
            "assistant_speech_segments": [[0.10, 0.50], [2.00, answer_end]],
            "assistant_audio_sha256": audio_sha256,
            "T": {
                "outcome": TARGET_HIT_OUTCOME,
                "t_end_s": 1.96,
                "onset_s": 2.00,
                "onset_latency_s": 0.04,
                "endpoint_s": answer_end,
                "response_duration_s": round(answer_end - 2.00, 4),
                "forced_floor_return": False,
                "speaking_at_t_end": False,
            },
            "rows": [
                {
                    "turn_index": 0,
                    "event_type": "N2",
                    "speaker": "A",
                    "turn_start": 0.0,
                    "turn_end": 0.32,
                    "win_start": 0.0,
                    "win_end": 0.82,
                },
                {
                    "turn_index": 1,
                    "event_type": "T",
                    "speaker": "B",
                    "turn_start": 0.82,
                    "turn_end": 1.96,
                    "win_start": 0.82,
                    "win_end": 4.60,
                },
                {
                    "turn_index": 2,
                    "event_type": "N4_Q",
                    "speaker": "C",
                    "turn_start": 4.60,
                    "turn_end": 4.92,
                    "win_start": 4.60,
                    "win_end": 7.92,
                },
                {
                    "turn_index": 3,
                    "event_type": "N4_R",
                    "speaker": "A",
                    "turn_start": 7.92,
                    "turn_end": 8.00,
                    "win_start": 7.92,
                    "win_end": 8.00,
                },
            ],
        }

    tasks = build_semantic_tasks(reports, scenarios_by_id)
    assert len(tasks) == 4

    judge_config = JudgeEndpointConfig(
        model="qwen3.8-27b",
        base_url="http://127.0.0.1:30000/v1/secret-route",
        api_key_env="DUPLEX_MPE_TEST_JUDGE_KEY",
        max_concurrency=2,
    )
    monkeypatch.setenv("DUPLEX_MPE_TEST_JUDGE_KEY", "test-placeholder-not-saved")
    redacted = public_judge_metadata(judge_config)
    assert redacted["base_url"] == "http://127.0.0.1:30000"
    assert "test-placeholder-not-saved" not in json.dumps(redacted)

    asr_calls: list[str] = []
    judge_bodies: list[dict[str, JsonValue]] = []

    def handle_mock_endpoints(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/audio/transcriptions"):
            asr_calls.append(request.url.path)
            if len(asr_calls) in (1, 3):
                return httpx.Response(200, json={"text": "Mm-hmm."})
            else:
                return httpx.Response(
                    200,
                    json={"text": "The tacos on Fifth cost nine dollars per person."},
                )
        else:
            body = json.loads(request.content.decode("utf-8"))
            judge_bodies.append(body)
            user_prompt = body["messages"][0]["content"]
            if "BACKCHANNEL" in user_prompt:
                content = json.dumps({"label": "BACKCHANNEL", "reason": "phatic"})
            else:
                content = json.dumps(
                    {"correct": True, "reason": "matches nine dollars"}
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": content}}
                    ]
                },
            )

    original_async_client = httpx.AsyncClient

    def make_mock_async_client(
        *, timeout: float = 180.0, trust_env: bool = False
    ) -> httpx.AsyncClient:
        return original_async_client(
            transport=httpx.MockTransport(handle_mock_endpoints),
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr(httpx, "AsyncClient", make_mock_async_client)
    transcribed = asyncio.run(
        transcribe_tasks_via_http(
            tasks,
            audio_directory=audio_dir,
            output_path=tmp_path / "run" / "transcribed.json",
            asr_base_url="http://127.0.0.1:8001",
        )
    )
    assert len(asr_calls) == 3

    labels_path = tmp_path / "run" / "labels.json"
    verdicts = asyncio.run(
        judge_tasks_via_http(
            transcribed,
            judge_config=judge_config,
            output_path=labels_path,
        )
    )
    assert len(verdicts) == 4
    # note (machengqian): Reference answer requests omit temperature while silence requests set temperature=0.0.
    answer_bodies = [
        body for body in judge_bodies if "BACKCHANNEL" not in str(body["messages"])
    ]
    silence_bodies = [
        body for body in judge_bodies if "BACKCHANNEL" in str(body["messages"])
    ]
    assert len(answer_bodies) == 1 and "temperature" not in answer_bodies[0]
    assert len(silence_bodies) == 2 and silence_bodies[0]["temperature"] == 0.0

    # note (machengqian): Changing judge_model or base_url on the same labels file must raise ValueError.
    with pytest.raises(ValueError, match="produced by judge"):
        asyncio.run(
            judge_tasks_via_http(
                transcribed,
                judge_config=JudgeEndpointConfig(
                    model="different-judge",
                    base_url="http://127.0.0.1:30000/v1/secret-route",
                ),
                output_path=labels_path,
            )
        )
    with pytest.raises(ValueError, match="different judge endpoint"):
        asyncio.run(
            judge_tasks_via_http(
                transcribed,
                judge_config=JudgeEndpointConfig(
                    model="qwen3.8-27b",
                    base_url="http://127.0.0.1:30001/v1/other-route",
                ),
                output_path=labels_path,
            )
        )

    for row in verdicts:
        if row.get("error_kind") != "asr_empty":
            matching_task = next(
                candidate
                for candidate in transcribed
                if candidate["scenario_id"] == row["scenario_id"]
                and candidate["turn_index"] == row["turn_index"]
                and candidate["kind"] == row["kind"]
            )
            assert row["prompt_sha256"] == compute_prompt_sha256(matching_task)
        else:
            pass

    # note (machengqian): Missing retained assistant audio fails offline score unless audio is present or verification is disabled.
    first_id = scenarios[0].scenario_id
    partial_dir = tmp_path / "partial_run"
    write_json_atomic(partial_dir / "report.json", {first_id: reports[first_id]})
    with pytest.raises(FileNotFoundError, match="Retained assistant audio"):
        asyncio.run(
            run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=partial_dir,
                    phase="score",
                    addressing="all",
                )
            )
        )
    (partial_dir / "audio").mkdir(parents=True)
    (partial_dir / "audio" / f"{first_id}.wav").write_bytes(
        (audio_dir / f"{first_id}.wav").read_bytes()
    )
    partial_summary = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=partial_dir,
                phase="score",
                addressing="all",
            )
        )
    )
    assert partial_summary["status"] == "incomplete"
    assert partial_summary["missing_scenarios"] == [scenarios[1].scenario_id]

    # note (machengqian): Complete report plus verified transcripts and verdicts reports status == "complete".
    write_json_atomic(tmp_path / "run" / "report.json", reports)
    exit_code = run_duplex_mpe_cli(
        [
            "--phase",
            "score",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(tmp_path / "run"),
            "--model",
            "openbmb/MiniCPM-o-4_5",
        ]
    )
    assert exit_code == 0
    cli_summary = json.loads(
        (tmp_path / "run" / "duplex_mpe_results.json").read_text(encoding="utf-8")
    )
    assert cli_summary["status"] == "complete"

    # note (machengqian): Offline subset scoring (--addressing explicit) succeeds against a completed multi-condition cache without HTTP requests.
    subset_summary = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=tmp_path / "run",
                phase="score",
                addressing="explicit",
                model="openbmb/MiniCPM-o-4_5",
            )
        )
    )
    assert subset_summary["status"] == "complete"
    assert subset_summary["selected_scenarios"] == 1
    subset_results = subset_summary["results"]
    assert isinstance(subset_results, dict)
    assert "explicit" in subset_results and "implicit" not in subset_results

    # note (machengqian): Subset --phase asr and --phase judge preserve unselected cached transcripts and verdicts.
    judge_cfg_for_subset = tmp_path / "subset_judge.json"
    write_json_atomic(
        judge_cfg_for_subset,
        {
            "model": judge_config.model,
            "base_url": judge_config.base_url,
            "api_key_env": judge_config.api_key_env,
            "max_concurrency": judge_config.max_concurrency,
        },
    )
    asr_subset_result = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=tmp_path / "run",
                phase="asr",
                addressing="explicit",
                asr_base_url="http://127.0.0.1:8001",
            )
        )
    )
    assert asr_subset_result["status"] == "transcribed"
    preserved_transcribed = json.loads(
        (tmp_path / "run" / "transcribed.json").read_text(encoding="utf-8")
    )
    assert len(preserved_transcribed) == 4

    judge_subset_result = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=tmp_path / "run",
                phase="judge",
                addressing="explicit",
                judge_config_path=judge_cfg_for_subset,
            )
        )
    )
    assert judge_subset_result["status"] == "judged"
    preserved_labels = json.loads(
        (tmp_path / "run" / "labels.json").read_text(encoding="utf-8")
    )
    assert len(preserved_labels) == 4

    full_after_subsets = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=tmp_path / "run",
                phase="score",
                addressing="all",
                model="openbmb/MiniCPM-o-4_5",
            )
        )
    )
    assert full_after_subsets["status"] == "complete"

    # note (machengqian): Unknown turn/kind keys on an excluded scenario and duplicate excluded keys are still rejected.
    implicit_id = scenarios[1].scenario_id
    bogus_transcribed = [
        *transcribed,
        {
            "scenario_id": implicit_id,
            "turn_index": 999,
            "kind": "answer",
            "hyp": "bogus",
        },
    ]
    write_json_atomic(tmp_path / "run" / "transcribed.json", bogus_transcribed)
    with pytest.raises(ValueError, match="task keys"):
        asyncio.run(
            run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=tmp_path / "run",
                    phase="score",
                    addressing="explicit",
                    model="openbmb/MiniCPM-o-4_5",
                )
            )
        )
    write_json_atomic(tmp_path / "run" / "transcribed.json", transcribed)

    bogus_labels = [
        *verdicts,
        {
            "scenario_id": implicit_id,
            "turn_index": 999,
            "kind": "answer",
            "correct": True,
        },
    ]
    write_json_atomic(tmp_path / "run" / "labels.json", bogus_labels)
    with pytest.raises(ValueError, match="unknown task"):
        asyncio.run(
            run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=tmp_path / "run",
                    phase="score",
                    addressing="explicit",
                    model="openbmb/MiniCPM-o-4_5",
                )
            )
        )
    write_json_atomic(tmp_path / "run" / "labels.json", verdicts)

    # note (machengqian): Offline subset scoring succeeds when the selected subset has zero semantic tasks while another scenario has cached transcripts/labels.
    zero_task_dir = tmp_path / "zero_task_subset"
    (zero_task_dir / "audio").mkdir(parents=True)
    for scenario_row in scenarios:
        (zero_task_dir / "audio" / f"{scenario_row.scenario_id}.wav").write_bytes(
            (audio_dir / f"{scenario_row.scenario_id}.wav").read_bytes()
        )
    zero_explicit_report = {
        **reports[first_id],
        "assistant_speech_segments": [],
        "T": {"outcome": TARGET_MISS_OUTCOME},
    }
    mixed_reports = {
        first_id: zero_explicit_report,
        implicit_id: reports[implicit_id],
    }
    write_json_atomic(zero_task_dir / "report.json", mixed_reports)
    write_json_atomic(
        zero_task_dir / "transcribed.json",
        [row for row in transcribed if row["scenario_id"] == implicit_id],
    )
    write_json_atomic(
        zero_task_dir / "labels.json",
        [row for row in verdicts if row["scenario_id"] == implicit_id],
    )
    zero_subset_summary = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=zero_task_dir,
                phase="score",
                addressing="explicit",
                model="openbmb/MiniCPM-o-4_5",
            )
        )
    )
    assert zero_subset_summary["status"] == "complete"
    assert zero_subset_summary["semantic_tasks"] == 0

    # note (machengqian): Mutating gold_answer in the manifest invalidates cached prompt bindings on score.
    explicit_manifest_path = dataset_root / "manifests" / "explicit.json"
    manifest_rows = json.loads(explicit_manifest_path.read_text(encoding="utf-8"))
    manifest_rows[0]["gold_answer"] = "Mutated twenty dollars answer."
    write_json_atomic(explicit_manifest_path, manifest_rows)
    with pytest.raises(ValueError, match="gold"):
        run_duplex_mpe_cli(
            [
                "--phase",
                "score",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )


def test_generate_resume_rejects_orphan_report_and_revision_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = build_synthetic_dataset(tmp_path / "ds", pair_count=1)
    selected = load_duplex_mpe_scenarios(dataset_root, addressing="explicit")[:1]
    run_dir = tmp_path / "gen"

    endpoint_spec = RealtimeEndpointSpec(
        websocket_url="ws://127.0.0.1:9999/v1/realtime",
        capabilities_url="http://127.0.0.1:9999/v1/realtime/capabilities",
        protocol="native",
        input_sample_rate_hz=16000,
        output_sample_rate_hz=24000,
        turn_detection=None,
    )

    async def fake_negotiate(
        base_url: str,
        *,
        protocol_selection: str = "auto",
        input_sample_rate_hz: int | None = None,
        output_sample_rate_hz: int | None = None,
        turn_detection: dict[str, JsonValue] | None = None,
        timeout_seconds: float = 30.0,
        trust_env: bool = False,
    ) -> RealtimeEndpointSpec:
        return endpoint_spec

    async def fake_record(
        scenario: DuplexMpeScenarioRecord,
        *,
        dataset_root: Path,
        endpoint_spec: RealtimeEndpointSpec,
        output_audio_path: Path,
        session_timeout_seconds: float = 230.0,
        seed_mapping: dict[str, int] | None = None,
    ) -> dict[str, JsonValue]:
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(
            str(output_audio_path),
            np.zeros(24000, dtype=np.float32),
            24000,
            format="WAV",
            subtype="FLOAT",
        )
        return {
            "protocol_id": "mpeval.unified.v3",
            "aed_config": SileroActivityConfig().to_report_dict(),
            "total_duration_s": 1.0,
            "assistant_speech_segments": [],
            "assistant_audio_sha256": compute_file_sha256(output_audio_path),
            "T": {"outcome": TARGET_MISS_OUTCOME},
            "rows": [
                {
                    "turn_index": turn.turn_index,
                    "event_type": turn.event_type,
                    "speaker": turn.speaker,
                    "turn_start": 0.0,
                    "turn_end": 0.2,
                    "win_start": 0.0,
                    "win_end": 0.4,
                }
                for turn in scenario.turns
            ],
        }

    monkeypatch.setattr(
        "benchmarks.eval.benchmark_duplex_mpe.negotiate_realtime_endpoint",
        fake_negotiate,
    )
    monkeypatch.setattr(
        "benchmarks.eval.benchmark_duplex_mpe.run_realtime_scenario",
        fake_record,
    )

    base_config = DuplexMpeEvalConfig(
        dataset_root=dataset_root,
        output_dir=run_dir,
        phase="generate",
        addressing="explicit",
        max_scenarios=1,
        base_url="http://127.0.0.1:9999",
        model="openbmb/MiniCPM-o-4_5",
        model_revision="rev-1",
        launch_command=(
            "bash -lc 'AWS_SECRET_ACCESS_KEY=aws-secret OPENAI_API_KEY=secret-val "
            "python -m sglang_omni.cli serve "
            "--model-path openbmb/MiniCPM-o-4_5 --port 9999 "
            '--password "alpha beta" --api_key underscore-secret '
            "--openai-api-key=prefixed-secret "
            '-H "X-Password: header-password-secret" '
            '--header "Cookie: sessionid=cookie-sep-secret" '
            '--header="Cookie: sessionid=cookie-eq-secret" '
            "--user alice:basic-user-secret --proxy-user=bob:proxy-user-secret "
            "--pass pass-sep-secret --pass=pass-eq-secret'"
        ),
        warmup_sessions=0,
    )
    asyncio.run(run_generate_phase(base_config, selected))
    raw_saved_manifest = (run_dir / "run.json").read_text(encoding="utf-8")
    for leaked in (
        "aws-secret",
        "secret-val",
        "alpha",
        "beta",
        "underscore-secret",
        "prefixed-secret",
        "header-password-secret",
        "cookie-sep-secret",
        "cookie-eq-secret",
        "basic-user-secret",
        "proxy-user-secret",
        "pass-sep-secret",
        "pass-eq-secret",
    ):
        assert leaked not in raw_saved_manifest

    # note (machengqian): Split-phase offline score inherits sanitized launch_command from run.json and rejects conflicting declarations.
    split_score = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=run_dir,
                phase="score",
                addressing="explicit",
                max_scenarios=1,
            )
        )
    )
    split_prov = split_score["provenance"]
    assert isinstance(split_prov, dict)
    inherited_launch = split_prov["launch_command"]
    assert isinstance(inherited_launch, str)
    for leaked in (
        "aws-secret",
        "secret-val",
        "alpha",
        "beta",
        "underscore-secret",
        "prefixed-secret",
        "header-password-secret",
        "cookie-sep-secret",
        "cookie-eq-secret",
        "basic-user-secret",
        "proxy-user-secret",
        "pass-sep-secret",
        "pass-eq-secret",
    ):
        assert leaked not in inherited_launch
    assert "openbmb/MiniCPM-o-4_5" in inherited_launch

    # note (machengqian): Allowlist launch-command grammar rejects unenumerated dotted flags, URL-bearing executable tokens, JSON-embedded URL userinfo, unknown env/CLI containers, and inline code.
    for blocked_launch in (
        "uv run python -c 'serve(api_key=\"AUDIT_SENTINEL\")'",
        "uv run python -c'print(\"AUDIT_SENTINEL\")'",
        "python -uc 'print(\"AUDIT_SENTINEL\")'",
        "python -m sglang_omni.cli serve --backend.pwd AUDIT_SENTINEL",
        "python -m sglang_omni.cli serve --backend.pwd=AUDIT_SENTINEL",
        "python -m sglang_omni.cli serve --backend.custom_opt AUDIT_SENTINEL",
        "python -m sglang_omni.cli serve --backend.custom_opt=AUDIT_SENTINEL",
        "https://alice:AUDIT_SENTINEL@example.invalid/env python -m sglang_omni.cli serve",
        "curl -u alice:AUDIT_SENTINEL https://example.invalid/",
        "curl --pass AUDIT_SENTINEL --cert client.pem --key client.key https://example.invalid/",
        "CURL_OPTS='-u alice:AUDIT_SENTINEL' python -m sglang_omni.cli serve",
        "CURL_OPTS='--pass AUDIT_SENTINEL' python -m sglang_omni.cli serve",
        'BACKEND_CONFIG=\'{"url":"https://alice:AUDIT_SENTINEL@example.invalid"}\' python -m sglang_omni.cli serve',
        'python -m sglang_omni.cli serve --config \'{"url":"https://alice:AUDIT_SENTINEL@example.invalid"}\'',
        'bash -p -c \'python -c "print(\\"AUDIT_SENTINEL\\")"\'',
        'python -m app --config \'{"api_key":"AUDIT_SENTINEL"}\'',
    ):
        with pytest.raises(ValueError) as exc_sanitize:
            sanitize_launch_command(blocked_launch)
        assert "AUDIT_SENTINEL" not in str(exc_sanitize.value)
        with pytest.raises(ValueError) as exc_config:
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=tmp_path / "blocked_out",
                phase="generate",
                base_url="http://127.0.0.1:9999",
                launch_command=blocked_launch,
            )
        assert "AUDIT_SENTINEL" not in str(exc_config.value)
        assert not (tmp_path / "blocked_out" / "run.json").exists()

    # note (machengqian): Rejected credential-bearing --launch-command and JudgeEndpointConfig inputs must not leak sentinels into exception text or CLI stderr.
    bad_judge_path = tmp_path / "bad_judge_secret.json"
    write_json_atomic(
        bad_judge_path,
        {
            "model": "judge",
            "base_url": "http://127.0.0.1:30000/v1",
            "api_key": "AUDIT_SENTINEL",
        },
    )
    repo_root = Path(__file__).resolve().parents[3]
    for cli_args in (
        [
            "--phase",
            "generate",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(tmp_path / "cli_blocked_1"),
            "--base-url",
            "http://127.0.0.1:9999",
            "--launch-command",
            "https://alice:AUDIT_SENTINEL@example.invalid/env python -m sglang_omni.cli serve",
        ],
        [
            "--phase",
            "judge",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(run_dir),
            "--judge-config",
            str(bad_judge_path),
        ],
    ):
        cli_proc = subprocess.run(
            [sys.executable, "-m", "benchmarks.eval.benchmark_duplex_mpe", *cli_args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert cli_proc.returncode != 0
        assert "AUDIT_SENTINEL" not in cli_proc.stdout
        assert "AUDIT_SENTINEL" not in cli_proc.stderr

    # note (machengqian): Runtime HTTP/WebSocket failure diagnostics (401/500 warnings, exceptions, and CLI stderr) must never leak accepted private endpoint paths.
    class UnauthorizedHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(401)
            self.end_headers()

        def do_POST(self) -> None:
            content_len = int(self.headers.get("Content-Length", "0"))
            if content_len > 0:
                self.rfile.read(content_len)
            else:
                pass
            self.send_response(401)
            self.end_headers()

        def log_message(self, format_str: str, *args: str) -> None:
            return

    http_401_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), UnauthorizedHandler
    )
    http_401_port = http_401_server.server_address[1]
    server_thread = threading.Thread(target=http_401_server.serve_forever, daemon=True)
    server_thread.start()
    private_endpoint_url = f"http://127.0.0.1:{http_401_port}/private/AUDIT_SENTINEL/v1"
    try:
        captured_log_records: list[str] = []

        class MemoryLogHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_log_records.append(self.format(record))

        mem_handler = MemoryLogHandler()
        logging.getLogger("benchmarks.duplex.mpe").addHandler(mem_handler)
        try:
            fallback_spec = asyncio.run(
                negotiate_realtime_endpoint(
                    private_endpoint_url, protocol_selection="native"
                )
            )
            assert fallback_spec.protocol == "native"
            with pytest.raises(RuntimeError) as auto_exc:
                asyncio.run(
                    negotiate_realtime_endpoint(
                        private_endpoint_url, protocol_selection="auto"
                    )
                )
            assert "AUDIT_SENTINEL" not in "".join(
                traceback.format_exception(auto_exc.value)
            )

            judge_401_verdicts = asyncio.run(
                judge_tasks_via_http(
                    [
                        {
                            "scenario_id": selected[0].scenario_id,
                            "turn_index": 1,
                            "kind": "answer",
                            "context": "(none)",
                            "request": "How much?",
                            "gold": "Nine dollars.",
                            "hyp": "Nine dollars.",
                        }
                    ],
                    judge_config=JudgeEndpointConfig(
                        model="judge-m",
                        base_url=private_endpoint_url,
                        max_concurrency=1,
                    ),
                    output_path=tmp_path / "judge_401_labels.json",
                )
            )
            assert judge_401_verdicts == []
            assert "AUDIT_SENTINEL" not in "\n".join(captured_log_records)
        finally:
            logging.getLogger("benchmarks.duplex.mpe").removeHandler(mem_handler)

        asr_401_report_dir = tmp_path / "asr_401_run"
        (asr_401_report_dir / "audio").mkdir(parents=True)
        asr_401_wav = asr_401_report_dir / "audio" / f"{selected[0].scenario_id}.wav"
        soundfile.write(
            str(asr_401_wav),
            np.full(24000, 0.2, dtype=np.float32),
            24000,
            format="WAV",
            subtype="FLOAT",
        )
        write_json_atomic(
            asr_401_report_dir / "report.json",
            {
                selected[0].scenario_id: {
                    "protocol_id": "mpeval.unified.v3",
                    "aed_config": SileroActivityConfig().to_report_dict(),
                    "total_duration_s": 1.0,
                    "assistant_speech_segments": [[0.2, 0.8]],
                    "assistant_audio_sha256": compute_file_sha256(asr_401_wav),
                    "T": {"outcome": TARGET_HIT_OUTCOME},
                    "rows": [
                        {
                            "turn_index": turn.turn_index,
                            "event_type": turn.event_type,
                            "speaker": turn.speaker,
                            "turn_start": 0.0,
                            "turn_end": 0.2,
                            "win_start": 0.0,
                            "win_end": 0.9,
                        }
                        for turn in selected[0].turns
                    ],
                }
            },
        )
        asr_cli_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.eval.benchmark_duplex_mpe",
                "--phase",
                "asr",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(asr_401_report_dir),
                "--addressing",
                "explicit",
                "--max-scenarios",
                "1",
                "--asr-base-url",
                private_endpoint_url,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert asr_cli_proc.returncode != 0
        assert "AUDIT_SENTINEL" not in asr_cli_proc.stdout
        assert "AUDIT_SENTINEL" not in asr_cli_proc.stderr

        # note (machengqian): Malformed API keys with trailing newlines or control chars fail pre-dispatch without disclosing the key in exceptions or CLI stderr.
        newline_key_env = dict(os.environ, BAD_BEARER_ENV="AUDIT_SENTINEL\n")
        asr_bad_key_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.eval.benchmark_duplex_mpe",
                "--phase",
                "asr",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(asr_401_report_dir),
                "--addressing",
                "explicit",
                "--max-scenarios",
                "1",
                "--asr-base-url",
                f"http://127.0.0.1:{http_401_port}/v1",
                "--asr-api-key-env",
                "BAD_BEARER_ENV",
            ],
            cwd=str(repo_root),
            env=newline_key_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert asr_bad_key_proc.returncode != 0
        assert "AUDIT_SENTINEL" not in asr_bad_key_proc.stdout
        assert "AUDIT_SENTINEL" not in asr_bad_key_proc.stderr

        bad_bearer_judge_cfg = tmp_path / "bad_bearer_judge.json"
        write_json_atomic(
            bad_bearer_judge_cfg,
            {
                "model": "judge-m",
                "base_url": f"http://127.0.0.1:{http_401_port}/v1",
                "api_key_env": "BAD_BEARER_ENV",
            },
        )
        write_json_atomic(
            asr_401_report_dir / "transcribed.json",
            [
                {
                    **row,
                    "hyp": "Nine dollars.",
                    "assistant_audio_sha256": compute_file_sha256(asr_401_wav),
                }
                for row in build_semantic_tasks(
                    json.loads(
                        (asr_401_report_dir / "report.json").read_text(encoding="utf-8")
                    ),
                    {selected[0].scenario_id: selected[0]},
                )
            ],
        )
        judge_bad_key_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.eval.benchmark_duplex_mpe",
                "--phase",
                "judge",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(asr_401_report_dir),
                "--addressing",
                "explicit",
                "--max-scenarios",
                "1",
                "--judge-config",
                str(bad_bearer_judge_cfg),
            ],
            cwd=str(repo_root),
            env=newline_key_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert judge_bad_key_proc.returncode != 0
        assert "AUDIT_SENTINEL" not in judge_bad_key_proc.stdout
        assert "AUDIT_SENTINEL" not in judge_bad_key_proc.stderr
    finally:
        http_401_server.shutdown()
        http_401_server.server_close()

    # note (machengqian): WebSocket application-level frames (probe, error, failed response.done, invalid session.updated echoes) must never leak reflected private paths.
    async def exercise_websocket_nondisclosure() -> None:
        original_client_cls = httpx.AsyncClient

        def mock_404_caps(
            *, timeout: float = 30.0, trust_env: bool = False
        ) -> httpx.AsyncClient:
            return original_client_cls(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(404, json={"detail": "not found"})
                ),
                timeout=timeout,
                trust_env=trust_env,
            )

        monkeypatch.setattr(httpx, "AsyncClient", mock_404_caps)

        async def bad_probe_peer(connection: ServerConnection) -> None:
            await connection.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": "Route /private/AUDIT_SENTINEL/v1 invalid",
                    }
                )
            )

        async with websockets.serve(bad_probe_peer, "127.0.0.1", 0) as probe_srv:
            probe_port = probe_srv.sockets[0].getsockname()[1]
            with pytest.raises(RuntimeError) as probe_exc:
                await negotiate_realtime_endpoint(
                    f"http://127.0.0.1:{probe_port}/private/AUDIT_SENTINEL/v1",
                    protocol_selection="auto",
                )
            assert "AUDIT_SENTINEL" not in "".join(
                traceback.format_exception(probe_exc.value)
            )

        for failure_mode in ("error_event", "failed_response", "bad_echo"):

            async def reflecting_peer(
                connection: ServerConnection, mode: str = failure_mode
            ) -> None:
                await connection.send(
                    json.dumps(
                        {
                            "type": "session.created",
                            "event_id": "evt_created",
                            "session": {"id": "sess_reflect"},
                        }
                    )
                )
                async for raw_frame in connection:
                    msg = json.loads(raw_frame)
                    if msg["type"] == "session.update":
                        if mode == "error_event":
                            await connection.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "error": {
                                            "code": "AUDIT_SENTINEL",
                                            "message": "Route /private/AUDIT_SENTINEL/v1 unavailable",
                                        },
                                    }
                                )
                            )
                            return
                        elif mode == "bad_echo":
                            await connection.send(
                                json.dumps(
                                    {
                                        "type": "session.updated",
                                        "event_id": "evt_updated",
                                        "session": {
                                            "audio": {
                                                "input": {
                                                    "format": {
                                                        "type": "/private/AUDIT_SENTINEL",
                                                        "rate": 16000,
                                                    },
                                                    "turn_detection": {
                                                        "type": "/private/AUDIT_SENTINEL"
                                                    },
                                                },
                                                "output": {
                                                    "format": {
                                                        "type": "audio/pcm",
                                                        "rate": 24000,
                                                    }
                                                },
                                            }
                                        },
                                    }
                                )
                            )
                            return
                        else:
                            await connection.send(
                                json.dumps(
                                    {
                                        "type": "session.updated",
                                        "event_id": "evt_updated",
                                        "session": {
                                            "audio": {
                                                "input": {
                                                    "format": {
                                                        "type": "audio/pcm",
                                                        "rate": 16000,
                                                    },
                                                    "turn_detection": {
                                                        "type": "server_vad"
                                                    },
                                                },
                                                "output": {
                                                    "format": {
                                                        "type": "audio/pcm",
                                                        "rate": 24000,
                                                    }
                                                },
                                            }
                                        },
                                    }
                                )
                            )
                    elif msg["type"] == "input_audio_buffer.append":
                        await connection.send(
                            json.dumps(
                                {
                                    "type": "response.done",
                                    "event_id": "evt_done",
                                    "response": {
                                        "id": "/private/AUDIT_SENTINEL",
                                        "status": "failed",
                                        "status_details": {
                                            "reason": "/private/AUDIT_SENTINEL"
                                        },
                                    },
                                }
                            )
                        )
                        return
                    else:
                        pass

            async with websockets.serve(reflecting_peer, "127.0.0.1", 0) as ws_srv:
                ws_port = ws_srv.sockets[0].getsockname()[1]
                ws_spec = RealtimeEndpointSpec(
                    websocket_url=f"ws://127.0.0.1:{ws_port}/v1/realtime",
                    capabilities_url=f"http://127.0.0.1:{ws_port}/v1/realtime/capabilities",
                    protocol="turn_based",
                    input_sample_rate_hz=16000,
                    output_sample_rate_hz=24000,
                    turn_detection={"type": "server_vad", "silence_duration_ms": 500},
                )
                with pytest.raises(RuntimeError) as ws_exc:
                    await run_realtime_scenario(
                        selected[0],
                        dataset_root=dataset_root,
                        endpoint_spec=ws_spec,
                        output_audio_path=tmp_path / f"reflect_{failure_mode}.wav",
                        vad_model=EnvelopeVadModel(),
                        realtime_pacing=False,
                        session_timeout_seconds=10.0,
                    )
                assert "AUDIT_SENTINEL" not in "".join(
                    traceback.format_exception(ws_exc.value)
                )

    asyncio.run(exercise_websocket_nondisclosure())
    for bad_echo_session in (
        {
            "audio": {
                "input": {"format": {"type": "/private/AUDIT_SENTINEL", "rate": 16000}},
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            }
        },
        {
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": "/private/AUDIT_SENTINEL"}
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            }
        },
    ):
        with pytest.raises(RuntimeError) as echo_exc:
            validate_accepted_session(
                {"type": "session.updated", "session": bad_echo_session},
                endpoint_spec,
            )
        assert "AUDIT_SENTINEL" not in "".join(
            traceback.format_exception(echo_exc.value)
        )

    # note (machengqian): Unsafe scenario_id traversal/absolute/symlink retained-audio paths are rejected without reading outside audio_directory or dispatching ASR HTTP calls.
    outside_wav = tmp_path / "outside_private.wav"
    soundfile.write(
        str(outside_wav),
        np.ones(16000, dtype=np.float32),
        16000,
        format="WAV",
        subtype="FLOAT",
    )
    isolated_audio_dir = tmp_path / "isolated_audio"
    isolated_audio_dir.mkdir()
    (isolated_audio_dir / "symlink_escape.wav").symlink_to(outside_wav)
    asr_hit_count: list[str] = []
    original_http = httpx.AsyncClient

    def counting_asr_client(
        *, timeout: float = 120.0, trust_env: bool = False
    ) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            asr_hit_count.append(str(request.url))
            return httpx.Response(200, json={"text": "leaked"})

        return original_http(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr(httpx, "AsyncClient", counting_asr_client)
    for bad_scenario_id in (
        "../outside_private",
        str(outside_wav.with_suffix("")),
        "symlink_escape",
    ):
        with pytest.raises(ValueError):
            resolve_retained_scenario_audio_path(
                isolated_audio_dir, bad_scenario_id, must_exist=True
            )
        with pytest.raises(ValueError):
            asyncio.run(
                transcribe_tasks_via_http(
                    [
                        {
                            "scenario_id": bad_scenario_id,
                            "turn_index": 0,
                            "kind": "answer",
                            "context": "",
                            "request": "?",
                            "gold": "A",
                            "segments": [[0.0, 0.5]],
                        }
                    ],
                    audio_directory=isolated_audio_dir,
                    output_path=tmp_path / "bad_transcribed.json",
                    asr_base_url="http://127.0.0.1:8001",
                )
            )
    assert asr_hit_count == []

    # note (machengqian): Switching or mutating dataset manifests/turn WAVs after generate is rejected on fresh offline score/asr even without transcribed.json.
    dataset_b = build_synthetic_dataset(tmp_path / "ds_mutated", pair_count=1)
    mutated_manifest_path = dataset_b / "manifests" / "explicit.json"
    mutated_rows = json.loads(mutated_manifest_path.read_text(encoding="utf-8"))
    mutated_rows[0]["gold_answer"] = "Different dataset gold answer."
    write_json_atomic(mutated_manifest_path, mutated_rows)
    with pytest.raises(ValueError, match="run.json"):
        asyncio.run(
            run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_b,
                    output_dir=run_dir,
                    phase="score",
                    addressing="explicit",
                    max_scenarios=1,
                )
            )
        )

    with pytest.raises(ValueError, match="launch-command"):
        asyncio.run(
            run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=run_dir,
                    phase="score",
                    addressing="explicit",
                    max_scenarios=1,
                    launch_command="python -m sglang_omni.cli serve --port 7777",
                )
            )
        )

    with pytest.raises(ValueError, match="Run configuration"):
        asyncio.run(
            run_generate_phase(
                dataclasses.replace(base_config, model_revision="rev-2"), selected
            )
        )

    (run_dir / "run.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        asyncio.run(run_generate_phase(base_config, selected))


def test_eof_clamped_short_clip_and_unfinished_or_contradictory_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = build_synthetic_dataset(tmp_path / "ds", pair_count=1)
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit")[0]
    run_dir = tmp_path / "run"
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True)

    wav_path = audio_dir / f"{scenario.scenario_id}.wav"
    soundfile.write(
        str(wav_path),
        np.full(24000 * 2, 0.2, dtype=np.float32),
        24000,
        format="WAV",
        subtype="FLOAT",
    )
    audio_sha256 = compute_file_sha256(wav_path)
    # note (machengqian): Segment [1.88, 2.032] is within +40 ms rounding of 2.00 s EOF; actual sliced audio is 0.12 s (< 0.15 s).
    eof_task: dict[str, JsonValue] = {
        "scenario_id": scenario.scenario_id,
        "turn_index": 1,
        "kind": "answer",
        "context": "(none)",
        "request": "How much?",
        "gold": scenario.gold_answer,
        "segments": [[1.88, 2.032]],
        "expected_audio_sha256": audio_sha256,
    }
    dispatched_requests: list[str] = []
    original_async_client = httpx.AsyncClient

    def make_counting_client(
        *, timeout: float = 120.0, trust_env: bool = False
    ) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            dispatched_requests.append(str(request.url))
            return httpx.Response(200, json={"text": "unexpected"})

        return original_async_client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr(httpx, "AsyncClient", make_counting_client)
    transcribed_eof = asyncio.run(
        transcribe_tasks_via_http(
            [eof_task],
            audio_directory=audio_dir,
            output_path=run_dir / "eof_transcribed.json",
            asr_base_url="http://127.0.0.1:8001",
        )
    )
    assert dispatched_requests == []
    assert transcribed_eof[0]["hyp"] == ""

    # note (machengqian): Truncated finish_reason="length" verdicts stay pending and model identity inherits from run.json.
    report_entry: dict[str, JsonValue] = {
        "protocol_id": "mpeval.unified.v3",
        "aed_config": SileroActivityConfig().to_report_dict(),
        "total_duration_s": 2.0,
        "assistant_speech_segments": [[1.00, 1.80]],
        "assistant_audio_sha256": audio_sha256,
        "T": {
            "outcome": TARGET_HIT_OUTCOME,
            "t_end_s": 0.90,
            "onset_s": 1.00,
            "onset_latency_s": 0.10,
            "endpoint_s": 1.80,
            "response_duration_s": 0.80,
            "forced_floor_return": False,
            "speaking_at_t_end": False,
        },
        "rows": [
            {
                "turn_index": 0,
                "event_type": "N2",
                "speaker": "A",
                "turn_start": 0.0,
                "turn_end": 0.2,
                "win_start": 0.0,
                "win_end": 0.4,
            },
            {
                "turn_index": 1,
                "event_type": "T",
                "speaker": "B",
                "turn_start": 0.4,
                "turn_end": 0.9,
                "win_start": 0.4,
                "win_end": 1.9,
            },
            {
                "turn_index": 2,
                "event_type": "N4_Q",
                "speaker": "C",
                "turn_start": 1.9,
                "turn_end": 1.95,
                "win_start": 1.9,
                "win_end": 1.98,
            },
            {
                "turn_index": 3,
                "event_type": "N4_R",
                "speaker": "A",
                "turn_start": 1.98,
                "turn_end": 2.0,
                "win_start": 1.98,
                "win_end": 2.0,
            },
        ],
    }
    reports = {scenario.scenario_id: report_entry}
    built_task = build_semantic_tasks(reports, {scenario.scenario_id: scenario})[0]
    transcribed_task: dict[str, JsonValue] = {
        **built_task,
        "assistant_audio_sha256": audio_sha256,
        "hyp": "Nine dollars.",
    }
    truncated_verdict: dict[str, JsonValue] = {
        "scenario_id": scenario.scenario_id,
        "turn_index": 1,
        "kind": "answer",
        "correct": True,
        "finish_reason": "length",
        "judge_model": "judge-A",
        "assistant_audio_sha256": audio_sha256,
        "prompt_sha256": compute_prompt_sha256(transcribed_task),
    }
    write_json_atomic(
        run_dir / "run.json",
        {
            "model": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
            "model_revision": "rev-nemotron",
        },
    )
    write_json_atomic(run_dir / "report.json", reports)
    write_json_atomic(run_dir / "transcribed.json", [transcribed_task])
    write_json_atomic(run_dir / "labels.json", [truncated_verdict])

    summary = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=run_dir,
                phase="score",
                addressing="explicit",
            )
        )
    )
    assert summary["model"] == "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
    assert summary["status"] == "incomplete"
    results_block = summary["results"]
    assert isinstance(results_block, dict)
    explicit_block = results_block["explicit"]
    assert isinstance(explicit_block, dict)
    aa_cell = explicit_block["conditional_answer_accuracy"]
    assert isinstance(aa_cell, dict) and aa_cell["pending"] == 1

    with pytest.raises(ValueError, match="Contradictory asr_empty"):
        score_scenario_reports(
            reports,
            {scenario.scenario_id: scenario},
            [
                {
                    "scenario_id": scenario.scenario_id,
                    "turn_index": 1,
                    "kind": "answer",
                    "correct": True,
                    "error_kind": "asr_empty",
                    "assistant_audio_sha256": audio_sha256,
                }
            ],
        )


def test_premature_websocket_close_fails_promptly(tmp_path: Path) -> None:
    dataset_root = build_synthetic_dataset(tmp_path / "ds", pair_count=1)
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit")[0]

    async def exercise_abort() -> None:
        async def crashing_peer(connection: ServerConnection) -> None:
            await connection.send(
                json.dumps({"type": "session.created", "event_id": "evt_1"})
            )
            await connection.recv()
            await connection.close(1011, "simulated crash")

        async with websockets.serve(crashing_peer, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = RealtimeEndpointSpec(
                websocket_url=f"ws://127.0.0.1:{port}/v1/realtime",
                capabilities_url=f"http://127.0.0.1:{port}/v1/realtime/capabilities",
                protocol="native",
                input_sample_rate_hz=16000,
                output_sample_rate_hz=24000,
                turn_detection=None,
            )
            with pytest.raises(RuntimeError, match="closed"):
                await asyncio.wait_for(
                    run_realtime_scenario(
                        scenario,
                        dataset_root=dataset_root,
                        endpoint_spec=spec,
                        output_audio_path=tmp_path / "out.wav",
                        vad_model=EnvelopeVadModel(),
                        realtime_pacing=False,
                        session_timeout_seconds=30.0,
                    ),
                    timeout=3.0,
                )

    asyncio.run(exercise_abort())


def test_terminal_receipts_idle_cancel_and_cache_integrity_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # note (machengqian): Cancelling an idle or drained timeline buffer must not advance tail_samples past render_cursor_samples.
    timeline = PlaybackTimelineBuffer(24000)
    rendered_first = timeline.render_interval(0, 4800)
    assert len(rendered_first) == 4800
    timeline.cancel_unplayed(3.5)
    assert timeline.tail_samples == 4800

    # note (machengqian): Native and turn-based session.updated must declare audio/pcm formats with matching sample rates.
    native_spec = RealtimeEndpointSpec(
        websocket_url="ws://127.0.0.1:9999/v1/realtime",
        capabilities_url="http://127.0.0.1:9999/v1/realtime/capabilities",
        protocol="native",
        input_sample_rate_hz=16000,
        output_sample_rate_hz=22050,
        turn_detection=None,
    )
    with pytest.raises(RuntimeError, match="format"):
        validate_accepted_session(
            {"type": "session.updated", "session": {}}, native_spec
        )

    dataset_root = build_synthetic_dataset(tmp_path / "ds", pair_count=1)
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit")[0]

    # note (machengqian): phase=all with --judge-config and no --asr-base-url (and no transcribed.json) fails immediately.
    judge_cfg_file = tmp_path / "judge_cfg.json"
    write_json_atomic(
        judge_cfg_file,
        {"model": "qwen3.8-27b", "base_url": "http://127.0.0.1:30000/v1"},
    )
    with pytest.raises(ValueError, match="--asr-base-url"):
        DuplexMpeEvalConfig(
            dataset_root=dataset_root,
            output_dir=tmp_path / "fresh_out",
            phase="all",
            base_url="http://127.0.0.1:8000",
            judge_config_path=judge_cfg_file,
        )

    # note (machengqian): Missing terminal receipts (session.closed or legacy session.updated) must fail rather than succeed.
    for protocol_mode in ("native", "turn_based", "legacy"):

        async def exercise_missing_terminal_receipt(
            selected_mode: RealtimeProtocolMode,
        ) -> None:
            async def incomplete_peer(connection: ServerConnection) -> None:
                await connection.send(
                    json.dumps(
                        {
                            "type": "session.created",
                            "event_id": "evt_created",
                            "session": {"id": "sess_1"},
                        }
                    )
                )
                seen_configure = False
                async for raw_frame in connection:
                    message = json.loads(raw_frame)
                    message_type = message["type"]
                    if message_type == "session.update":
                        if not seen_configure:
                            seen_configure = True
                            session_body: dict[str, JsonValue] = (
                                {
                                    "modalities": ["text", "audio"],
                                    "turn_detection": {"type": "server_vad"},
                                }
                                if selected_mode == "legacy"
                                else {
                                    "audio": {
                                        "input": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 16000,
                                            },
                                            "turn_detection": {"type": "server_vad"},
                                        },
                                        "output": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 24000,
                                            }
                                        },
                                    }
                                }
                            )
                            await connection.send(
                                json.dumps(
                                    {
                                        "type": "session.updated",
                                        "event_id": "evt_updated",
                                        "session": session_body,
                                    }
                                )
                            )
                        else:
                            await connection.close(1000, "closed without echo")
                            return
                    elif message_type == "sglang.input_audio.end":
                        await connection.send(
                            json.dumps(
                                {
                                    "type": "sglang.input_audio.drained",
                                    "event_id": "evt_drained",
                                }
                            )
                        )
                    elif message_type == "session.close":
                        await connection.close(1000, "closed without session.closed")
                        return
                    else:
                        pass

            async with websockets.serve(incomplete_peer, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                turn_cfg: dict[str, JsonValue] | None = (
                    None
                    if selected_mode == "native"
                    else {"type": "server_vad", "silence_duration_ms": 500}
                )
                spec = RealtimeEndpointSpec(
                    websocket_url=f"ws://127.0.0.1:{port}/v1/realtime",
                    capabilities_url=f"http://127.0.0.1:{port}/v1/realtime/capabilities",
                    protocol=selected_mode,
                    input_sample_rate_hz=16000,
                    output_sample_rate_hz=24000,
                    turn_detection=turn_cfg,
                )
                with pytest.raises(RuntimeError, match="closed|failed"):
                    await run_realtime_scenario(
                        scenario,
                        dataset_root=dataset_root,
                        endpoint_spec=spec,
                        output_audio_path=tmp_path / f"incomplete_{selected_mode}.wav",
                        vad_model=EnvelopeVadModel(),
                        realtime_pacing=False,
                        session_timeout_seconds=10.0,
                    )

        asyncio.run(exercise_missing_terminal_receipt(protocol_mode))

    # note (machengqian): ASR cache binds endpoint URL; empty ASR produces 0.0 accuracy on score without judge; duplicate unfinished verdict raises ValueError.
    score_dir = tmp_path / "score_empty_asr"
    (score_dir / "audio").mkdir(parents=True)
    wav_file = score_dir / "audio" / f"{scenario.scenario_id}.wav"
    soundfile.write(
        str(wav_file),
        np.full(24000 * 2, 0.2, dtype=np.float32),
        24000,
        format="WAV",
        subtype="FLOAT",
    )
    wav_digest = compute_file_sha256(wav_file)
    report_obj: dict[str, JsonValue] = {
        "protocol_id": "mpeval.unified.v3",
        "aed_config": SileroActivityConfig().to_report_dict(),
        "total_duration_s": 2.0,
        "assistant_speech_segments": [[1.00, 1.50]],
        "assistant_audio_sha256": wav_digest,
        "T": {
            "outcome": TARGET_HIT_OUTCOME,
            "t_end_s": 0.90,
            "onset_s": 1.00,
            "onset_latency_s": 0.10,
            "endpoint_s": 1.50,
            "response_duration_s": 0.50,
            "forced_floor_return": False,
            "speaking_at_t_end": False,
        },
        "rows": [
            {
                "turn_index": 0,
                "event_type": "N2",
                "speaker": "A",
                "turn_start": 0.0,
                "turn_end": 0.2,
                "win_start": 0.0,
                "win_end": 0.4,
            },
            {
                "turn_index": 1,
                "event_type": "T",
                "speaker": "B",
                "turn_start": 0.4,
                "turn_end": 0.9,
                "win_start": 0.4,
                "win_end": 1.9,
            },
            {
                "turn_index": 2,
                "event_type": "N4_Q",
                "speaker": "C",
                "turn_start": 1.9,
                "turn_end": 1.95,
                "win_start": 1.9,
                "win_end": 1.98,
            },
            {
                "turn_index": 3,
                "event_type": "N4_R",
                "speaker": "A",
                "turn_start": 1.98,
                "turn_end": 2.0,
                "win_start": 1.98,
                "win_end": 2.0,
            },
        ],
    }
    reports_map = {scenario.scenario_id: report_obj}
    with pytest.raises(ValueError, match="Duplicate"):
        score_scenario_reports(
            reports_map,
            {scenario.scenario_id: scenario},
            [
                {
                    "scenario_id": scenario.scenario_id,
                    "turn_index": 1,
                    "kind": "answer",
                    "correct": True,
                    "finish_reason": "length",
                },
                {
                    "scenario_id": scenario.scenario_id,
                    "turn_index": 1,
                    "kind": "answer",
                    "correct": True,
                    "assistant_audio_sha256": wav_digest,
                },
            ],
        )

    asr_urls: list[str] = []
    original_client = httpx.AsyncClient

    def make_asr_client(
        *, timeout: float = 120.0, trust_env: bool = False
    ) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            asr_urls.append(str(request.url))
            transcript = "" if "empty-asr" in str(request.url) else "Nine dollars."
            return httpx.Response(200, json={"text": transcript})

        return original_client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr(httpx, "AsyncClient", make_asr_client)
    tasks_for_asr = build_semantic_tasks(reports_map, {scenario.scenario_id: scenario})
    first_asr = asyncio.run(
        transcribe_tasks_via_http(
            tasks_for_asr,
            audio_directory=score_dir / "audio",
            output_path=score_dir / "transcribed.json",
            asr_base_url="http://127.0.0.1:8001/deploy-a/v1",
        )
    )
    assert first_asr[0]["hyp"] == "Nine dollars."
    second_asr = asyncio.run(
        transcribe_tasks_via_http(
            tasks_for_asr,
            audio_directory=score_dir / "audio",
            output_path=score_dir / "transcribed.json",
            asr_base_url="http://127.0.0.1:8001/empty-asr/v1",
        )
    )
    assert len(asr_urls) == 2 and second_asr[0]["hyp"] == ""

    write_json_atomic(score_dir / "report.json", reports_map)
    empty_score_summary = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=score_dir,
                phase="score",
                addressing="explicit",
            )
        )
    )
    assert empty_score_summary["status"] == "complete"
    explicit_metrics = empty_score_summary["results"]
    assert isinstance(explicit_metrics, dict)
    explicit_split = explicit_metrics["explicit"]
    assert isinstance(explicit_split, dict)
    assert explicit_split["conditional_answer_accuracy"] == {
        "numerator": 0,
        "denominator": 1,
        "pending": 0,
        "value": 0.0,
    }

    # note (machengqian): Offline score rejects conflicting ASR endpoint/model flags and records consumed ASR provenance.
    with pytest.raises(ValueError, match="different ASR endpoint"):
        asyncio.run(
            run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=score_dir,
                    phase="score",
                    addressing="explicit",
                    asr_base_url="http://127.0.0.1:8001/other-deploy/v1",
                )
            )
        )
    with pytest.raises(ValueError, match="asr_model"):
        asyncio.run(
            run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=score_dir,
                    phase="score",
                    addressing="explicit",
                    asr_model="other/asr-model",
                )
            )
        )

    # note (machengqian): Malformed non-list speech segment fields and non-string WebSocket audio deltas must raise errors.
    malformed_segments_report = {
        **report_obj,
        "assistant_speech_segments": "not-a-segment-list",
    }
    with pytest.raises(ValueError, match="Malformed segment collection"):
        validate_scenario_reports(
            {scenario.scenario_id: malformed_segments_report},
            {scenario.scenario_id: scenario},
        )

    async def exercise_non_string_audio_delta() -> None:
        async def bad_delta_peer(connection: ServerConnection) -> None:
            sent_bad_delta = False
            await connection.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "event_id": "evt_created",
                        "session": {"id": "sess_bad_delta"},
                    }
                )
            )
            async for raw_frame in connection:
                message = json.loads(raw_frame)
                if message["type"] == "session.update":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "event_id": "evt_updated",
                                "session": {
                                    "audio": {
                                        "input": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 16000,
                                            }
                                        },
                                        "output": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 24000,
                                            }
                                        },
                                    }
                                },
                            }
                        )
                    )
                elif (
                    message["type"] == "input_audio_buffer.append"
                    and not sent_bad_delta
                ):
                    sent_bad_delta = True
                    await connection.send(
                        json.dumps(
                            {
                                "type": "response.output_audio.delta",
                                "event_id": "evt_bad_delta",
                                "delta": 123,
                            }
                        )
                    )
                elif message["type"] == "sglang.input_audio.end":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "sglang.input_audio.drained",
                                "event_id": "evt_drained",
                            }
                        )
                    )
                elif message["type"] == "session.close":
                    await connection.send(
                        json.dumps({"type": "session.closed", "event_id": "evt_closed"})
                    )
                    return
                else:
                    pass

        async with websockets.serve(bad_delta_peer, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = RealtimeEndpointSpec(
                websocket_url=f"ws://127.0.0.1:{port}/v1/realtime",
                capabilities_url=f"http://127.0.0.1:{port}/v1/realtime/capabilities",
                protocol="native",
                input_sample_rate_hz=16000,
                output_sample_rate_hz=24000,
                turn_detection=None,
            )
            with pytest.raises(RuntimeError, match="failed"):
                await run_realtime_scenario(
                    scenario,
                    dataset_root=dataset_root,
                    endpoint_spec=spec,
                    output_audio_path=tmp_path / "bad_delta.wav",
                    vad_model=EnvelopeVadModel(),
                    realtime_pacing=False,
                    session_timeout_seconds=10.0,
                )
            assert not (tmp_path / "bad_delta.wav").exists()

    asyncio.run(exercise_non_string_audio_delta())

    # note (machengqian): response.done with status=="failed" must abort the session even if drained/closed follow.
    async def exercise_failed_response_done() -> None:
        async def failed_response_peer(connection: ServerConnection) -> None:
            sent_failure = False
            await connection.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "event_id": "evt_created",
                        "session": {"id": "sess_failed_resp"},
                    }
                )
            )
            async for raw_frame in connection:
                message = json.loads(raw_frame)
                if message["type"] == "session.update":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "event_id": "evt_updated",
                                "session": {
                                    "audio": {
                                        "input": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 16000,
                                            }
                                        },
                                        "output": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 24000,
                                            }
                                        },
                                    }
                                },
                            }
                        )
                    )
                elif (
                    message["type"] == "input_audio_buffer.append" and not sent_failure
                ):
                    sent_failure = True
                    await connection.send(
                        json.dumps(
                            {
                                "type": "response.done",
                                "event_id": "evt_resp_failed",
                                "response": {
                                    "id": "resp_1",
                                    "status": "failed",
                                    "status_details": {
                                        "reason": "Backend failed at /private/AUDIT_SENTINEL/v1"
                                    },
                                },
                            }
                        )
                    )
                elif message["type"] == "sglang.input_audio.end":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "sglang.input_audio.drained",
                                "event_id": "evt_drained",
                            }
                        )
                    )
                elif message["type"] == "session.close":
                    await connection.send(
                        json.dumps({"type": "session.closed", "event_id": "evt_closed"})
                    )
                    return
                else:
                    pass

        async with websockets.serve(failed_response_peer, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = RealtimeEndpointSpec(
                websocket_url=f"ws://127.0.0.1:{port}/v1/realtime",
                capabilities_url=f"http://127.0.0.1:{port}/v1/realtime/capabilities",
                protocol="native",
                input_sample_rate_hz=16000,
                output_sample_rate_hz=24000,
                turn_detection=None,
            )
            with pytest.raises(RuntimeError, match="failed") as failed_resp_exc:
                await run_realtime_scenario(
                    scenario,
                    dataset_root=dataset_root,
                    endpoint_spec=spec,
                    output_audio_path=tmp_path / "failed_resp.wav",
                    vad_model=EnvelopeVadModel(),
                    realtime_pacing=False,
                    session_timeout_seconds=10.0,
                )
            assert "AUDIT_SENTINEL" not in "".join(
                traceback.format_exception(failed_resp_exc.value)
            )
            assert not (tmp_path / "failed_resp.wav").exists()

    asyncio.run(exercise_failed_response_done())

    # note (machengqian): Server error events and unexpected capability-probe WebSocket frames must not leak raw diagnostic messages into tracebacks or CLI stderr.
    async def exercise_ws_error_and_probe_redaction() -> None:
        async def bad_probe_peer(connection: ServerConnection) -> None:
            await connection.send(
                json.dumps(
                    {
                        "type": "unexpected.frame",
                        "detail": "Upstream path /private/AUDIT_SENTINEL/v1",
                    }
                )
            )

        def process_404(
            connection: ServerConnection, request: websockets.Request
        ) -> websockets.Response | None:
            if request.path.endswith("/capabilities"):
                return connection.respond(http.HTTPStatus.NOT_FOUND, "{}")
            else:
                return None

        async with websockets.serve(
            bad_probe_peer, "127.0.0.1", 0, process_request=process_404
        ) as probe_srv:
            probe_port = probe_srv.sockets[0].getsockname()[1]
            with pytest.raises(RuntimeError) as probe_exc:
                await negotiate_realtime_endpoint(
                    f"http://127.0.0.1:{probe_port}/private/AUDIT_SENTINEL/v1",
                    protocol_selection="auto",
                )
            assert "AUDIT_SENTINEL" not in "".join(
                traceback.format_exception(probe_exc.value)
            )

        async def error_event_peer(connection: ServerConnection) -> None:
            await connection.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "event_id": "evt_created",
                        "session": {"id": "sess_err"},
                    }
                )
            )
            async for _ in connection:
                await connection.send(
                    json.dumps(
                        {
                            "type": "error",
                            "event_id": "evt_err",
                            "error": {
                                "code": "backend_failure",
                                "message": "Backend request failed at /private/AUDIT_SENTINEL/v1",
                            },
                        }
                    )
                )
                return

        async with websockets.serve(
            error_event_peer, "127.0.0.1", 0, process_request=process_404
        ) as err_srv:
            err_port = err_srv.sockets[0].getsockname()[1]
            err_spec = RealtimeEndpointSpec(
                websocket_url=f"ws://127.0.0.1:{err_port}/private/AUDIT_SENTINEL/v1/realtime",
                capabilities_url=f"http://127.0.0.1:{err_port}/private/AUDIT_SENTINEL/v1/realtime/capabilities",
                protocol="native",
                input_sample_rate_hz=16000,
                output_sample_rate_hz=24000,
                turn_detection=None,
            )
            with pytest.raises(RuntimeError, match="backend_failure") as ws_err_exc:
                await run_realtime_scenario(
                    scenario,
                    dataset_root=dataset_root,
                    endpoint_spec=err_spec,
                    output_audio_path=tmp_path / "ws_err.wav",
                    vad_model=EnvelopeVadModel(),
                    realtime_pacing=False,
                    session_timeout_seconds=10.0,
                )
            assert "AUDIT_SENTINEL" not in "".join(
                traceback.format_exception(ws_err_exc.value)
            )

            cli_ws_proc = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-m",
                    "benchmarks.eval.benchmark_duplex_mpe",
                    "--phase",
                    "generate",
                    "--dataset-root",
                    str(dataset_root),
                    "--output-dir",
                    str(tmp_path / "ws_err_cli_out"),
                    "--addressing",
                    "explicit",
                    "--max-scenarios",
                    "1",
                    "--warmup",
                    "0",
                    "--protocol",
                    "native",
                    "--base-url",
                    f"http://127.0.0.1:{err_port}/private/AUDIT_SENTINEL/v1",
                ],
                cwd=str(Path(__file__).resolve().parents[3]),
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert cli_ws_proc.returncode != 0
            assert "AUDIT_SENTINEL" not in cli_ws_proc.stdout
            assert "AUDIT_SENTINEL" not in cli_ws_proc.stderr

        turn_spec = RealtimeEndpointSpec(
            websocket_url="ws://127.0.0.1:9999/v1/realtime",
            capabilities_url="http://127.0.0.1:9999/v1/realtime/capabilities",
            protocol="turn_based",
            input_sample_rate_hz=16000,
            output_sample_rate_hz=24000,
            turn_detection={"type": "/private/AUDIT_SENTINEL/v1"},
        )
        legacy_spec = RealtimeEndpointSpec(
            websocket_url="ws://127.0.0.1:9999/v1/realtime",
            capabilities_url="http://127.0.0.1:9999/v1/realtime/capabilities",
            protocol="legacy",
            input_sample_rate_hz=16000,
            output_sample_rate_hz=24000,
            turn_detection={"type": "server_vad"},
        )
        for bad_updated_payload, candidate_spec in (
            (
                {
                    "type": "session.updated",
                    "session": {
                        "audio": {
                            "input": {
                                "turn_detection": {"type": "other_AUDIT_SENTINEL"}
                            }
                        }
                    },
                },
                turn_spec,
            ),
            (
                {
                    "type": "session.updated",
                    "session": {
                        "turn_detection": {"type": "/private/AUDIT_SENTINEL/v1"}
                    },
                },
                legacy_spec,
            ),
            (
                {
                    "type": "session.updated",
                    "session": {
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "/private/AUDIT_SENTINEL/v1",
                                    "rate": 16000,
                                }
                            },
                            "output": {"format": {"type": "audio/pcm", "rate": 22050}},
                        }
                    },
                },
                native_spec,
            ),
            (
                {
                    "type": "session.updated",
                    "session": {
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": "/private/AUDIT_SENTINEL/v1",
                                }
                            },
                            "output": {"format": {"type": "audio/pcm", "rate": 22050}},
                        }
                    },
                },
                native_spec,
            ),
            (
                {
                    "type": "session.updated",
                    "session": {
                        "turn_detection": {"type": "server_vad"},
                        "audio": {
                            "output": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": "/private/AUDIT_SENTINEL/v1",
                                }
                            }
                        },
                    },
                },
                legacy_spec,
            ),
        ):
            with pytest.raises(RuntimeError) as bad_upd_exc:
                validate_accepted_session(bad_updated_payload, candidate_spec)
            assert "AUDIT_SENTINEL" not in "".join(
                traceback.format_exception(bad_upd_exc.value)
            )

        async def bad_session_updated_peer(connection: ServerConnection) -> None:
            await connection.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "event_id": "evt_created",
                        "session": {"id": "sess_bad_upd"},
                    }
                )
            )
            async for raw_frame in connection:
                message = json.loads(raw_frame)
                if message["type"] == "session.update":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "event_id": "evt_updated",
                                "session": {
                                    "audio": {
                                        "input": {
                                            "turn_detection": {
                                                "type": "/private/AUDIT_SENTINEL/v1"
                                            },
                                            "format": {
                                                "type": "/private/AUDIT_SENTINEL/v1",
                                                "rate": "/private/AUDIT_SENTINEL/v1",
                                            },
                                        },
                                        "output": {
                                            "format": {
                                                "type": "/private/AUDIT_SENTINEL/v1",
                                                "rate": "/private/AUDIT_SENTINEL/v1",
                                            }
                                        },
                                    }
                                },
                            }
                        )
                    )
                    return
                else:
                    pass

        async with websockets.serve(
            bad_session_updated_peer, "127.0.0.1", 0, process_request=process_404
        ) as bad_upd_srv:
            bad_upd_port = bad_upd_srv.sockets[0].getsockname()[1]
            for cli_protocol in ("native", "turn_based"):
                cli_upd_proc = await asyncio.to_thread(
                    subprocess.run,
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.eval.benchmark_duplex_mpe",
                        "--phase",
                        "generate",
                        "--dataset-root",
                        str(dataset_root),
                        "--output-dir",
                        str(tmp_path / f"upd_cli_{cli_protocol}"),
                        "--addressing",
                        "explicit",
                        "--max-scenarios",
                        "1",
                        "--warmup",
                        "0",
                        "--protocol",
                        cli_protocol,
                        "--base-url",
                        f"http://127.0.0.1:{bad_upd_port}/private/AUDIT_SENTINEL/v1",
                    ],
                    cwd=str(Path(__file__).resolve().parents[3]),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert cli_upd_proc.returncode != 0
                assert "AUDIT_SENTINEL" not in cli_upd_proc.stdout
                assert "AUDIT_SENTINEL" not in cli_upd_proc.stderr

    monkeypatch.setattr(httpx, "AsyncClient", original_client)
    asyncio.run(exercise_ws_error_and_probe_redaction())

    # note (machengqian): Explicitly supplying default --asr-model rejects non-default cached transcripts, and offline score inherits warmup/timeout from run.json.
    write_json_atomic(
        score_dir / "run.json",
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "warmup_sessions": 0,
            "session_timeout_seconds": 120.0,
        },
    )
    inherited_summary = asyncio.run(
        run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=score_dir,
                phase="score",
                addressing="explicit",
            )
        )
    )
    provenance_obj = inherited_summary["provenance"]
    assert isinstance(provenance_obj, dict)
    server_prov = provenance_obj["declared_server_config"]
    assert isinstance(server_prov, dict)
    assert server_prov["warmup_sessions"] == 0
    assert server_prov["session_timeout_seconds"] == 120.0

    custom_asr_rows = json.loads(
        (score_dir / "transcribed.json").read_text(encoding="utf-8")
    )
    custom_asr_rows[0]["asr_model"] = "custom/asr-v2"
    write_json_atomic(score_dir / "transcribed.json", custom_asr_rows)
    with pytest.raises(ValueError, match="asr_model"):
        run_duplex_mpe_cli(
            [
                "--phase",
                "score",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(score_dir),
                "--addressing",
                "explicit",
                "--asr-model",
                "Qwen/Qwen3-ASR-1.7B",
            ]
        )

    # note (machengqian): Modifying retained assistant WAV on disk fails offline score.
    soundfile.write(
        str(wav_file),
        np.zeros(24000 * 2, dtype=np.float32),
        24000,
        format="WAV",
        subtype="FLOAT",
    )
    with pytest.raises(ValueError, match="Retained assistant audio"):
        asyncio.run(
            run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=score_dir,
                    phase="score",
                    addressing="explicit",
                )
            )
        )


def test_concurrent_realtime_pacing_thread_offload_and_lag_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlapping session preparation/persistence runs off the pacing loop, and excessive loop stalls fail closed."""
    dataset_root = build_synthetic_dataset(tmp_path / "ds_paced", pair_count=1)
    # note (machengqian): Shorten synthetic clips to 80 ms so two real-time paced sessions finish in ~0.8 s.
    tiny_clip = np.zeros(1280, dtype=np.float32)
    for wav_candidate in (dataset_root / "audio").rglob("*.wav"):
        soundfile.write(
            str(wav_candidate), tiny_clip, 16000, format="WAV", subtype="FLOAT"
        )
    scenarios = load_duplex_mpe_scenarios(dataset_root, addressing="all")

    monkeypatch.setattr(
        "benchmarks.duplex.mpe.realtime_session.ORDINARY_GAP_SECONDS", 0.08
    )
    monkeypatch.setattr("benchmarks.duplex.mpe.realtime_session.N4_PAUSE_SECONDS", 0.08)
    monkeypatch.setattr(
        "benchmarks.duplex.mpe.realtime_session.TARGET_ONSET_DEADLINE_SECONDS", 0.16
    )

    class SlowInitVadModel(EnvelopeVadModel):
        def __init__(self) -> None:
            super().__init__()
            # note (machengqian): Simulate 180 ms blocking model load; if executed on the event loop, active peer pacing would lag >180 ms.
            time.sleep(0.18)

    async def exercise_parallel_paced_sessions() -> None:
        async def paced_peer(connection: ServerConnection) -> None:
            await connection.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "event_id": "evt_created",
                        "session": {"id": "sess_paced"},
                    }
                )
            )
            async for raw_frame in connection:
                message = json.loads(raw_frame)
                if message["type"] == "session.update":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "event_id": "evt_updated",
                                "session": {
                                    "audio": {
                                        "input": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 16000,
                                            }
                                        },
                                        "output": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 24000,
                                            }
                                        },
                                    }
                                },
                            }
                        )
                    )
                elif message["type"] == "sglang.input_audio.end":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "sglang.input_audio.drained",
                                "event_id": "evt_drained",
                            }
                        )
                    )
                elif message["type"] == "session.close":
                    await connection.send(
                        json.dumps({"type": "session.closed", "event_id": "evt_closed"})
                    )
                    return
                else:
                    pass

        monkeypatch.setattr(duplex_mpe_vad, "load_silero_vad", SlowInitVadModel)
        async with websockets.serve(paced_peer, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = RealtimeEndpointSpec(
                websocket_url=f"ws://127.0.0.1:{port}/v1/realtime",
                capabilities_url=f"http://127.0.0.1:{port}/v1/realtime/capabilities",
                protocol="native",
                input_sample_rate_hz=16000,
                output_sample_rate_hz=24000,
                turn_detection=None,
            )

            async def delayed_second_session() -> dict[str, JsonValue]:
                await asyncio.sleep(0.10)
                return await run_realtime_scenario(
                    scenarios[1],
                    dataset_root=dataset_root,
                    endpoint_spec=spec,
                    output_audio_path=tmp_path / "paced_1.wav",
                    realtime_pacing=True,
                    max_pacing_lag_seconds=0.14,
                    session_timeout_seconds=15.0,
                )

            report_a, report_b = await asyncio.gather(
                run_realtime_scenario(
                    scenarios[0],
                    dataset_root=dataset_root,
                    endpoint_spec=spec,
                    output_audio_path=tmp_path / "paced_0.wav",
                    realtime_pacing=True,
                    max_pacing_lag_seconds=0.14,
                    session_timeout_seconds=15.0,
                ),
                delayed_second_session(),
            )
            lag_a = report_a["max_pacing_lag_ms"]
            lag_b = report_b["max_pacing_lag_ms"]
            assert isinstance(lag_a, (int, float)) and float(lag_a) < 140.0
            assert isinstance(lag_b, (int, float)) and float(lag_b) < 140.0

            # note (machengqian): Genuine event-loop stalls exceeding max_pacing_lag_seconds must raise RuntimeError rather than silently catching up.
            async def inject_event_loop_stall() -> None:
                await asyncio.sleep(0.24)
                time.sleep(0.18)

            with pytest.raises(RuntimeError, match="pacing lagged"):
                await asyncio.gather(
                    run_realtime_scenario(
                        scenarios[0],
                        dataset_root=dataset_root,
                        endpoint_spec=spec,
                        output_audio_path=tmp_path / "stalled.wav",
                        vad_model=EnvelopeVadModel(),
                        realtime_pacing=True,
                        max_pacing_lag_seconds=0.08,
                        session_timeout_seconds=15.0,
                    ),
                    inject_event_loop_stall(),
                )
            assert not (tmp_path / "stalled.wav").exists()

        # note (machengqian): A stall synchronized to the very last input_audio_buffer.append must also fail closed without persisting artifacts.
        final_append_seen = asyncio.Event()

        async def final_stall_peer(connection: ServerConnection) -> None:
            append_count = 0
            await connection.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "event_id": "evt_created",
                        "session": {"id": "sess_final_stall"},
                    }
                )
            )
            async for raw_frame in connection:
                message = json.loads(raw_frame)
                if message["type"] == "session.update":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "event_id": "evt_updated",
                                "session": {
                                    "audio": {
                                        "input": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 16000,
                                            }
                                        },
                                        "output": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": 24000,
                                            }
                                        },
                                    }
                                },
                            }
                        )
                    )
                elif message["type"] == "input_audio_buffer.append":
                    append_count += 1
                    # 1 preamble + 1 gap + 4*(1 turn + gap/deadline) = 11 total 80 ms packets.
                    if append_count == 11:
                        final_append_seen.set()
                    else:
                        pass
                elif message["type"] == "sglang.input_audio.end":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "sglang.input_audio.drained",
                                "event_id": "evt_drained",
                            }
                        )
                    )
                elif message["type"] == "session.close":
                    await connection.send(
                        json.dumps({"type": "session.closed", "event_id": "evt_closed"})
                    )
                    return
                else:
                    pass

        async def stall_on_final_append() -> None:
            await final_append_seen.wait()
            time.sleep(0.18)

        async with websockets.serve(final_stall_peer, "127.0.0.1", 0) as final_server:
            final_port = final_server.sockets[0].getsockname()[1]
            final_spec = RealtimeEndpointSpec(
                websocket_url=f"ws://127.0.0.1:{final_port}/v1/realtime",
                capabilities_url=f"http://127.0.0.1:{final_port}/v1/realtime/capabilities",
                protocol="native",
                input_sample_rate_hz=16000,
                output_sample_rate_hz=24000,
                turn_detection=None,
            )
            with pytest.raises(RuntimeError, match="pacing lagged"):
                await asyncio.gather(
                    run_realtime_scenario(
                        scenarios[0],
                        dataset_root=dataset_root,
                        endpoint_spec=final_spec,
                        output_audio_path=tmp_path / "stalled_final.wav",
                        vad_model=EnvelopeVadModel(),
                        realtime_pacing=True,
                        max_pacing_lag_seconds=0.08,
                        session_timeout_seconds=15.0,
                    ),
                    stall_on_final_append(),
                )
            assert not (tmp_path / "stalled_final.wav").exists()

    asyncio.run(exercise_parallel_paced_sessions())
