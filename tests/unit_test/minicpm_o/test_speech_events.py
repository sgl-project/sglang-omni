# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MiniCPM-o speech-path profiler events (GPU-free)."""

import json

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.stages import (
    _run_code2wav_payload,
    _run_single_encoder_payload,
)
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(
    state: MiniCPMOPipelineState, request_id: str = "req-events"
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi", params={}, metadata={}),
        data=state.to_dict(),
    )


def _read_events(event_dir) -> list[dict]:
    events = []
    for path in sorted(event_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                events.append(json.loads(line))
    return events


@pytest.fixture()
def event_dir(tmp_path):
    recorder = get_recorder()
    # The recorder is a process-global singleton; a leaked active session
    # from another test would silently swallow this test's events.
    assert not recorder.is_active()
    recorder.start("test-run", str(tmp_path), "speech")
    yield tmp_path
    recorder.stop()


class _FakeCode2Wav:
    def __call__(self, *, codec_tokens):
        n = int(codec_tokens.numel())
        return {
            "waveform": np.zeros(n * 480, dtype=np.float32),
            "sample_rate": 24000,
        }


def test_code2wav_events_carry_audio_metadata(event_dir):
    state = MiniCPMOPipelineState(
        engine_outputs={"talker": {"codec_tokens": torch.arange(5, dtype=torch.long)}}
    )
    out = _run_code2wav_payload(_payload(state), model=_FakeCode2Wav())
    assert out.data["sample_rate"] == 24000

    events = {e["event_name"]: e for e in _read_events(event_dir)}
    assert events["code2wav_decode_start"]["metadata"] == {"codec_tokens": 5}
    assert events["code2wav_decode_end"]["metadata"] == {
        "codec_tokens": 5,
        "audio_samples": 2400,
        "audio_seconds": pytest.approx(0.1),
        "status": "ok",
    }
    assert events["code2wav_first_audio"]["metadata"] == {"samples": 2400}


def test_code2wav_empty_output_skips_first_audio(event_dir):
    state = MiniCPMOPipelineState(
        engine_outputs={"talker": {"codec_tokens": torch.empty(0, dtype=torch.long)}}
    )
    _run_code2wav_payload(_payload(state), model=_FakeCode2Wav())

    names = [e["event_name"] for e in _read_events(event_dir)]
    assert "code2wav_decode_end" in names
    assert "code2wav_first_audio" not in names


class _FakeEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, **inputs):
        self.calls += 1
        return {"embeddings": torch.ones(2, 4)}


def _encoder_payload() -> StagePayload:
    state = MiniCPMOPipelineState(
        encoder_inputs={
            "image_encoder": {
                "cache_key": "k1",
                "pixel_values": [torch.zeros(3, 2, 2), torch.zeros(3, 2, 2)],
            }
        }
    )
    return _payload(state)


def test_encoder_events_record_cache_hits(event_dir):
    from sglang_omni.scheduling.stage_cache import StageOutputCache

    model = _FakeEncoder()
    cache = StageOutputCache(max_size=4, max_bytes=1 << 20, cache_device="cpu")
    for _ in range(2):
        _run_single_encoder_payload(
            _encoder_payload(), stage_name="image_encoder", model=model, cache=cache
        )
    assert model.calls == 1

    events = _read_events(event_dir)
    starts = [e for e in events if e["event_name"] == "encoder_start"]
    assert all(e["metadata"]["num_items"] == 2 for e in starts)
    ends = [e for e in events if e["event_name"] == "encoder_end"]
    assert [e["metadata"]["cache_hit"] for e in ends] == [False, True]
    assert all(e["metadata"]["modality"] == "image" for e in ends)
    assert all(e["metadata"]["cacheable"] for e in ends)
    assert all(e["metadata"]["status"] == "ok" for e in ends)


def test_encoder_item_count_covers_both_modalities():
    from sglang_omni.models.minicpm_o.stages import _encoder_item_count

    assert _encoder_item_count({"pixel_values": [1, 2, 3]}) == 3
    assert _encoder_item_count({"audio_features": torch.zeros(4, 80, 10)}) == 4
    assert _encoder_item_count({}) is None


def test_encoder_skip_run_emits_no_events(event_dir):
    model = _FakeEncoder()
    _run_single_encoder_payload(
        _payload(MiniCPMOPipelineState()),
        stage_name="image_encoder",
        model=model,
        cache=None,
    )
    assert model.calls == 0
    assert _read_events(event_dir) == []
