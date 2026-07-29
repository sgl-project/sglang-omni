# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from sglang_omni.models.minicpmo_4_5 import tts_runtime as runtime_module
from sglang_omni.models.minicpmo_4_5.tts_runtime import (
    MiniCPMO45TTSRuntime,
    MiniCPMTTS,
    MiniCPMTTSArchitectureConfig,
)


def _tiny_architecture() -> MiniCPMTTSArchitectureConfig:
    return MiniCPMTTSArchitectureConfig(
        llm_dim=4,
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=1,
        num_hidden_layers=1,
        num_key_value_heads=1,
        max_position_embeddings=32,
        num_audio_tokens=8,
        num_text_tokens=10,
        num_vq=1,
        audio_bos_token_id=5,
    )


def test_condition_preserves_token_hidden_alignment_and_appends_audio_bos() -> None:
    model = MiniCPMTTS(_tiny_architecture())
    projector = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        projector.weight.copy_(torch.eye(4))
        model.emb_text.weight.zero_()
        model.emb_text.weight[1] = torch.tensor([10.0, 0.0, 0.0, 0.0])
        model.emb_text.weight[2] = torch.tensor([0.0, 10.0, 0.0, 0.0])
        model.emb_text.weight[5] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    model.projector_semantic = projector

    condition = model.build_condition(
        [1, 2],
        [
            torch.tensor([[[3.0, 0.0, 0.0, 0.0]]]),
            torch.tensor([[[0.0, 4.0, 0.0, 0.0]]]),
        ],
    )

    assert condition.shape == (1, 3, 4)
    torch.testing.assert_close(condition[0, 0], torch.tensor([11.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(condition[0, 1], torch.tensor([0.0, 11.0, 0.0, 0.0]))
    torch.testing.assert_close(condition[0, 2], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    with pytest.raises(ValueError, match="alignment mismatch"):
        model.build_condition([1, 2], torch.zeros(1, 4))


class _FakeTTS:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.config = SimpleNamespace(num_audio_tokens=16)
        self.generate_kwargs = None

    def build_condition(self, token_ids, hidden_states):
        del hidden_states
        return torch.zeros(1, len(token_ids) + 1, 4)

    def generate_chunk(self, **kwargs):
        self.events.append("generate")
        self.generate_kwargs = kwargs
        return torch.tensor([[[7], [8]]], dtype=torch.long), {"next": "kv"}


class _FakeToken2wav:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.flow = SimpleNamespace(pre_lookahead_len=3)
        self.cache = object()
        self.stream_cache = None
        self.hift_cache_dict = None
        self.closed = False

    def set_stream_cache(self, prompt_wav_path: str):
        self.events.append(f"open:{prompt_wav_path}")
        self.stream_cache = {"flow": torch.tensor([1])}
        self.hift_cache_dict = {"hift": torch.tensor([2])}
        return self.stream_cache, self.hift_cache_dict

    def stream(self, tokens, *, prompt_wav: str, last_chunk: bool = False):
        del tokens, prompt_wav
        self.events.append("stream:last" if last_chunk else "stream")
        # Two little-endian PCM16 samples.
        return np.asarray([1024, -1024], dtype="<i2").tobytes()

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


def _fake_runtime(*, owns_token2wav: bool = False):
    events: list[str] = []
    tts = _FakeTTS(events)
    token2wav = _FakeToken2wav(events)
    runtime = MiniCPMO45TTSRuntime(
        tts,  # type: ignore[arg-type]
        token2wav,
        sample_rate=4,
        owns_token2wav=owns_token2wav,
    )
    return runtime, tts, token2wav, events


def test_turn_end_flushes_token2wav_before_resetting_session_state(tmp_path) -> None:
    runtime, tts, _, events = _fake_runtime()
    prompt = tmp_path / "voice.wav"
    prompt.write_bytes(b"wav")
    state = runtime.open_session("session", prompt_wav_path=str(prompt))
    state.past_key_values = {"old": "kv"}
    state.text_start_pos = 9

    original_reset = runtime._reset_turn_state

    def record_reset(target_state):
        events.append("reset")
        original_reset(target_state)

    runtime._reset_turn_state = record_reset  # type: ignore[method-assign]
    chunk = runtime.synthesize(
        "session",
        [1, 2],
        torch.zeros(2, 4),
        end_of_turn=True,
    )

    assert events.index("stream:last") < events.index("reset")
    assert chunk.end_of_turn is True
    assert chunk.waveform is not None
    np.testing.assert_allclose(chunk.waveform, np.asarray([1 / 32, -1 / 32]))
    assert tts.generate_kwargs["past_key_values"] == {"old": "kv"}
    assert tts.generate_kwargs["text_start_pos"] == 9
    assert tts.generate_kwargs["min_new_tokens"] == 0
    assert state.past_key_values is None
    assert state.text_start_pos == 0
    assert state.token_buffer == [4218, 4218, 4218]


def test_close_releases_session_and_owned_token2wav(tmp_path) -> None:
    runtime, _, token2wav, events = _fake_runtime(owns_token2wav=True)
    first_prompt = tmp_path / "a.wav"
    first_prompt.write_bytes(b"wav")
    first = runtime.open_session("first", prompt_wav_path=str(first_prompt))
    first.past_key_values = object()

    runtime.close()
    runtime.close()

    assert runtime.session_count == 0
    assert first.closed
    assert first.past_key_values is None
    assert first.flow_cache is None and first.hift_cache is None
    assert token2wav.cache is None
    assert token2wav.stream_cache is None
    assert token2wav.hift_cache_dict is None
    assert token2wav.closed
    assert events.count("close") == 1
    with pytest.raises(RuntimeError, match="runtime is closed"):
        runtime.open_session("later", prompt_wav_path="c.wav")


def test_provider_global_prompt_cache_rejects_second_active_session(tmp_path) -> None:
    runtime, _, _, _ = _fake_runtime()
    first_prompt = tmp_path / "a.wav"
    second_prompt = tmp_path / "b.wav"
    first_prompt.write_bytes(b"wav")
    second_prompt.write_bytes(b"wav")
    runtime.open_session("first", prompt_wav_path=str(first_prompt))

    with pytest.raises(RuntimeError, match="one active session"):
        runtime.open_session("second", prompt_wav_path=str(second_prompt))


def test_last_session_close_detaches_provider_prompt_cache(tmp_path) -> None:
    runtime, _, token2wav, _ = _fake_runtime()
    prompt = tmp_path / "voice.wav"
    prompt.write_bytes(b"wav")
    runtime.open_session("session", prompt_wav_path=str(prompt))
    token2wav.cache = object()

    runtime.close_session("session")

    assert runtime.session_count == 0
    assert token2wav.cache is None
    assert token2wav.stream_cache is None
    assert token2wav.hift_cache_dict is None


def test_interrupt_can_flush_buffer_before_reset(tmp_path) -> None:
    runtime, _, _, events = _fake_runtime()
    prompt = tmp_path / "voice.wav"
    prompt.write_bytes(b"wav")
    state = runtime.open_session("session", prompt_wav_path=str(prompt))
    state.text_start_pos = 10
    state.past_key_values = object()
    state.token_buffer.extend([7, 8])

    original_reset = runtime._reset_turn_state

    def record_reset(target_state):
        events.append("reset")
        original_reset(target_state)

    runtime._reset_turn_state = record_reset  # type: ignore[method-assign]
    waveform = runtime.interrupt_session("session", flush=True)

    assert waveform is not None
    assert events.index("stream:last") < events.index("reset")
    assert state.past_key_values is None
    assert state.text_start_pos == 0


def test_from_pretrained_builds_from_config_and_loads_only_tts_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = {
        "tts_config": {
            "llm_dim": 4,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "num_audio_tokens": 8,
            "num_text_tokens": 10,
            "num_vq": 1,
            "audio_bos_token_id": 5,
            "projector_type": "mlp",
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "assets" / "token2wav").mkdir(parents=True)
    default_prompt = tmp_path / "assets" / "HT_ref_audio.wav"
    default_prompt.write_bytes(b"wav")
    load_calls = []

    resolve_calls = []

    def fake_resolve_model_path(*args, **kwargs):
        resolve_calls.append((args, kwargs))
        return tmp_path

    monkeypatch.setattr(runtime_module, "resolve_model_path", fake_resolve_model_path)

    def fake_load_module(module, model_path, **kwargs):
        load_calls.append((module, model_path, kwargs))
        return module

    monkeypatch.setattr(runtime_module, "load_module", fake_load_module)
    fake_accelerate = ModuleType("accelerate")
    fake_accelerate.init_empty_weights = nullcontext
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    token_events: list[str] = []

    def token2wav_factory(model_dir, **kwargs):
        token_events.append(f"factory:{model_dir}:{kwargs}")
        return _FakeToken2wav(token_events)

    runtime = MiniCPMO45TTSRuntime.from_pretrained(
        "unused/model-id",
        revision="checkpoint-sha",
        device="cpu",
        dtype="float32",
        token2wav_factory=token2wav_factory,
    )

    assert len(load_calls) == 1
    assert resolve_calls == [
        (
            ("unused/model-id",),
            {"local_files_only": False, "revision": "checkpoint-sha"},
        )
    ]
    loaded_model, loaded_path, kwargs = load_calls[0]
    assert isinstance(loaded_model, MiniCPMTTS)
    assert loaded_model.config.num_text_tokens == 10
    assert loaded_path == str(tmp_path)
    assert kwargs["prefix"] == "tts."
    assert kwargs["strict"] is False
    assert kwargs["require_all_module_keys"] is True
    assert kwargs["allowed_unexpected_prefixes"] == ("projector_spk.",)
    assert kwargs["device"] == "cpu"
    assert token_events[0].startswith(f"factory:{tmp_path / 'assets' / 'token2wav'}:")
    state = runtime.open_session("default-voice")
    assert state.prompt_wav_path == str(default_prompt)
    runtime.close()
