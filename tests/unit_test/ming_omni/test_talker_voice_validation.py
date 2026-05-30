# SPDX-License-Identifier: Apache-2.0
"""Voice-preset validation, duration cap, and final-chunk flush."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.models.ming_omni.components.streaming_talker import (
    MingStreamingTalkerScheduler,
)
from sglang_omni.models.ming_omni.components.talker_executor import MingTalkerExecutor
from sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker import MingOmniTalker


def _write_voice_manifest(
    talker_dir: Path, presets: dict, wav_files: dict | None = None
) -> Path:
    """Write a voice_name.json manifest plus referenced wav stubs."""
    data_dir = talker_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "voice_name.json"
    manifest.write_text(json.dumps(presets), encoding="utf-8")
    if wav_files is not None:
        for rel, content in wav_files.items():
            wav_path = talker_dir / rel
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.write_bytes(content)
    return manifest


def test_executor_validate_resolves_paths_when_default_voice_present(tmp_path: Path):
    talker_dir = tmp_path / "talker"
    wav_rel = "spks/DB30.wav"
    manifest = _write_voice_manifest(
        talker_dir,
        {"DB30": {"prompt_text": "x", "prompt_wav_path": wav_rel}},
        {wav_rel: b"\x00" * 16},
    )
    voice_dict = {"DB30": {"prompt_text": "x", "prompt_wav_path": wav_rel}}
    executor = MingTalkerExecutor(
        model_path=str(tmp_path / "model"),
        talker_model_path=str(talker_dir),
        voice="DB30",
    )
    executor._validate_voice_presets(voice_dict, str(manifest), str(talker_dir))
    assert voice_dict["DB30"]["prompt_wav_path"] == str(talker_dir / wav_rel)


def test_executor_validate_raises_when_default_voice_missing(tmp_path: Path):
    talker_dir = tmp_path / "talker"
    wav_rel = "spks/OTHER.wav"
    manifest = _write_voice_manifest(
        talker_dir,
        {"OTHER": {"prompt_text": "x", "prompt_wav_path": wav_rel}},
        {wav_rel: b"\x00" * 16},
    )
    voice_dict = {"OTHER": {"prompt_text": "x", "prompt_wav_path": wav_rel}}
    executor = MingTalkerExecutor(
        model_path=str(tmp_path / "model"),
        talker_model_path=str(talker_dir),
        voice="DB30",
    )
    with pytest.raises(ValueError, match="default voice 'DB30' not found"):
        executor._validate_voice_presets(voice_dict, str(manifest), str(talker_dir))


def test_executor_validate_raises_when_wav_missing(tmp_path: Path):
    talker_dir = tmp_path / "talker"
    wav_rel = "spks/DB30.wav"
    manifest = _write_voice_manifest(
        talker_dir,
        {"DB30": {"prompt_text": "x", "prompt_wav_path": wav_rel}},
        wav_files=None,
    )
    voice_dict = {"DB30": {"prompt_text": "x", "prompt_wav_path": wav_rel}}
    executor = MingTalkerExecutor(
        model_path=str(tmp_path / "model"),
        talker_model_path=str(talker_dir),
        voice="DB30",
    )
    with pytest.raises(FileNotFoundError, match="missing prompt wav"):
        executor._validate_voice_presets(voice_dict, str(manifest), str(talker_dir))


def test_executor_validate_raises_value_error_not_key_error_on_bad_manifest(
    tmp_path: Path,
):
    """Manifest entry without prompt_wav_path must surface a clear ValueError,
    not a bare KeyError from dict[...] access."""
    talker_dir = tmp_path / "talker"
    manifest = _write_voice_manifest(
        talker_dir,
        {"DB30": {"prompt_text": "x"}},  # missing prompt_wav_path key
    )
    voice_dict = {"DB30": {"prompt_text": "x"}}
    executor = MingTalkerExecutor(
        model_path=str(tmp_path / "model"),
        talker_model_path=str(talker_dir),
        voice="DB30",
    )
    with pytest.raises(ValueError, match="missing prompt_wav_path"):
        executor._validate_voice_presets(voice_dict, str(manifest), str(talker_dir))


def test_executor_validate_allows_voice_none_with_unrelated_presets(tmp_path: Path):
    talker_dir = tmp_path / "talker"
    wav_rel = "spks/OTHER.wav"
    manifest = _write_voice_manifest(
        talker_dir,
        {"OTHER": {"prompt_text": "x", "prompt_wav_path": wav_rel}},
        {wav_rel: b"\x00" * 16},
    )
    voice_dict = {"OTHER": {"prompt_text": "x", "prompt_wav_path": wav_rel}}
    executor = MingTalkerExecutor(
        model_path=str(tmp_path / "model"),
        talker_model_path=str(talker_dir),
        voice=None,
    )
    executor._validate_voice_presets(voice_dict, str(manifest), str(talker_dir))


def test_scheduler_validate_raises_when_default_voice_missing(tmp_path: Path):
    talker_dir = tmp_path / "talker"
    wav_rel = "spks/OTHER.wav"
    manifest = _write_voice_manifest(
        talker_dir,
        {"OTHER": {"prompt_text": "x", "prompt_wav_path": wav_rel}},
        {wav_rel: b"\x00" * 16},
    )
    voice_dict = {"OTHER": {"prompt_text": "x", "prompt_wav_path": wav_rel}}
    scheduler = MingStreamingTalkerScheduler(
        model_path=str(tmp_path / "model"),
        voice="DB30",
        talker=object(),
        sample_rate=44100,
    )
    with pytest.raises(ValueError, match="default voice 'DB30' not found"):
        scheduler._validate_voice_presets(voice_dict, str(manifest), str(talker_dir))


def test_scheduler_validate_raises_when_wav_missing(tmp_path: Path):
    talker_dir = tmp_path / "talker"
    wav_rel = "spks/DB30.wav"
    manifest = _write_voice_manifest(
        talker_dir,
        {"DB30": {"prompt_text": "x", "prompt_wav_path": wav_rel}},
        wav_files=None,
    )
    voice_dict = {"DB30": {"prompt_text": "x", "prompt_wav_path": wav_rel}}
    scheduler = MingStreamingTalkerScheduler(
        model_path=str(tmp_path / "model"),
        voice="DB30",
        talker=object(),
        sample_rate=44100,
    )
    with pytest.raises(FileNotFoundError, match="missing prompt wav"):
        scheduler._validate_voice_presets(voice_dict, str(manifest), str(talker_dir))


def test_scheduler_validate_raises_value_error_on_bad_manifest(tmp_path: Path):
    talker_dir = tmp_path / "talker"
    manifest = _write_voice_manifest(
        talker_dir,
        {"DB30": {"prompt_text": "x"}},  # missing prompt_wav_path key
    )
    voice_dict = {"DB30": {"prompt_text": "x"}}
    scheduler = MingStreamingTalkerScheduler(
        model_path=str(tmp_path / "model"),
        voice="DB30",
        talker=object(),
        sample_rate=44100,
    )
    with pytest.raises(ValueError, match="missing prompt_wav_path"):
        scheduler._validate_voice_presets(voice_dict, str(manifest), str(talker_dir))


def _bare_talker(voice_json_dict: dict | None = None) -> MingOmniTalker:
    talker = object.__new__(MingOmniTalker)
    talker.voice_json_dict = voice_json_dict if voice_json_dict is not None else {}
    return talker


def test_omni_audio_generation_unknown_voice_no_prompt_wav_raises():
    talker = _bare_talker()
    gen = MingOmniTalker.omni_audio_generation(
        talker,
        tts_text="hi",
        voice_name="DB30",
        prompt_wav_path=None,
    )
    with pytest.raises(ValueError, match="not found in loaded voice presets"):
        next(gen)


def test_omni_audio_generation_no_voice_no_prompt_wav_raises():
    talker = _bare_talker()
    gen = MingOmniTalker.omni_audio_generation(
        talker,
        tts_text="hi",
        voice_name=None,
        prompt_wav_path=None,
    )
    with pytest.raises(ValueError, match="requires either voice_name"):
        next(gen)


def test_omni_audio_generation_explicit_prompt_wav_overrides_preset(monkeypatch):
    """An explicit prompt_wav_path must take precedence over a registered
    preset (callers use this to override the voice for a single request)."""
    talker = _bare_talker(
        voice_json_dict={
            "DB30": {
                "prompt_text": "preset text",
                "prompt_wav_path": "/preset/wav.wav",
            }
        }
    )

    captured = {}

    def fake_run(
        self,
        text,
        prompt,
        instruction,
        spk_emb,
        audio_detok,
        prompt_text,
        prompt_wav_lat,
        prompt_wav_emb,
        **kwargs,
    ):
        captured["prompt_text"] = prompt_text
        if False:
            yield None

    def fake_get_prompt_emb(self, prompt_wav_path, *args, **kwargs):
        captured["prompt_wav_path"] = prompt_wav_path
        return None, None, None

    monkeypatch.setattr(MingOmniTalker, "_run_tts_segments", fake_run)
    monkeypatch.setattr(MingOmniTalker, "get_prompt_emb", fake_get_prompt_emb)
    monkeypatch.setattr(
        MingOmniTalker, "initial_graph", lambda self: None, raising=False
    )

    import torch as _torch

    class _NullStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(_torch.cuda, "stream", lambda s: _NullStream())
    monkeypatch.setattr(
        _torch.cuda, "Stream", lambda device=None: object(), raising=False
    )
    monkeypatch.setattr(
        MingOmniTalker,
        "device",
        property(lambda self: "cuda:0"),
        raising=False,
    )

    list(
        MingOmniTalker.omni_audio_generation(
            talker,
            tts_text="hi",
            voice_name="DB30",
            prompt_wav_path="/explicit/override.wav",
            prompt_text="explicit text",
        )
    )

    assert captured["prompt_wav_path"] == "/explicit/override.wav"
    assert captured["prompt_text"] == "explicit text"


def _fake_detokenizer(sample_rate=44100, vae_patch_size=4, hop_size=480):
    encoder = SimpleNamespace(patch_size=vae_patch_size, hop_size=hop_size)
    config = SimpleNamespace(sample_rate=sample_rate)
    return SimpleNamespace(encoder=encoder, config=config)


def test_duration_capped_steps_short_text_uses_floor():
    talker = object.__new__(MingOmniTalker)
    talker.patch_size = 2

    detok = _fake_detokenizer()
    seconds_per_step = (2 * 4 * 480) / 44100  # ≈ 0.0871
    expected = max(1, int(2.0 / seconds_per_step))  # short-text floor = 2.0s

    capped = talker.duration_capped_steps(
        text_len=1, audio_detokenizer=detok, requested_max_steps=10_000
    )
    assert capped == expected


def test_duration_capped_steps_long_text_uses_per_char_rate():
    talker = object.__new__(MingOmniTalker)
    talker.patch_size = 2
    detok = _fake_detokenizer()

    text_len = 100
    seconds_per_step = (2 * 4 * 480) / 44100
    max_duration_s = max(2.0, text_len * (5818.0 / 16000.0))
    expected = max(1, int(max_duration_s / seconds_per_step))

    capped = talker.duration_capped_steps(
        text_len=text_len,
        audio_detokenizer=detok,
        requested_max_steps=10_000,
    )
    assert capped == expected


def test_duration_capped_steps_respects_requested_ceiling():
    talker = object.__new__(MingOmniTalker)
    talker.patch_size = 2
    detok = _fake_detokenizer()

    capped = talker.duration_capped_steps(
        text_len=100, audio_detokenizer=detok, requested_max_steps=5
    )
    assert capped == 5


def test_duration_capped_steps_no_detokenizer_is_passthrough():
    talker = object.__new__(MingOmniTalker)
    talker.patch_size = 2
    capped = talker.duration_capped_steps(
        text_len=100, audio_detokenizer=None, requested_max_steps=42
    )
    assert capped == 42


def test_duration_capped_steps_missing_encoder_attr_is_passthrough():
    talker = object.__new__(MingOmniTalker)
    talker.patch_size = 2
    detok = SimpleNamespace(
        encoder=SimpleNamespace(patch_size=4),
        config=SimpleNamespace(sample_rate=44100),
    )
    capped = talker.duration_capped_steps(
        text_len=100, audio_detokenizer=detok, requested_max_steps=42
    )
    assert capped == 42


def _stub_for_generate(talker: MingOmniTalker, num_steps_before_stop: int):
    """Wire MingOmniTalker.generate's collaborators with deterministic stubs."""
    import torch as _torch

    target_device = _torch.device("cpu")

    talker.his_patch_size = 1
    talker.patch_size = 1
    talker.latent_dim = 1
    talker.model = SimpleNamespace(
        config=SimpleNamespace(),
        device=target_device,
    )

    class _NoopCache:
        def __init__(self):
            self.calls = []

        def reset(self):
            pass

        def get_seq_length(self):
            return 1

    pool_state = {
        "tuple": (_NoopCache(), None, None, None, None, None, None),
    }
    talker.model_graph_pool = SimpleNamespace(
        get=lambda: pool_state["tuple"],
        put=lambda x: pool_state.__setitem__("tuple", x),
    )

    def _fake_forward(**kwargs):
        return SimpleNamespace(
            hidden_states=(_torch.zeros(1, 1, 1, device=target_device),),
        )

    talker.model = _fake_forward
    type(talker).device = property(lambda self: target_device)

    step_counter = {"n": 0}

    def _execute(hidden_out, his_lat, cfg, sigma, temperature, abort_event):
        n = step_counter["n"]
        step_counter["n"] += 1
        gen_lat = _torch.full((1, 1, 1), float(n), device=target_device)
        new_embeds = _torch.zeros(1, 1, 1, device=target_device)
        stop_value = (
            1.0
            if (num_steps_before_stop is not None and n == num_steps_before_stop)
            else 0.0
        )
        stop_out = _torch.tensor([[0.0, stop_value]], device=target_device)
        return gen_lat, new_embeds, stop_out

    talker.sampler_pool = SimpleNamespace(execute=_execute)


def test_generate_emits_final_true_when_stop_token_fires(monkeypatch):
    """Stop-token exit terminates with last_chunk=True."""
    import torch as _torch

    talker = object.__new__(MingOmniTalker)
    _stub_for_generate(talker, num_steps_before_stop=4)

    monkeypatch.setattr(
        "sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker." "StaticCache",
        lambda **kw: object(),
    )

    input_ids = _torch.zeros(1, 2, dtype=_torch.long)
    inputs_embeds = _torch.zeros(1, 2, 1)

    yields = list(
        MingOmniTalker.generate(
            talker,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            min_new_token=0,
            max_decode_steps=20,
        )
    )
    assert len(yields) >= 2
    flags = [bool(flag) for _, flag in yields]
    assert flags[-1] is True
    assert all(flag is False for flag in flags[:-1])


def test_generate_emits_final_true_when_duration_cap_hits(monkeypatch):
    """Loop exit via effective_max_decode_steps ceiling must still emit a
    last_chunk=True so the streaming VAE flushes its tail."""
    import torch as _torch

    talker = object.__new__(MingOmniTalker)
    # never trigger stop_out=1.0 inside the loop
    _stub_for_generate(talker, num_steps_before_stop=None)

    monkeypatch.setattr(
        "sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker." "StaticCache",
        lambda **kw: object(),
    )

    input_ids = _torch.zeros(1, 2, dtype=_torch.long)
    inputs_embeds = _torch.zeros(1, 2, 1)

    yields = list(
        MingOmniTalker.generate(
            talker,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            min_new_token=0,
            max_decode_steps=3,
        )
    )
    assert len(yields) == 3
    flags = [bool(flag) for _, flag in yields]
    assert flags == [False, False, True]
