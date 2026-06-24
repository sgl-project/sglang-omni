# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
import sys
import types

import numpy as np
import pytest
import torch

from sglang_omni.client.audio import encode_audio, encode_wav
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.models.moss_tts_local.audio_tokenizer import MossTTSLocalAudioTokenizer
from sglang_omni.models.moss_tts_local.config import (
    MossTTSLocalColocatedPipelineConfig,
    MossTTSLocalPipelineConfig,
    MossTTSLocalSplitPipelineConfig,
)
from sglang_omni.models.moss_tts_local.local_transformer import (
    MossTTSLocalTransformer,
    _rotate_half_interleaved,
)
from sglang_omni.models.moss_tts_local.payload_types import (
    MossTTSLocalState,
    moss_tts_local_special_token_defaults,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    MossTTSLocalSGLangRequestData,
    apply_sglang_moss_tts_local_result,
    build_generation_kwargs,
    build_moss_tts_local_state,
    clear_moss_tts_local_preprocessing_context,
    preprocess_moss_tts_local_payload,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.audio_payload import audio_waveform_payload

N_VQ = 12


class _FakeEncodedAudio:
    def __init__(self, audio_codes: torch.Tensor, audio_codes_lengths: torch.Tensor):
        self.audio_codes = audio_codes
        self.audio_codes_lengths = audio_codes_lengths


class _FakeAudioTokenizerModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(sampling_rate=48000, number_channels=2)
        self.calls: list[tuple[list[torch.Tensor], int]] = []

    def batch_encode(self, wavs: list[torch.Tensor], *, num_quantizers: int):
        assert all(wav.ndim == 2 and wav.shape[0] == 2 for wav in wavs)
        self.calls.append((wavs, int(num_quantizers)))
        max_len = max(int(wav.shape[-1]) for wav in wavs)
        audio_codes = torch.zeros(num_quantizers, len(wavs), max_len, dtype=torch.long)
        audio_codes_lengths = torch.tensor(
            [int(wav.shape[-1]) for wav in wavs], dtype=torch.long
        )
        for index, wav in enumerate(wavs):
            length = int(wav.shape[-1])
            base = int(wav[0, 0].item()) if wav.numel() else 0
            audio_codes[:, index, :length] = (
                torch.arange(num_quantizers, dtype=torch.long).view(-1, 1)
                + base
                + torch.arange(length, dtype=torch.long).view(1, -1)
            )
        return _FakeEncodedAudio(audio_codes, audio_codes_lengths)


# Local transformer numerics


def _hf_rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Verbatim port of the upstream gpt2_decoder.rotate_half."""
    even = hidden_states[..., ::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(hidden_states)


def _reference_full_forward(
    module: MossTTSLocalTransformer, inputs: torch.Tensor
) -> torch.Tensor:
    """Full-sequence forward replicating the upstream eager math.

    ``inputs`` is ``[batch, seq, hidden]``; positions are 0..seq-1 with a
    causal mask, interleaved RoPE, fp32 softmax via explicit matmuls.
    """
    batch, seq, hidden = inputs.shape
    num_heads = module.num_heads
    head_dim = module.head_dim

    inv_freq = 1.0 / (
        1_000_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)

    x = inputs
    for block in module.h:
        normed = block.ln_1(x)
        qkv = block.attn.c_attn(normed)
        query, key, value = qkv.split(hidden, dim=-1)
        query = query.view(batch, seq, num_heads, head_dim)
        key = key.view(batch, seq, num_heads, head_dim)
        value = value.view(batch, seq, num_heads, head_dim)
        cos_b = cos.view(1, seq, 1, head_dim)
        sin_b = sin.view(1, seq, 1, head_dim)
        query = query * cos_b + _hf_rotate_half(query) * sin_b
        key = key * cos_b + _hf_rotate_half(key) * sin_b

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / head_dim**0.5
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, value).transpose(1, 2).reshape(batch, seq, hidden)
        x = x + block.attn.c_proj(attn)
        x = x + block.mlp(block.ln_2(x))
    return module.ln_f(x)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_local_transformer_incremental_matches_full_recompute(num_layers: int):
    torch.manual_seed(0)
    module = MossTTSLocalTransformer(
        hidden_size=64,
        num_heads=4,
        inner_size=96,
        num_layers=num_layers,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    )
    module.eval()
    batch, seq = 3, N_VQ + 1
    inputs = torch.randn(batch, seq, 64)

    reference = _reference_full_forward(module, inputs)
    stepped = torch.stack([module.step(inputs[:, t], t) for t in range(seq)], dim=1)
    torch.testing.assert_close(stepped, reference, rtol=1e-4, atol=1e-5)


def test_local_transformer_kv_cache_grows_with_batch():
    module = MossTTSLocalTransformer(
        hidden_size=32,
        num_heads=2,
        inner_size=48,
        num_layers=1,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    )
    out_small = module.step(torch.randn(2, 32), 0)
    assert out_small.shape == (2, 32)
    out_large = module.step(torch.randn(8, 32), 0)
    assert out_large.shape == (8, 32)
    assert module._kv_capacity >= 8


def test_local_transformer_rejects_out_of_range_position():
    module = MossTTSLocalTransformer(
        hidden_size=32,
        num_heads=2,
        inner_size=48,
        num_layers=1,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    )
    with pytest.raises(ValueError):
        module.step(torch.randn(1, 32), N_VQ + 1)


def test_rotate_half_interleaved_matches_upstream():
    x = torch.randn(5, 4, 8)
    torch.testing.assert_close(_rotate_half_interleaved(x), _hf_rotate_half(x))


# MOSS-Audio-Tokenizer-v2 wrapper


def test_audio_tokenizer_returns_row_major_trimmed_codes():
    model = _FakeAudioTokenizerModel()
    tokenizer = MossTTSLocalAudioTokenizer(model, device="cpu")
    wavs = [
        torch.full((1, 3), 10.0),
        torch.full((1, 5), 20.0),
    ]

    encoded = tokenizer.encode_wavs(wavs, 48000, num_quantizers=N_VQ)

    assert len(model.calls) == 1
    assert model.calls[0][1] == N_VQ
    assert [tuple(wav.shape) for wav in model.calls[0][0]] == [(2, 3), (2, 5)]
    assert [tuple(codes.shape) for codes in encoded] == [(3, N_VQ), (5, N_VQ)]


def test_audio_tokenizer_batches_mixed_sample_rates(monkeypatch):
    model = _FakeAudioTokenizerModel()
    tokenizer = MossTTSLocalAudioTokenizer(model, device="cpu")
    resample_calls = []

    def fake_load(path):
        if path == "ref-16k.wav":
            return torch.full((1, 4), 16.0), 16000
        return torch.full((1, 6), 48.0), 48000

    def fake_resample(waveform, *, orig_freq, new_freq):
        resample_calls.append((orig_freq, new_freq))
        return waveform + 1

    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        types.SimpleNamespace(
            load=fake_load,
            functional=types.SimpleNamespace(resample=fake_resample),
        ),
    )

    encoded = tokenizer.encode_paths(
        ["ref-16k.wav", "ref-48k.wav"],
        num_quantizers=N_VQ,
    )

    assert len(model.calls) == 1
    assert resample_calls == [(16000, 48000)]
    assert [tuple(wav.shape) for wav in model.calls[0][0]] == [(2, 4), (2, 6)]
    assert [tuple(codes.shape) for codes in encoded] == [(4, N_VQ), (6, N_VQ)]


def test_audio_tokenizer_path_resamples_before_channel_fold(monkeypatch):
    model = _FakeAudioTokenizerModel()
    tokenizer = MossTTSLocalAudioTokenizer(model, device="cpu")
    observed_resample_shapes = []

    def fake_load(path):
        return torch.ones(1, 4), 16000

    def fake_resample(waveform, *, orig_freq, new_freq):
        observed_resample_shapes.append(tuple(waveform.shape))
        return waveform

    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        types.SimpleNamespace(
            load=fake_load,
            functional=types.SimpleNamespace(resample=fake_resample),
        ),
    )

    tokenizer.encode_paths(["ref.wav"], num_quantizers=N_VQ)

    assert observed_resample_shapes == [(1, 4)]
    scale = 10.0 ** (-3.0 / 20.0)
    expected = torch.ones(1, 4).repeat(2, 1) * scale
    torch.testing.assert_close(model.calls[0][0][0], expected)


def test_audio_tokenizer_matches_processor_waveform_prep_for_stereo():
    model = _FakeAudioTokenizerModel()
    tokenizer = MossTTSLocalAudioTokenizer(model, device="cpu")
    stereo = torch.stack(
        [torch.full((4,), 1.0), torch.full((4,), 3.0)],
        dim=0,
    )

    tokenizer.encode_wavs([stereo], 48000, num_quantizers=N_VQ)

    prepared = model.calls[0][0][0]
    expected = stereo * (10.0 ** (-3.0 / 20.0))
    torch.testing.assert_close(prepared, expected)


def test_audio_tokenizer_matches_processor_waveform_prep_for_mono_and_extra_channels():
    model = _FakeAudioTokenizerModel()
    tokenizer = MossTTSLocalAudioTokenizer(model, device="cpu")
    mono = torch.full((1, 4), 2.0)
    three_channel = torch.stack(
        [torch.full((4,), 1.0), torch.full((4,), 3.0), torch.full((4,), 5.0)],
        dim=0,
    )

    tokenizer.encode_wavs([mono, three_channel], 48000, num_quantizers=N_VQ)

    scale = 10.0 ** (-3.0 / 20.0)
    torch.testing.assert_close(model.calls[0][0][0], mono.repeat(2, 1) * scale)
    torch.testing.assert_close(model.calls[0][0][1], three_channel[:2] * scale)


def test_audio_tokenizer_reference_encode_uses_processor_stereo_contract():
    model = _FakeAudioTokenizerModel()
    model.config.number_channels = 1
    tokenizer = MossTTSLocalAudioTokenizer(model, device="cpu")
    mono = torch.full((1, 4), 2.0)

    tokenizer.encode_wavs([mono], 48000, num_quantizers=N_VQ)

    scale = 10.0 ** (-3.0 / 20.0)
    torch.testing.assert_close(model.calls[0][0][0], mono.repeat(2, 1) * scale)


def test_audio_tokenizer_loader_matches_processor_codec_weight_dtype(monkeypatch):
    from contextlib import nullcontext

    from sglang_omni.models.moss_tts_local import audio_tokenizer as audio_tokenizer_mod
    from sglang_omni.models.moss_tts_local.audio_tokenizer import (
        load_moss_tts_local_audio_tokenizer,
    )

    class _FakeLoadedCodec(_FakeAudioTokenizerModel):
        def __init__(self):
            super().__init__()
            self.eval_called = False
            self.to_device = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.to_device = device
            return self

    loaded_kwargs = {}
    loaded_model = _FakeLoadedCodec()

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            loaded_kwargs["model_path"] = model_path
            loaded_kwargs.update(kwargs)
            return loaded_model

    monkeypatch.setattr(
        audio_tokenizer_mod,
        "resolve_moss_checkpoint",
        lambda model_path: f"/resolved/{model_path}",
    )
    monkeypatch.setattr(
        audio_tokenizer_mod,
        "moss_transformers_processor_compat",
        nullcontext,
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=_FakeAutoModel),
    )

    tokenizer = load_moss_tts_local_audio_tokenizer("codec", device="cuda:7")

    assert tokenizer.model is loaded_model
    assert loaded_model.eval_called
    assert loaded_model.to_device == "cuda:7"
    assert loaded_kwargs == {
        "model_path": "/resolved/codec",
        "trust_remote_code": True,
        "codec_weight_dtype": "bf16",
    }


# Registry / config


def test_registry_resolves_local_architecture():
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("MossTTSLocalModel")
    assert config_cls is MossTTSLocalPipelineConfig
    # The Delay family keeps its own architecture.
    delay_cls = PIPELINE_CONFIG_REGISTRY.get_config("MossTTSDelayModel")
    assert delay_cls is not MossTTSLocalPipelineConfig


def test_pipeline_stage_wiring():
    config = MossTTSLocalPipelineConfig(model_path="OpenMOSS-Team/moss-local-test")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    stages = {stage.name: stage for stage in config.stages}
    assert set(stages) == {"preprocessing", "tts_engine", "vocoder"}
    assert stages["preprocessing"].next == "tts_engine"
    assert stages["tts_engine"].next == "vocoder"
    assert stages["vocoder"].terminal
    for stage in stages.values():
        assert "moss_tts_local" in stage.factory
    assert stages["preprocessing"].process == "pipeline"
    assert stages["preprocessing"].gpu == 0
    assert stages["preprocessing"].factory_args["device"] == "cuda:0"
    assert stages["preprocessing"].factory_args["ref_audio_cache_max_items"] == 8192
    assert config.supports_uploaded_voice_references() is True
    assert stages["tts_engine"].process == "pipeline"
    assert stages["tts_engine"].gpu == 0
    tts_engine_runtime = stages["tts_engine"].runtime
    assert tts_engine_runtime.resources.total_gpu_memory_fraction == pytest.approx(0.90)
    assert tts_engine_runtime.sglang_server_args.mem_fraction_static is None
    assert stages["tts_engine"].factory_args["codec_mem_reserve"] == pytest.approx(0.15)
    assert stages["vocoder"].process == "pipeline"
    assert stages["vocoder"].gpu == 0
    assert stages["vocoder"].factory_args["device"] == "cuda:0"

    placement = build_stage_placement_plan(config)
    assert placement.stages["tts_engine"].gpu_ids == (0,)
    assert placement.stages["preprocessing"].gpu_ids == (0,)
    assert placement.stages["vocoder"].gpu_ids == (0,)

    colocated = MossTTSLocalColocatedPipelineConfig(
        model_path="OpenMOSS-Team/moss-local-test"
    )
    colocated_stages = {stage.name: stage for stage in colocated.stages}
    assert colocated_stages["preprocessing"].factory_args["device"] == "cuda:0"
    assert (
        colocated_stages["preprocessing"].factory_args["ref_audio_cache_max_items"]
        == 8192
    )
    assert colocated_stages["vocoder"].factory_args["device"] == "cuda:0"

    split = MossTTSLocalSplitPipelineConfig(model_path="OpenMOSS-Team/moss-local-test")
    split_stages = {stage.name: stage for stage in split.stages}
    assert split_stages["preprocessing"].factory_args["device"] == "cuda:1"
    assert split_stages["tts_engine"].factory_args["gpu_id"] == 0
    split_runtime = split_stages["tts_engine"].runtime
    assert split_runtime.resources.total_gpu_memory_fraction is None
    assert split_runtime.sglang_server_args.mem_fraction_static == pytest.approx(0.85)
    assert split_stages["vocoder"].factory_args["device"] == "cuda:1"


def _install_fake_moss_ar_factory(
    monkeypatch,
    *,
    process_memory_bytes: int | None,
):
    pytest.importorskip("PIL")

    from sglang_omni.models.moss_tts_local import stages
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling import omni_scheduler, sglang_backend
    from sglang_omni.utils import gpu_memory as gpu_memory_utils

    infrastructure_calls = []
    process_memory_queries = []

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        return types.SimpleNamespace(
            model_path=model_path,
            context_length=context_length,
            **kwargs,
        )

    class FakeModelRunner:
        model = object()

    model_worker = types.SimpleNamespace(
        model_runner=FakeModelRunner(),
        model_config=types.SimpleNamespace(),
    )

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        infrastructure_calls.append(
            {
                "mem_fraction_static": server_args.mem_fraction_static,
                "gpu_id": gpu_id,
                "total_gpu_memory_fraction": kwargs.get("total_gpu_memory_fraction"),
            }
        )
        return (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            model_worker.model_config,
        )

    class FakeMossRunner:
        def __init__(self, *args, **kwargs):
            self.stream_outbox = None

        def set_stream_outbox(self, outbox):
            self.stream_outbox = outbox

    class FakeScheduler:
        def __init__(self, **kwargs):
            self.outbox = object()
            self.kwargs = kwargs

    fake_runner_module = types.SimpleNamespace(MossTTSLocalModelRunner=FakeMossRunner)
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.moss_tts_local.model_runner",
        fake_runner_module,
    )
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        stages,
        "make_moss_tts_local_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        stages, "resolve_moss_checkpoint", lambda model_path: model_path
    )
    monkeypatch.setattr(omni_scheduler, "OmniScheduler", FakeScheduler)

    def fake_get_process_gpu_memory_bytes(gpu_id):
        process_memory_queries.append(gpu_id)
        return process_memory_bytes

    monkeypatch.setattr(
        gpu_memory_utils,
        "get_process_gpu_memory_bytes",
        fake_get_process_gpu_memory_bytes,
    )

    return stages, infrastructure_calls, process_memory_queries


def test_colocated_moss_ar_factory_threads_effective_budget(monkeypatch):
    stages, infrastructure_calls, process_memory_queries = (
        _install_fake_moss_ar_factory(
            monkeypatch,
            process_memory_bytes=1024,
        )
    )

    stages.create_sglang_tts_engine_executor(
        "dummy",
        server_args_overrides={"disable_cuda_graph": True},
        total_gpu_memory_fraction=0.90,
        codec_mem_reserve=0.05,
    )

    assert infrastructure_calls == [
        {
            "mem_fraction_static": pytest.approx(0.85),
            "gpu_id": 0,
            "total_gpu_memory_fraction": pytest.approx(0.85),
        }
    ]
    assert process_memory_queries == [0]


def test_colocated_moss_ar_factory_uses_upstream_profile_without_process_accounting(
    monkeypatch,
):
    stages, infrastructure_calls, process_memory_queries = (
        _install_fake_moss_ar_factory(
            monkeypatch,
            process_memory_bytes=None,
        )
    )

    stages.create_sglang_tts_engine_executor(
        "dummy",
        server_args_overrides={"disable_cuda_graph": True},
        total_gpu_memory_fraction=0.90,
        codec_mem_reserve=0.05,
    )

    assert infrastructure_calls == [
        {
            "mem_fraction_static": pytest.approx(0.85),
            "gpu_id": 0,
            "total_gpu_memory_fraction": None,
        }
    ]
    assert process_memory_queries == [0]


def test_colocated_moss_ar_factory_accepts_explicit_effective_budget():
    pytest.importorskip("PIL")

    from sglang_omni.models.moss_tts_local import stages

    budget = stages._apply_colocated_ar_memory_budget(
        {"mem_fraction_static": 0.70},
        total_gpu_memory_fraction=0.90,
        codec_mem_reserve=0.05,
    )
    assert budget.effective_total_gpu_memory_fraction == pytest.approx(0.70)
    assert budget.applied_codec_mem_reserve == pytest.approx(0.20)

    with pytest.raises(ValueError, match="cannot exceed"):
        stages._apply_colocated_ar_memory_budget(
            {"mem_fraction_static": 0.95},
            total_gpu_memory_fraction=0.90,
            codec_mem_reserve=0.05,
        )


def test_special_token_defaults_match_v15_checkpoint():
    defaults = dict(moss_tts_local_special_token_defaults())
    assert defaults["audio_start_token_id"] == 151669
    assert defaults["audio_end_token_id"] == 151670
    assert defaults["audio_user_slot_token_id"] == 151654
    assert defaults["audio_assistant_slot_token_id"] == 151656
    assert defaults["audio_pad_code"] == 1024


# Generation kwargs / state


def test_build_generation_kwargs_defaults():
    kwargs = build_generation_kwargs({}, tts_params={})
    assert kwargs["max_new_tokens"] == 4096
    assert kwargs["text_temperature"] == 1.0
    assert kwargs["text_top_p"] == 1.0
    assert kwargs["text_top_k"] == 50
    assert kwargs["audio_temperature"] == 1.7
    assert kwargs["audio_top_p"] == 0.8
    assert kwargs["audio_top_k"] == 25
    assert kwargs["audio_repetition_penalty"] == 1.0


def test_build_generation_kwargs_explicit_overrides():
    kwargs = build_generation_kwargs(
        {"temperature": 0.9, "top_p": 0.7},
        tts_params={
            "explicit_generation_params": ["temperature", "top_p"],
            "audio_top_k": 11,
        },
    )
    assert kwargs["text_temperature"] == 0.9
    assert kwargs["audio_temperature"] == 0.9
    assert kwargs["audio_top_p"] == 0.7
    assert kwargs["audio_top_k"] == 11


def test_state_round_trip():
    state = MossTTSLocalState(
        text="hello",
        language="English",
        token_count=125,
        audio_codes=torch.zeros((3, N_VQ), dtype=torch.long),
        sample_rate=48000,
        prompt_tokens=7,
        completion_tokens=3,
    )
    restored = MossTTSLocalState.from_dict(state.to_dict())
    assert restored.text == "hello"
    assert restored.language == "English"
    assert restored.token_count == 125
    assert restored.sample_rate == 48000
    assert torch.as_tensor(restored.audio_codes).shape == (3, N_VQ)


def test_build_state_token_count_and_language():
    payload = StagePayload(
        request_id="r0",
        request=OmniRequest(
            inputs={"text": "${token:50} hello world", "references": []},
            params={"language": "English"},
            metadata={},
        ),
        data={},
    )
    state = build_moss_tts_local_state(payload)
    assert state.token_count == 50
    assert state.text == "hello world"
    assert state.language == "English"


# Preprocessing handoff + result adapter


class _FakeProcessor:
    """Builds deterministic [1, T, 13] rows from the message text length."""

    model_config = type("_FakeModelConfig", (), {"n_vq": N_VQ})()

    @staticmethod
    def build_user_message(**kwargs):
        return dict(kwargs, role="user")

    def __call__(self, conversations, mode):
        assert mode == "generation"
        message = conversations[0][0]
        text = str(message.get("text", ""))
        seq = max(4, len(text) % 7 + 4)
        rows = torch.full((1, seq, N_VQ + 1), 1024, dtype=torch.long)
        rows[0, :, 0] = torch.arange(seq)
        rows[0, -1, 0] = 151669  # trailing audio_start row
        return {"input_ids": rows}


def _payload(text: str = "hello") -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs={"text": text}, params={}, metadata={}),
        data={},
    )


def test_create_preprocessing_executor_env_toggle(monkeypatch):
    from sglang_omni.models.moss_tts_local import request_builders as rb
    from sglang_omni.models.moss_tts_local import stages

    class _FakeAudioTokenizer:
        def encode_paths(self, paths, *, num_quantizers):
            assert num_quantizers == N_VQ
            return []

    monkeypatch.setattr(
        stages, "_load_moss_tts_local_processor", lambda *a, **k: _FakeProcessor()
    )
    monkeypatch.setattr(
        stages,
        "load_moss_tts_local_audio_tokenizer",
        lambda *a, **k: _FakeAudioTokenizer(),
    )

    # MOSS_REF_AUDIO_CACHE=0 disables the wrapper at startup (kill switch).
    monkeypatch.setenv("MOSS_REF_AUDIO_CACHE", "0")
    stages.create_preprocessing_executor("model", device="cpu")
    assert not isinstance(
        rb._PREPROCESSING_CONTEXT.reference_encoder, stages.CachedReferenceEncoder
    )

    # Unset -> kwarg default (True) -> wrapped.
    monkeypatch.delenv("MOSS_REF_AUDIO_CACHE")
    stages.create_preprocessing_executor("model", device="cpu")
    assert isinstance(
        rb._PREPROCESSING_CONTEXT.reference_encoder, stages.CachedReferenceEncoder
    )
    assert rb._PREPROCESSING_CONTEXT.reference_encoder._cache.max_size == 8192


def test_create_preprocessing_executor_uses_model_config_codec_path(monkeypatch):
    from sglang_omni.models.moss_tts_local import stages

    class _FakeAudioTokenizer:
        def encode_paths(self, paths, *, num_quantizers):
            return []

    processor = _FakeProcessor()
    processor.model_config = types.SimpleNamespace(
        n_vq=N_VQ,
        audio_tokenizer_name_or_path="codec-from-model-config",
    )
    loaded_codec_paths = []

    def fake_load_audio_tokenizer(model_path, *, device):
        loaded_codec_paths.append(model_path)
        return _FakeAudioTokenizer()

    monkeypatch.setattr(
        stages, "_load_moss_tts_local_processor", lambda model_path: processor
    )
    monkeypatch.setattr(
        stages,
        "load_moss_tts_local_audio_tokenizer",
        fake_load_audio_tokenizer,
    )

    stages.create_preprocessing_executor("model", device="cpu")

    assert loaded_codec_paths == ["codec-from-model-config"]


def test_preprocess_and_result_adapter():
    set_moss_tts_local_preprocessing_context(processor=_FakeProcessor())
    try:
        payload = preprocess_moss_tts_local_payload(_payload())
        assert payload.data.get("_moss_tts_local_prepared_request") == "req-1"

        from sglang_omni.models.moss_tts_local.request_builders import (
            pop_prepared_moss_tts_local_request,
        )

        prepared = pop_prepared_moss_tts_local_request(payload)
        assert prepared is not None
        assert prepared.prompt_rows.ndim == 2
        assert prepared.prompt_rows.shape[1] == N_VQ + 1
        assert len(prepared.input_ids_list) == prepared.prompt_rows.shape[0]

        data = MossTTSLocalSGLangRequestData(
            input_ids=prepared.input_ids,
            max_new_tokens=16,
            temperature=0.0,
            output_ids=[],
            state=prepared.state,
            prompt_rows=prepared.prompt_rows,
            stage_payload=payload,
            engine_start_s=0.0,
        )
        data.output_rows = [
            torch.cat([torch.tensor([151656]), torch.arange(N_VQ, dtype=torch.long)])
            for _ in range(3)
        ]
        result = apply_sglang_moss_tts_local_result(payload, data)
        codes = torch.as_tensor(result.data["audio_codes"])
        assert codes.shape == (3, N_VQ)
        assert result.data["completion_tokens"] == 3
        assert result.data["prompt_tokens"] == prepared.prompt_rows.shape[0]
    finally:
        clear_moss_tts_local_preprocessing_context()


def test_result_adapter_empty_generation():
    payload = _payload()
    data = MossTTSLocalSGLangRequestData(
        input_ids=torch.zeros(4, dtype=torch.long),
        max_new_tokens=16,
        temperature=0.0,
        output_ids=[],
        prompt_rows=torch.full((4, N_VQ + 1), 1024, dtype=torch.long),
        stage_payload=payload,
        engine_start_s=0.0,
    )
    result = apply_sglang_moss_tts_local_result(payload, data)
    codes = torch.as_tensor(result.data["audio_codes"])
    assert codes.shape == (0, N_VQ)


# Repetition penalty parity


def test_audio_repetition_penalty_mask_matches_upstream_semantics():
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner

    logits = torch.tensor(
        [[2.0, -1.0, 0.5, 3.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
    )
    token_presence = torch.tensor(
        [
            [True, False, True, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    expected = logits.clone()
    penalty = 1.5
    expected[0, 0] = expected[0, 0] / penalty  # positive -> divide
    expected[0, 2] = expected[0, 2] / penalty

    MossTTSLocalModelRunner._apply_audio_repetition_penalty_mask(
        logits, token_presence, torch.tensor([penalty, 1.0])
    )
    torch.testing.assert_close(logits, expected)

    # Negative scores multiply.
    logits2 = torch.tensor([[-2.0, 1.0]], dtype=torch.float32)
    MossTTSLocalModelRunner._apply_audio_repetition_penalty_mask(
        logits2, torch.tensor([[True, False]]), torch.tensor([2.0])
    )
    torch.testing.assert_close(
        logits2, torch.tensor([[-4.0, 1.0]], dtype=torch.float32)
    )


def test_row_radix_token_ids_hash_rows_and_keep_eos():
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner

    end_id = 151670
    slot_id = 151656
    rows = torch.full((3, N_VQ + 1), 7, dtype=torch.long)
    rows[:, 0] = torch.tensor([slot_id, end_id, slot_id])
    rows[2, 1:] = torch.arange(N_VQ)
    next_text = rows[:, 0].clone()

    out = MossTTSLocalModelRunner._row_radix_token_ids(rows, next_text, end_id)
    # Post-090c9cf the generated-row key is the capture-safe GPU polynomial hash
    # (gpu_radix_row_hash), not the host blake2b; assert the spec the key must
    # satisfy, not a specific digest. Exact hash semantics live in
    # test_radix_hash.py / docs/design/gpu_radix_hash.md.
    assert int(out[1]) == end_id  # stop decision keeps the raw eos id
    assert int(out[0]) != int(out[2])  # full-row dependence: codes differ -> keys
    assert int(out[0]) != slot_id  # no longer the constant slot id
    # Hashed (non-eos) rows fold below the special-token band so the scheduler's
    # vocab-boundary finish never trips on a generated frame.
    assert int(out[0]) < 151643
    assert int(out[2]) < 151643
    assert all(0 <= int(v) < 151936 for v in out)


def test_audio_history_presence_mask_excludes_prompt_rows():
    from types import SimpleNamespace

    from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeStatePool

    model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.zeros(2, 4, dtype=torch.bfloat16)
        ),
        config=SimpleNamespace(n_vq=N_VQ, audio_vocab_size=1024),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    row = pool.acquire_row("rid")
    prompt_row = torch.cat(
        [torch.tensor([151656]), torch.full((N_VQ,), 99, dtype=torch.long)]
    )
    generated_row = torch.cat(
        [torch.tensor([151656]), torch.arange(N_VQ, dtype=torch.long)]
    )

    pool.update_audio_history(torch.tensor([row]), generated_row.reshape(1, -1))

    assert bool(pool.audio_token_presence[row, 0, 0])
    assert not bool(pool.audio_token_presence[row, 0, int(prompt_row[1])])


def test_build_generation_kwargs_precedence():
    # Direct field names apply tts_params-then-params (params wins, matching
    # the MOSS Delay semantics); both override the explicit generic aliases.
    kwargs = build_generation_kwargs(
        {"temperature": 0.5, "audio_temperature": 1.2},
        tts_params={
            "explicit_generation_params": ["temperature"],
            "audio_temperature": 1.9,
        },
    )
    assert kwargs["text_temperature"] == 0.5
    assert kwargs["audio_temperature"] == 1.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_decode_frame_graphed_matches_branchless_eager():
    """The captured frame graph must reproduce the branchless eager decode."""
    from sglang_omni.models.moss_tts_local.local_transformer import (
        sample_seeded_branchless,
    )

    torch.manual_seed(11)
    device = torch.device("cuda")
    module = MossTTSLocalTransformer(
        hidden_size=64,
        num_heads=4,
        inner_size=96,
        num_layers=1,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    ).to(device=device, dtype=torch.bfloat16)
    tables = [
        torch.randn(64, 64, device=device, dtype=torch.bfloat16) for _ in range(N_VQ)
    ]

    def frame(hidden, seeds, base):
        current = module.step(hidden, 0)
        codes = []
        for channel in range(N_VQ):
            logits = (current.float() @ tables[channel].float().T)[:, :32]
            code = sample_seeded_branchless(
                logits,
                temperature=torch.full((hidden.shape[0],), 1.7, device=device),
                top_p=torch.full((hidden.shape[0],), 0.8, device=device),
                top_k=torch.full(
                    (hidden.shape[0],), 25, device=device, dtype=torch.long
                ),
                seeds=seeds,
                positions=base + channel + 1,
            )
            codes.append(code)
            if channel + 1 < N_VQ:
                embed = torch.nn.functional.embedding(code, tables[channel][:32])
                current = module.step(embed.to(torch.bfloat16), channel + 1)
        return torch.stack(codes, dim=-1)

    batch = 4
    static_hidden = torch.zeros(batch, 64, device=device, dtype=torch.bfloat16)
    static_seeds = torch.zeros(batch, device=device, dtype=torch.long)
    static_base = torch.zeros(batch, device=device, dtype=torch.long)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            frame(static_hidden, static_seeds, static_base)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed_codes = frame(static_hidden, static_seeds, static_base)

    hidden = torch.randn(batch, 64, device=device, dtype=torch.bfloat16)
    seeds = torch.arange(batch, device=device, dtype=torch.long) * 999
    base = torch.full((batch,), 13, device=device, dtype=torch.long)

    static_hidden.copy_(hidden)
    static_seeds.copy_(seeds)
    static_base.copy_(base)
    graph.replay()
    from_graph = graphed_codes.clone()

    eager = frame(hidden, seeds, base)
    torch.testing.assert_close(from_graph, eager)


def test_batched_reference_encoder_coalesces_and_isolates_errors():
    import threading

    from sglang_omni.models.moss_tts_local.stages import _BatchedReferenceEncoder

    calls = []

    class _FakeAudioTokenizer:
        def load_paths(self, paths):
            return [(torch.full((1, len(path)), len(path)), 48000) for path in paths]

        def encode_waveforms(self, waveforms, *, num_quantizers):
            assert num_quantizers == N_VQ
            calls.append([int(wav.shape[-1]) for wav, _ in waveforms])
            out = []
            for wav, _ in waveforms:
                length = int(wav.shape[-1])
                if length == 4:
                    raise RuntimeError("cannot encode bad reference")
                out.append(torch.full((4, N_VQ), length, dtype=torch.long))
            return out

        def encode_paths(self, paths, *, num_quantizers):
            return self.encode_waveforms(
                self.load_paths(paths),
                num_quantizers=num_quantizers,
            )

    encoder = _BatchedReferenceEncoder(
        _FakeAudioTokenizer(),
        n_vq=N_VQ,
        max_batch_size=4,
        max_batch_wait_ms=20,
    )
    results = {}

    def run(path):
        try:
            results[path] = encoder.encode(path)
        except Exception as exc:
            results[path] = exc

    threads = [threading.Thread(target=run, args=(p,)) for p in ("aa", "bbb", "bad1")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert isinstance(results["bad1"], Exception)
    assert results["aa"].shape == (4, N_VQ) and int(results["aa"][0, 0]) == 2
    assert results["bbb"].shape == (4, N_VQ) and int(results["bbb"][0, 0]) == 3
    # The failing batch retried per item; good items still succeeded.
    assert any(len(c) > 1 for c in calls) or len(calls) >= 3


def test_batched_reference_encoder_mixes_path_and_waveform_jobs():
    import threading

    from sglang_omni.models.moss_tts_local.stages import _BatchedReferenceEncoder

    calls = []

    class _FakeAudioTokenizer:
        def load_paths(self, paths):
            return [(torch.full((1, len(path)), len(path)), 48000) for path in paths]

        def encode_waveforms(self, waveforms, *, num_quantizers):
            assert num_quantizers == N_VQ
            calls.append([int(wav.shape[-1]) for wav, _ in waveforms])
            return [
                torch.full((int(wav.shape[-1]), N_VQ), int(wav.shape[-1]))
                for wav, _ in waveforms
            ]

    encoder = _BatchedReferenceEncoder(
        _FakeAudioTokenizer(),
        n_vq=N_VQ,
        max_batch_size=2,
        max_batch_wait_ms=100,
    )
    results = {}
    errors = []
    barrier = threading.Barrier(2)

    def encode_path():
        try:
            barrier.wait(timeout=5)
            results["path"] = encoder.encode("aa")
        except Exception as exc:
            errors.append(exc)

    def encode_wav():
        try:
            barrier.wait(timeout=5)
            results["wav"] = encoder.encode_wav(torch.full((1, 5), 5), 48000)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=encode_path),
        threading.Thread(target=encode_wav),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert int(results["path"][0, 0]) == 2
    assert int(results["wav"][0, 0]) == 5
    assert calls[0] == [2, 5]


# CachedReferenceEncoder


def test_cached_reference_encoder_on_off_hit_bit_identical(tmp_path):
    """T5: OFF / ON-miss / ON-hit produce bit-identical prompt_rows and input_ids_list.

    Uses int32 boundary value 1023 to verify lossless round-trip through storage.
    ON-hit encode counter must stay at 1 (no second encode issued).
    """
    from sglang_omni.models.moss_tts_local.request_builders import (
        _prepare_moss_tts_local_request,
    )
    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    ref_file = tmp_path / "ref.wav"
    ref_file.write_bytes(b"fake wav bytes for T5")
    encode_count = 0

    class _FakeBatched:
        def encode(self, path: str) -> torch.Tensor:
            nonlocal encode_count
            encode_count += 1
            # Boundary value 1023 exercises int32 round-trip; varies by path length.
            codes = torch.full((10, N_VQ), 1023, dtype=torch.long)
            codes[0, 0] = len(path) % 1024
            return codes

    class _RefAwareProcessor:
        """Folds reference tensor sum into rows so bit-identity is non-trivial."""

        @staticmethod
        def build_user_message(**kwargs):
            return dict(kwargs, role="user")

        def __call__(self, conversations, mode):
            assert mode == "generation"
            msg = conversations[0][0]
            ref = msg.get("reference") or []
            ref_val = (
                int(ref[0].sum().item()) % 1024
                if ref and isinstance(ref[0], torch.Tensor)
                else 0
            )
            seq = 8
            rows = torch.full((1, seq, N_VQ + 1), ref_val, dtype=torch.long)
            rows[0, :, 0] = torch.arange(seq)
            rows[0, -1, 0] = 151669
            return {"input_ids": rows}

    def _ref_payload(rid: str) -> StagePayload:
        return StagePayload(
            request_id=rid,
            request=OmniRequest(
                inputs={"text": "hello", "references": [{"audio": str(ref_file)}]},
                params={},
                metadata={},
            ),
            data={},
        )

    processor = _RefAwareProcessor()
    fake_batched = _FakeBatched()

    # OFF: raw encoder, no cache wrapper
    prepared_off = _prepare_moss_tts_local_request(
        _ref_payload("t5-off"), processor=processor, reference_encoder=fake_batched
    )
    assert encode_count == 1

    # ON-miss: first call to CachedReferenceEncoder (cache empty)
    cached_enc = CachedReferenceEncoder(fake_batched, max_items=256, max_bytes=64 << 20)
    encode_count = 0
    prepared_miss = _prepare_moss_tts_local_request(
        _ref_payload("t5-miss"), processor=processor, reference_encoder=cached_enc
    )
    assert encode_count == 1, "ON-miss must call underlying encode exactly once"

    # ON-hit: second call, same file — cache must serve without re-encoding
    prepared_hit = _prepare_moss_tts_local_request(
        _ref_payload("t5-hit"), processor=processor, reference_encoder=cached_enc
    )
    assert encode_count == 1, "ON-hit must NOT call underlying encode again"

    # Hard gate: all three paths produce identical prompt_rows and input_ids_list
    assert torch.equal(prepared_off.prompt_rows, prepared_miss.prompt_rows)
    assert torch.equal(prepared_off.prompt_rows, prepared_hit.prompt_rows)
    assert prepared_off.input_ids_list == prepared_miss.input_ids_list
    assert prepared_off.input_ids_list == prepared_hit.input_ids_list

    stats = cached_enc.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["merged"] == 0


def test_cached_reference_encoder_lru_eviction(tmp_path):
    """T3: LRU eviction by item count and by byte budget."""
    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    encode_log = []

    class _FakeBatched:
        def encode(self, path: str) -> torch.Tensor:
            encode_log.append(path)
            return torch.full((2, N_VQ), ord(path[-1]), dtype=torch.long)

    # --- item-count eviction: max_items=2, insert 3 ---
    enc = CachedReferenceEncoder(_FakeBatched(), max_items=2, max_bytes=64 << 20)
    files = [tmp_path / f"{c}.wav" for c in "abc"]
    for i, f in enumerate(files):
        f.write_bytes(bytes([i + 1]) * 16)  # distinct content → distinct cache keys
    for f in files:
        enc.encode(str(f))
    encode_log.clear()
    # 'a' was evicted (LRU); 'b' and 'c' are still cached
    enc.encode(str(files[1]))  # b — should hit
    enc.encode(str(files[2]))  # c — should hit
    assert encode_log == [], "b and c should be cached (not re-encoded)"
    enc.encode(str(files[0]))  # a — must re-encode
    assert len(encode_log) == 1 and str(files[0]) in encode_log[0]

    # --- byte-budget eviction: each entry is 2*12*4=96 bytes as int32 ---
    entry_bytes = 2 * N_VQ * 4  # int32
    enc2 = CachedReferenceEncoder(_FakeBatched(), max_items=1000, max_bytes=entry_bytes)
    f1, f2 = tmp_path / "d.wav", tmp_path / "e.wav"
    f1.write_bytes(b"y" * 16)
    f2.write_bytes(b"z" * 16)
    encode_log.clear()
    enc2.encode(str(f1))
    enc2.encode(str(f2))  # f1 evicted by byte budget
    encode_log.clear()
    enc2.encode(str(f2))  # e — should hit
    assert encode_log == [], "f2 should be cached"
    enc2.encode(str(f1))  # d — must re-encode
    assert len(encode_log) == 1

    # --- oversized entry is gracefully rejected, does not crash ---
    enc3 = CachedReferenceEncoder(_FakeBatched(), max_items=10, max_bytes=1)
    f3 = tmp_path / "big.wav"
    f3.write_bytes(b"big" * 16)
    result = enc3.encode(str(f3))
    assert result is not None  # still returns a value


def test_cached_reference_encoder_rejects_nonpositive_capacity():
    """Negative/zero capacities fail fast at construction (P3, review)."""
    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    class _FakeBatched:
        def encode(self, path: str) -> torch.Tensor:
            return torch.zeros((2, N_VQ), dtype=torch.long)

    with pytest.raises(ValueError, match="max_items"):
        CachedReferenceEncoder(_FakeBatched(), max_items=-1)
    with pytest.raises(ValueError, match="max_items"):
        CachedReferenceEncoder(_FakeBatched(), max_items=0)
    with pytest.raises(ValueError, match="max_bytes"):
        CachedReferenceEncoder(_FakeBatched(), max_bytes=0)


def test_cached_reference_encoder_true_concurrency_dedup(tmp_path):
    """T4: concurrent same-key requests — exactly 1 encode, all results torch.equal,
    data_ptr pairwise distinct (each caller gets an independent clone).
    """
    import threading as _threading

    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"concurrent test")

    gate = _threading.Event()
    call_count = 0

    class _GatedBatched:
        def encode(self, path: str) -> torch.Tensor:
            nonlocal call_count
            gate.wait()  # stall until all threads have registered
            call_count += 1
            return torch.full((5, N_VQ), 42, dtype=torch.long)

    N = 8
    enc = CachedReferenceEncoder(_GatedBatched(), max_items=256, max_bytes=64 << 20)
    results = [None] * N
    errors = []

    def worker(idx):
        try:
            results[idx] = enc.encode(str(ref))
        except Exception as e:
            errors.append(e)

    threads = [_threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    # Give threads time to block inside encode() before releasing gate
    import time

    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"unexpected errors: {errors}"
    assert call_count == 1, f"expected 1 encode call, got {call_count}"
    assert all(r is not None for r in results)
    ref_tensor = results[0]
    for r in results[1:]:
        assert torch.equal(ref_tensor, r), "results not bit-identical"
    ptrs = [r.data_ptr() for r in results]
    assert len(set(ptrs)) == len(ptrs), "data_ptr not pairwise distinct (shared tensor)"

    stats = enc.stats()
    assert stats["misses"] == 1
    assert stats["merged"] == N - 1
    assert stats["hits"] == 0


def test_cached_reference_encoder_failure_does_not_poison(tmp_path):
    """T6: leader failure fans out independent exception instances; cache stays clean;
    third request becomes new leader and succeeds.
    """
    import threading as _threading

    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    ref = tmp_path / "flaky.wav"
    ref.write_bytes(b"flaky")

    gate = _threading.Event()
    call_count = 0

    class _FlakyBatched:
        def encode(self, path: str) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            gate.wait()
            if call_count == 1:
                raise RuntimeError("transient encode failure")
            return torch.full((3, N_VQ), 7, dtype=torch.long)

    enc = CachedReferenceEncoder(_FlakyBatched(), max_items=256, max_bytes=64 << 20)
    exc_results = [None, None]

    def fail_worker(idx):
        try:
            enc.encode(str(ref))
        except Exception as e:
            exc_results[idx] = e

    t0 = _threading.Thread(target=fail_worker, args=(0,))
    t1 = _threading.Thread(target=fail_worker, args=(1,))
    t0.start()
    t1.start()
    import time

    time.sleep(0.05)
    gate.set()
    t0.join(timeout=10)
    t1.join(timeout=10)

    # Both threads got an exception
    assert exc_results[0] is not None
    assert exc_results[1] is not None
    # Each gets an independent instance (not the same object)
    assert exc_results[0] is not exc_results[1]
    # Cache not poisoned
    assert enc.stats()["entries"] == 0
    # inflight cleared
    assert len(enc._inflight) == 0

    # Third request succeeds (new leader)
    gate.clear()
    gate.set()
    result = enc.encode(str(ref))
    assert result is not None
    assert torch.equal(result, torch.full((3, N_VQ), 7, dtype=torch.long))


def test_cached_reference_encoder_return_value_isolation(tmp_path):
    """T7: mutating the returned tensor does not corrupt the cached copy."""
    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    ref = tmp_path / "iso.wav"
    ref.write_bytes(b"isolation test")

    class _FakeBatched:
        def encode(self, path: str) -> torch.Tensor:
            return torch.full((4, N_VQ), 99, dtype=torch.long)

    enc = CachedReferenceEncoder(_FakeBatched(), max_items=256, max_bytes=64 << 20)
    enc.encode(str(ref))  # miss — populates cache

    hit1 = enc.encode(str(ref))  # first hit
    hit1.fill_(-1)  # mutate in place

    hit2 = enc.encode(str(ref))  # second hit — must still be 99
    assert torch.all(hit2 == 99), "cache was corrupted by mutation of first hit result"


def test_cached_reference_encoder_duration_gate(tmp_path, monkeypatch):
    """T8: references over 100 s are rejected before touching the cache."""
    torchaudio = pytest.importorskip("torchaudio")

    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    ref = tmp_path / "long.wav"
    ref.write_bytes(b"fake long audio")
    encode_count = 0

    class _FakeBatched:
        def encode(self, path: str) -> torch.Tensor:
            nonlocal encode_count
            encode_count += 1
            return torch.zeros((5, N_VQ), dtype=torch.long)

    # Patch torchaudio.info to report a 200-second file
    class _FakeInfo:
        num_frames = 200 * 48000
        sample_rate = 48000

    monkeypatch.setattr(torchaudio, "info", lambda path: _FakeInfo(), raising=False)

    enc = CachedReferenceEncoder(_FakeBatched(), max_items=256, max_bytes=64 << 20)

    # _BatchedReferenceEncoder.encode checks duration before enqueuing; CachedReferenceEncoder
    # calls through to the underlying encoder so the duration check still fires.
    with pytest.raises(ValueError, match="100"):
        enc.encode(str(ref))

    assert encode_count == 0, "oversized reference must not reach the codec"
    assert enc.stats()["entries"] == 0
    assert len(enc._inflight) == 0


# Data-URI reference path


def _make_wav_data_uri(
    n_samples: int = 100, sample_rate: int = 16000
) -> tuple[str, bytes]:
    """Minimal 16-bit mono PCM WAV wrapped as a data URI."""
    import base64

    samples = b"\x00\x00" * n_samples
    data_size = len(samples)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    raw = header + samples
    return f"data:audio/wav;base64,{base64.b64encode(raw).decode()}", raw


def test_cached_reference_encoder_data_uri_hit_miss(tmp_path):
    """bytes: keyspace: same data-URI encoded twice -> one codec encode."""
    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    pytest.importorskip("soundfile")
    data_uri, _ = _make_wav_data_uri()
    wav_call_count = 0

    class _FakeBatched:
        def encode_wav(self, wav, sample_rate):
            nonlocal wav_call_count
            wav_call_count += 1
            return torch.full((5, N_VQ), 42, dtype=torch.long)

    enc = CachedReferenceEncoder(_FakeBatched(), max_items=256, max_bytes=64 << 20)
    enc.encode_data_uri(data_uri)
    assert wav_call_count == 1, "first call must encode"

    result2 = enc.encode_data_uri(data_uri)
    assert wav_call_count == 1, "second call must hit cache"
    assert torch.equal(result2, torch.full((5, N_VQ), 42, dtype=torch.long))

    stats = enc.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_uncached_data_uri_uses_reference_encoder():
    from sglang_omni.models.moss_tts_local.request_builders import (
        _build_processor_message,
    )
    from sglang_omni.models.moss_tts_local.stages import _BatchedReferenceEncoder

    pytest.importorskip("soundfile")
    data_uri, _ = _make_wav_data_uri()
    model = _FakeAudioTokenizerModel()
    tokenizer = MossTTSLocalAudioTokenizer(model, device="cpu")
    reference_encoder = _BatchedReferenceEncoder(
        tokenizer,
        n_vq=N_VQ,
        max_batch_size=4,
        max_batch_wait_ms=20,
    )
    processor = _FakeProcessor()
    state = MossTTSLocalState(text="hello", ref_audio=data_uri)

    message = _build_processor_message(processor, state, reference_encoder)

    assert len(model.calls) == 1
    assert model.calls[0][1] == N_VQ
    assert len(message["reference"]) == 1
    assert message["reference"][0].shape[1] == N_VQ


def test_cached_reference_encoder_file_bytes_keyspaces_do_not_collide(tmp_path):
    """file: and bytes: keys are independent; same-content file ≠ data-URI in cache."""
    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    pytest.importorskip("soundfile")
    data_uri, raw = _make_wav_data_uri()

    # Write the same raw bytes as a file
    ref_file = tmp_path / "ref.wav"
    ref_file.write_bytes(raw)

    encode_count = 0

    class _FakeBatched:
        def encode(self, path: str) -> torch.Tensor:
            nonlocal encode_count
            encode_count += 1
            return torch.full((5, N_VQ), 7, dtype=torch.long)

        def encode_wav(self, wav, sample_rate):
            nonlocal encode_count
            encode_count += 1
            return torch.full((5, N_VQ), 7, dtype=torch.long)

    enc = CachedReferenceEncoder(_FakeBatched(), max_items=256, max_bytes=64 << 20)
    enc.encode(str(ref_file))  # populates file: key
    enc.encode_data_uri(data_uri)  # must NOT hit file: entry

    assert (
        enc.stats()["misses"] == 2
    ), "file: and bytes: are independent keyspaces; data-URI must be a fresh miss"


def test_cached_reference_encoder_data_uri_duration_gate():
    """data-URI references over 100 s are rejected."""
    import base64
    import io

    from sglang_omni.models.moss_tts_local.stages import CachedReferenceEncoder

    pytest.importorskip("soundfile")
    import soundfile as sf

    # Build a 200-second silent WAV (at 8 kHz to keep size small)
    sample_rate = 8000
    n_samples = 200 * sample_rate
    buf = io.BytesIO()
    import numpy as np

    sf.write(buf, np.zeros(n_samples, dtype=np.float32), sample_rate, format="WAV")
    raw = buf.getvalue()
    uri = f"data:audio/wav;base64,{base64.b64encode(raw).decode()}"

    class _FakeBatched:
        def encode_wav(self, wav, sample_rate):
            return torch.zeros((5, N_VQ), dtype=torch.long)

    enc = CachedReferenceEncoder(_FakeBatched(), max_items=256, max_bytes=64 << 20)
    with pytest.raises(ValueError, match="100"):
        enc.encode_data_uri(uri)

    assert enc.stats()["entries"] == 0
    assert len(enc._inflight) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs CUDA: post1 multinomial_with_seed is a Triton kernel (no CPU backend)",
)
def test_branchless_sampler_matches_eager_sampler():
    """The CUDA-graphable sampler must reproduce the eager path exactly."""
    pytest.importorskip("sglang")
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts_local.local_transformer import (
        sample_seeded_branchless,
    )

    torch.manual_seed(7)
    rows, vocab = 6, 64
    logits = torch.randn(rows, vocab, dtype=torch.float32, device="cuda") * 3
    temperature = torch.tensor([1.7, 1.0, 0.5, 1.7, 0.0, 1.7], device="cuda")
    top_p = torch.tensor([0.8, 1.0, 0.9, 0.8, 0.8, 0.8], device="cuda")
    top_k = torch.tensor([25, 50, 8, 64, 25, 1], dtype=torch.long, device="cuda")
    seeds = torch.arange(rows, dtype=torch.long, device="cuda") * 1234567
    positions = torch.arange(rows, dtype=torch.long, device="cuda") * 13

    eager = MossTTSModelRunner._sample_tokens(
        logits.clone(),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
    )
    branchless = sample_seeded_branchless(
        logits.clone(),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
    )
    torch.testing.assert_close(eager, branchless)


# Stereo audio payload + encoding


def test_audio_waveform_payload_keeps_stereo_shape():
    wav = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    payload = audio_waveform_payload(wav, keep_channels=True)
    assert payload["audio_waveform_shape"] == [2, 4]
    restored = np.frombuffer(payload["audio_waveform"], dtype=np.float32).reshape(2, 4)
    np.testing.assert_allclose(restored, wav.numpy())
    # Default behavior still flattens.
    flat = audio_waveform_payload(wav)
    assert flat["audio_waveform_shape"] == [8]


def test_encode_wav_stereo_header_and_interleave():
    stereo = np.stack(
        [np.full(4, 0.5, dtype=np.float32), np.full(4, -0.5, dtype=np.float32)]
    )
    blob = encode_wav(stereo, 48000)
    assert blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"
    num_channels = struct.unpack("<H", blob[22:24])[0]
    sample_rate = struct.unpack("<I", blob[24:28])[0]
    assert num_channels == 2
    assert sample_rate == 48000
    pcm = np.frombuffer(blob[44:], dtype=np.int16).reshape(-1, 2)
    assert (pcm[:, 0] > 0).all() and (pcm[:, 1] < 0).all()


def test_encode_audio_stereo_wav_and_mono_fallback():
    stereo = np.stack(
        [np.ones(64, dtype=np.float32) * 0.1, np.ones(64, dtype=np.float32) * -0.1]
    )
    blob, mime = encode_audio(stereo, response_format="wav", sample_rate=48000)
    assert mime == "audio/wav"
    assert struct.unpack("<H", blob[22:24])[0] == 2
    # Mono input keeps the legacy single-channel header.
    mono_blob, _ = encode_audio(
        np.ones(64, dtype=np.float32) * 0.1, response_format="wav", sample_rate=48000
    )
    assert struct.unpack("<H", mono_blob[22:24])[0] == 1


def test_post_process_outputs_skips_chunked_rows():
    """Chunked-prefill rows must not be appended to output_rows."""
    pytest.importorskip("sglang")
    import types

    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
    from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeJournal

    batch_size = 2

    # Minimal model stub providing only what post_process_outputs needs.
    model_stub = types.SimpleNamespace(
        config=types.SimpleNamespace(audio_end_token_id=151670)
    )
    runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
    runner.model = model_stub

    # Two rows: row 0 is chunked (mid-prefill), row 1 is normal.
    rows = torch.arange(batch_size * (N_VQ + 1), dtype=torch.long).reshape(
        batch_size, N_VQ + 1
    )

    # Build minimal sched_req stubs.
    def _req(is_chunked):
        return types.SimpleNamespace(is_chunked=is_chunked)

    def _sched_req(rid, is_chunked):
        data = types.SimpleNamespace(req=_req(is_chunked), output_rows=[])
        return types.SimpleNamespace(request_id=rid, data=data)

    req_a = _sched_req("r0", is_chunked=1)  # mid-prefill chunk, must be skipped
    req_b = _sched_req("r1", is_chunked=0)  # normal decode row

    sched_output = types.SimpleNamespace(requests=[req_a, req_b])
    outputs = {
        "r0": types.SimpleNamespace(data=1000),  # non-end token
        "r1": types.SimpleNamespace(data=1001),  # non-end token
    }

    # Output collection goes solely through the per-step journal. Chunked rows
    # are not journaled because no frame should be emitted or fed back.
    result = types.SimpleNamespace(
        moss_journal=MossTTSLocalDecodeJournal(
            rids=["r1"], pool_rows=[1], rows=rows[1:]
        )
    )
    runner.post_process_outputs(result, sched_output, outputs)

    assert req_a.data.output_rows == [], "chunked row must not be appended"
    assert len(req_b.data.output_rows) == 1, "normal row must be appended"


def test_finalize_skip_rids_selects_chunked_rows():
    pytest.importorskip("sglang")
    import types

    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner

    runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)

    def _sched_req(rid, is_chunked):
        data = types.SimpleNamespace(req=types.SimpleNamespace(is_chunked=is_chunked))
        return types.SimpleNamespace(request_id=rid, data=data)

    sched_output = types.SimpleNamespace(
        requests=[
            _sched_req("c0", is_chunked=1),
            _sched_req("c1", is_chunked=2),
            _sched_req("final", is_chunked=0),
        ]
    )
    assert runner.finalize_skip_rids(sched_output) == {"c0", "c1"}


def test_chunked_prefill_generation_steps_matches_single_shot():
    # A K-chunk prefill (mid chunks is_chunked>0, final is_chunked==0) must leave
    # generation_steps identical to a single-shot prefill, so the first decode
    # frame samples at the same position (position = generation_steps *
    # num_channels + channel) — bit-identical to the no-chunk path.
    pytest.importorskip("sglang")
    import types

    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner

    class _OutputProcessor:
        def process(self, batch_result, scheduler_output):
            del batch_result
            return {
                req.request_id: types.SimpleNamespace(extra=None)
                for req in scheduler_output.requests
            }

    def _make_runner():
        runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
        runner.model = types.SimpleNamespace(
            config=types.SimpleNamespace(audio_end_token_id=151670)
        )
        runner.output_processor = _OutputProcessor()
        return runner

    def _finalize_once(runner, sched_req):
        runner._finalize(
            types.SimpleNamespace(
                next_token_ids=torch.tensor([0]),
                logits_output=None,
                can_run_cuda_graph=False,
                moss_journal=None,
            ),
            types.SimpleNamespace(),
            types.SimpleNamespace(is_prefill_only=False, output_ids=None),
            types.SimpleNamespace(),
            types.SimpleNamespace(requests=[sched_req]),
        )

    # Single-shot prefill: the only chunk is final → exactly one advance.
    runner = _make_runner()
    data = types.SimpleNamespace(
        req=types.SimpleNamespace(is_chunked=0),
        generation_steps=0,
        extra_model_outputs={},
    )
    _finalize_once(runner, types.SimpleNamespace(request_id="r", data=data))
    assert data.generation_steps == 1

    # 3-chunk prefill on the same request: mid chunks (is_chunked>0) suppressed,
    # final chunk advances → same end state as single-shot.
    runner = _make_runner()
    data = types.SimpleNamespace(
        req=types.SimpleNamespace(is_chunked=2),
        generation_steps=0,
        extra_model_outputs={},
    )
    sched_req = types.SimpleNamespace(request_id="r", data=data)
    for is_chunked in (2, 1, 0):
        data.req.is_chunked = is_chunked
        _finalize_once(runner, sched_req)
    assert data.generation_steps == 1


def test_lookahead_eligible_routes_eager_batches_to_sync():
    """Lookahead is eligible only when bs <= frame_graph_max_bs AND every
    request has audio_repetition_penalty == 1.0; a rep-penalty request or a
    batch over the graph cap forces the eager path and must route to sync.
    """
    pytest.importorskip("sglang")
    import types

    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner

    runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
    runner.model = types.SimpleNamespace(frame_graph_max_bs=16)

    def _batch(penalties):
        return types.SimpleNamespace(
            reqs=[
                types.SimpleNamespace(
                    _omni_data=types.SimpleNamespace(audio_repetition_penalty=p)
                )
                for p in penalties
            ]
        )

    assert runner.lookahead_eligible(_batch([1.0, 1.0])) is True
    assert runner.lookahead_eligible(_batch([1.0, 1.3])) is False  # rep-penalty eager
    assert runner.lookahead_eligible(_batch([1.0] * 17)) is False  # bs over graph cap


def test_async_launch_resolve_matches_sync_collect():
    """post_decode_launch + post_decode_resolve must yield the same published
    next_token_ids and the same output_rows append as synchronous _collect_frame.
    The launch hands resolve a device snapshot of the published ids so they
    survive the next step clobbering the aliased output_ids tensor in place; CPU
    stub: eager decode (no CUDA graph).
    """
    pytest.importorskip("sglang")
    import types

    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
    from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeStatePool

    hidden_size = 4

    def _make_runner():
        weight = torch.zeros(2, hidden_size, dtype=torch.bfloat16)
        model = types.SimpleNamespace(
            _decode_input_embedding=types.SimpleNamespace(weight=weight),
            _state_pool=None,
            config=types.SimpleNamespace(
                n_vq=12, audio_assistant_slot_token_id=1000, audio_end_token_id=1001
            ),
            frame_graph_max_bs=0,  # eager path
            device=torch.device("cpu"),
        )
        pool = MossTTSLocalDecodeStatePool(model)
        model._state_pool = pool
        model.acquire_row = pool.acquire_row
        model.decode_frame = lambda hidden, *, sample_text, sample_audio: (
            torch.zeros(1, dtype=torch.long),  # stop_choice=0 -> continue (slot)
            torch.arange(12, dtype=torch.long).reshape(1, 12),
        )
        model._prepare_multi_modal_inputs = lambda rows: torch.full(
            (1, hidden_size), 3, dtype=torch.bfloat16
        )
        runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
        runner.model = model
        return runner

    def _sched_req():
        data = types.SimpleNamespace(
            req=None,
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=1.0,
            audio_top_p=1.0,
            audio_top_k=50,
            sampling_seed=0,
            generation_steps=0,
            audio_repetition_penalty=1.0,
            output_rows=[],
        )
        return types.SimpleNamespace(request_id="rid", data=data)

    def _result():
        return types.SimpleNamespace(
            logits_output=types.SimpleNamespace(
                hidden_states=torch.zeros(1, hidden_size)
            )
        )

    # Synchronous collect.
    rs = _make_runner()
    req_s, res_s, sb_s = _sched_req(), _result(), types.SimpleNamespace()
    rs._collect_frame(res_s, None, sb_s, [req_s])

    # Async launch + resolve (separate runner/pool to avoid cross-overwrite).
    ra = _make_runner()
    req_a, res_a = _sched_req(), _result()
    host_buf = ra.post_decode_launch(res_a, None, [req_a])
    # Launch hands resolve a private device snapshot of the published ids.
    assert host_buf is not None
    assert torch.equal(host_buf, res_a.next_token_ids)
    # Simulate the next decode step overwriting the aliased published tensor in
    # place (the output_ids -> input_ids clobber): resolve must still recover the
    # real ids from the snapshot.
    res_a.next_token_ids.zero_()
    ra.post_decode_resolve(host_buf, res_a, None, None, [req_a])

    # Resolve restored the snapshot, so async and sync yield identical ids.
    assert torch.equal(res_s.next_token_ids, res_a.next_token_ids)
    assert torch.equal(sb_s.output_ids, res_s.next_token_ids)  # sync still publishes

    # output_rows append parity through the shared post_process_outputs tail.
    rs.post_process_outputs(
        res_s,
        types.SimpleNamespace(requests=[req_s]),
        {"rid": types.SimpleNamespace(data=int(res_s.next_token_ids[0]))},
    )
    ra.post_process_outputs(
        res_a,
        types.SimpleNamespace(requests=[req_a]),
        {"rid": types.SimpleNamespace(data=int(res_a.next_token_ids[0]))},
    )
    assert len(req_s.data.output_rows) == len(req_a.data.output_rows) == 1
    assert torch.equal(req_s.data.output_rows[0], req_a.data.output_rows[0])


def test_async_resolve_preserves_stop_id_through_output_ids_clobber():
    """bs=1 stop-boundary regression. A stop frame publishes end_id as
    next_token_ids; the base aliases it onto schedule_batch.output_ids, which the
    next decode step overwrites in place. Under lookahead that clobber races ahead
    of this step's resolve, so post_decode_launch must snapshot the ids and resolve
    must restore them — otherwise the eos finish never reaches process_batch_result
    and a bs=1 request never stops (the 4096-frame runaway)."""
    pytest.importorskip("sglang")
    import types

    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
    from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeStatePool

    hidden_size = 4
    end_id = 1001

    weight = torch.zeros(2, hidden_size, dtype=torch.bfloat16)
    model = types.SimpleNamespace(
        _decode_input_embedding=types.SimpleNamespace(weight=weight),
        _state_pool=None,
        config=types.SimpleNamespace(
            n_vq=12, audio_assistant_slot_token_id=1000, audio_end_token_id=end_id
        ),
        frame_graph_max_bs=0,  # eager path
        device=torch.device("cpu"),
    )
    pool = MossTTSLocalDecodeStatePool(model)
    model._state_pool = pool
    model.acquire_row = pool.acquire_row
    model.decode_frame = lambda hidden, *, sample_text, sample_audio: (
        torch.ones(1, dtype=torch.long),  # stop_choice=1 -> stop (end_id)
        torch.arange(12, dtype=torch.long).reshape(1, 12),
    )
    model._prepare_multi_modal_inputs = lambda rows: torch.full(
        (1, hidden_size), 3, dtype=torch.bfloat16
    )
    runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
    runner.model = model

    data = types.SimpleNamespace(
        req=None,
        text_temperature=1.0,
        text_top_p=1.0,
        text_top_k=50,
        audio_temperature=1.0,
        audio_top_p=1.0,
        audio_top_k=50,
        sampling_seed=0,
        generation_steps=0,
        audio_repetition_penalty=1.0,
        output_rows=[],
    )
    req = types.SimpleNamespace(request_id="rid", data=data)
    res = types.SimpleNamespace(
        logits_output=types.SimpleNamespace(hidden_states=torch.zeros(1, hidden_size))
    )

    host_buf = runner.post_decode_launch(res, None, [req])
    # The stop frame's published id is the raw end_id (eos detection keys on it).
    assert int(res.next_token_ids[0]) == end_id
    assert host_buf is not None
    # The next step clobbers the aliased published tensor in place.
    res.next_token_ids.zero_()
    assert int(res.next_token_ids[0]) != end_id
    # Resolve must restore the stop id so the eos finish still fires.
    runner.post_decode_resolve(host_buf, res, None, None, [req])
    assert int(res.next_token_ids[0]) == end_id


def test_chunked_rows_do_not_advance_sampling_steps():
    """A non-final chunked-prefill row's garbage frame must not advance the
    launch-side sampling counter, so the final chunk samples at the same RNG
    position as a single-shot prefill (mirrors D1's generation_steps handling).
    """
    pytest.importorskip("sglang")
    import types

    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
    from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeStatePool

    hidden_size = 4

    def _make_runner():
        weight = torch.zeros(2, hidden_size, dtype=torch.bfloat16)
        model = types.SimpleNamespace(
            _decode_input_embedding=types.SimpleNamespace(weight=weight),
            _state_pool=None,
            config=types.SimpleNamespace(
                n_vq=12, audio_assistant_slot_token_id=1000, audio_end_token_id=1001
            ),
            frame_graph_max_bs=0,
            device=torch.device("cpu"),
        )
        pool = MossTTSLocalDecodeStatePool(model)
        model._state_pool = pool
        model.acquire_row = pool.acquire_row
        model.decode_frame = lambda hidden, *, sample_text, sample_audio: (
            torch.zeros(1, dtype=torch.long),
            torch.arange(12, dtype=torch.long).reshape(1, 12),
        )
        model._prepare_multi_modal_inputs = lambda rows: torch.full(
            (1, hidden_size), 3, dtype=torch.bfloat16
        )
        runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
        runner.model = model
        return runner

    def _result():
        return types.SimpleNamespace(
            logits_output=types.SimpleNamespace(
                hidden_states=torch.zeros(1, hidden_size)
            )
        )

    def _data(is_chunked):
        return types.SimpleNamespace(
            req=types.SimpleNamespace(is_chunked=is_chunked),
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=1.0,
            audio_top_p=1.0,
            audio_top_k=50,
            sampling_seed=0,
            generation_steps=0,
            sampling_steps=None,
            audio_repetition_penalty=1.0,
            output_rows=[],
        )

    def _pool_sampling_steps(runner, rid):
        pool = runner.model._state_pool
        row = pool.row_for(rid)
        assert row is not None
        return int(pool.sampling_steps[row])

    # Single-shot prefill: the only chunk is final, advances sampling_steps to 1.
    r = _make_runner()
    single = types.SimpleNamespace(request_id="r", data=_data(is_chunked=0))
    r._run_frame_decode(_result(), types.SimpleNamespace(), [single])
    assert _pool_sampling_steps(r, "r") == 1

    # Three-chunk prefill on the same request: the mid chunks do not advance, the
    # final chunk does, so the end state matches the single-shot path.
    r = _make_runner()
    data = _data(is_chunked=2)
    sched = types.SimpleNamespace(request_id="r", data=data)
    for is_chunked, expected_steps in ((2, 0), (1, 0), (0, 1)):
        data.req.is_chunked = is_chunked
        r._run_frame_decode(_result(), types.SimpleNamespace(), [sched])
        assert _pool_sampling_steps(r, "r") == expected_steps


def test_async_decode_cli_accepts_moss_local():
    """The decode-mode CLI gate accepts the MOSS-TTS-Local engine
    factory (no BadParameter) and writes the flags onto its tts_engine stage.
    Default stays OFF (config sets no key); only an explicit --decode-mode async
    turns it on, pending the Phase-3 flag-flip PR.
    """
    pytest.importorskip("sglang")

    from sglang_omni.cli.serve import apply_decode_mode_cli_overrides
    from sglang_omni.config import resolve_stage_factory_args
    from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig

    config = MossTTSLocalPipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        config, decode_mode="async", async_lookahead_min_batch_size=4
    )
    stage = next(s for s in config.stages if s.name == "tts_engine")
    args = resolve_stage_factory_args(stage, config)
    assert args["enable_async_decode"] is True
    assert args["async_decode_min_batch_size"] == 4
    assert args["total_gpu_memory_fraction"] == pytest.approx(0.90)
    assert args["codec_mem_reserve"] == pytest.approx(0.15)
