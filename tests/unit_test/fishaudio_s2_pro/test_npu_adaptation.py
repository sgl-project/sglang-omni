# SPDX-License-Identifier: Apache-2.0
"""NPU platform adaptation tests for FishAudio S2-Pro."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
    audio_decoder as fish_audio_decoder,
)


def test_npu_kvcache_attention_uses_fused_infer_attention(
    monkeypatch,
) -> None:
    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 4, 8)
    v = torch.randn(2, 1, 4, 8)
    k_cache = torch.zeros(2, 11, 4, 8)
    v_cache = torch.zeros(2, 11, 4, 8)
    calls = []

    def fake_fused_attention(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return torch.full_like(query, 7), torch.empty(0)

    monkeypatch.setattr(
        fish_audio_decoder.torch.ops,
        "npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=fake_fused_attention,
        ),
        raising=False,
    )

    out = fish_audio_decoder._npu_kvcache_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_position=3,
    )

    assert torch.equal(out, torch.full_like(q, 7))
    assert torch.equal(k_cache[:, 3:4], k)
    assert torch.equal(v_cache[:, 3:4], v)
    assert len(calls) == 1
    query, key, value, kwargs = calls[0]
    assert query is q
    assert key.shape == value.shape == (2, 4, 4, 8)
    assert kwargs == {
        "num_heads": 4,
        "num_key_value_heads": 4,
        "input_layout": "BSND",
        "scale": 1.0 / math.sqrt(8),
    }


def test_npu_kvcache_attention_requires_fused_infer_attention(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fish_audio_decoder.torch.ops,
        "npu",
        SimpleNamespace(),
        raising=False,
    )
    q = torch.randn(1, 1, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)

    with pytest.raises(RuntimeError, match="npu_fused_infer_attention_score"):
        fish_audio_decoder._npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=q,
            v=q,
            cache_position=0,
        )


def test_npu_kvcache_attention_requires_single_token_query(monkeypatch) -> None:
    monkeypatch.setattr(
        fish_audio_decoder.torch.ops,
        "npu",
        SimpleNamespace(npu_fused_infer_attention_score=lambda *args, **kwargs: None),
        raising=False,
    )
    q = torch.randn(1, 2, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)

    with pytest.raises(ValueError, match="single-token query"):
        fish_audio_decoder._npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=q,
            v=q,
            cache_position=0,
        )


def test_fast_ar_attention_rejects_unsupported_device_clearly() -> None:
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)

    with pytest.raises(RuntimeError, match="supports CUDA and NPU"):
        fish_audio_decoder.flash_attn_kvcache_op(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            cache_position=0,
        )


def test_npu_kvcache_attention_requires_kv_and_cache_position() -> None:
    q = torch.randn(1, 1, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)
    k = torch.randn(1, 1, 4, 8)

    with pytest.raises(ValueError, match="requires k and v"):
        fish_audio_decoder._npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=None,
            v=None,
            cache_position=0,
        )
    with pytest.raises(ValueError, match="requires cache_position"):
        fish_audio_decoder._npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=k,
            cache_position=-1,
        )


def test_fish_engine_builder_npu_defaults(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder as fish_engine

    fake_npu = SimpleNamespace(is_npu=lambda: True, device_type="npu")
    monkeypatch.setattr(fish_engine, "current_platform", fake_npu)

    builder = fish_engine.FishS2ProEngineBuilder(max_new_tokens=256, ras_window=16)
    builder.gpu_id = 0

    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is False
    assert defaults["cuda_graph_backend_decode"] == "full"
    assert defaults["max_running_requests"] == 16
    assert defaults["mem_fraction_static"] == 0.75
    assert defaults["enable_torch_compile"] is False
    assert defaults["dtype"] == "bfloat16"

    # NPU bounds decode graph buckets to keep eager-prefill headroom.
    overrides: dict = {}
    builder.adjust_overrides(overrides)
    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8, 16]
    assert overrides["cuda_graph_max_bs"] == 16

    assert fish_engine._resolve_fast_ar_attention_backend(gpu_id=0) == "ascend"


def test_fish_engine_builder_compiles_when_npu_opt_in_is_enabled(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder as fish_engine

    monkeypatch.setattr(
        fish_engine,
        "current_platform",
        SimpleNamespace(is_npu=lambda: True, device_type="npu"),
    )
    compiled: list[tuple[object, int]] = []
    monkeypatch.setattr(
        fish_engine.fish_stages,
        "_compile_s2pro_codebook_decoder",
        lambda model, *, max_batch_size: compiled.append((model, max_batch_size)),
    )

    def apply_override(server_args, _source, **fields):
        for key, value in fields.items():
            setattr(server_args, key, value)

    monkeypatch.setattr(fish_engine, "override_server_args", apply_override)

    builder = fish_engine.FishS2ProEngineBuilder(max_new_tokens=256, ras_window=16)
    model = object()
    server_args = SimpleNamespace(enable_torch_compile=True, torch_compile_max_bs=16)

    builder.compile_model(model, server_args)

    assert compiled == [(model, 16)]
    assert server_args.enable_torch_compile is False


def test_fish_engine_builder_cuda_defaults_unchanged(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder as fish_engine

    fake_cuda = SimpleNamespace(is_npu=lambda: False)
    monkeypatch.setattr(fish_engine, "current_platform", fake_cuda)
    monkeypatch.setattr(
        fish_engine,
        "get_visible_gpu_sm_version",
        lambda gpu_id: 90,
    )

    builder = fish_engine.FishS2ProEngineBuilder(max_new_tokens=256, ras_window=16)
    builder.gpu_id = 0

    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is False
    assert defaults["enable_torch_compile"] is True


def test_stage_devices_resolve_from_current_platform(monkeypatch) -> None:
    import inspect

    from sglang_omni.models.fishaudio_s2_pro import stages as fish_stages
    from sglang_omni.models.fishaudio_s2_pro import streaming_vocoder

    # The tts_engine stage resolves device=None via resolve_device_spec.
    default = (
        inspect.signature(fish_stages.create_sglang_tts_engine_executor)
        .parameters["device"]
        .default
    )
    assert default is None

    # The vocoder stage resolves device=None from gpu_id via resolve_device_spec.
    seen: list[str] = []
    monkeypatch.setattr(
        fish_stages,
        "resolve_device_spec",
        lambda device, index: f"npu:{index}" if index is not None else "npu",
    )
    monkeypatch.setattr(fish_stages, "_resolve_checkpoint", lambda model_path: "ckpt")
    monkeypatch.setattr(
        fish_stages,
        "_load_codec",
        lambda checkpoint_dir, device: seen.append(device) or object(),
    )
    monkeypatch.setattr(
        streaming_vocoder, "S2ProVocoderScheduler", lambda *args, **kwargs: None
    )

    fish_stages.create_vocoder_executor("model", device=None, gpu_id=0)
    assert seen[-1] == "npu:0"

    fish_stages.create_vocoder_executor("model", device=None, gpu_id=None)
    assert seen[-1] == "npu"

    monkeypatch.setattr(
        fish_stages,
        "resolve_device_spec",
        lambda device, index: f"cuda:{index}" if index is not None else "cpu",
    )
    fish_stages.create_vocoder_executor("model", device=None, gpu_id=0)
    assert seen[-1] == "cuda:0"


def test_s2pro_tts_engine_stage_uses_placement_gpu_id() -> None:
    from sglang_omni.config.runtime import resolve_stage_typed_kwargs
    from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig

    config = S2ProPipelineConfig(model_path="x")
    tts_stage = next(stage for stage in config.stages if stage.name == "tts_engine")

    # Device resolution is placement's job at runtime: the stage declares
    # gpu=0 and must not bake a platform-specific device string into its
    # factory kwargs.
    assert tts_stage.gpu == 0
    assert "device" not in resolve_stage_typed_kwargs(tts_stage)


def test_s2pro_tts_engine_factory_forwards_placement_gpu_id(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder as fish_engine
    from sglang_omni.models.fishaudio_s2_pro import stages as fish_stages

    captured: dict[str, object] = {}

    def build(self, model_path, **kwargs):
        del self
        captured["model_path"] = model_path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(fish_engine.FishS2ProEngineBuilder, "build", build)

    fish_stages.create_sglang_tts_engine_executor("model", device=None, gpu_id=1)

    assert captured["model_path"] == "model"
    assert captured["device"] is None
    assert captured["gpu_id"] == 1
