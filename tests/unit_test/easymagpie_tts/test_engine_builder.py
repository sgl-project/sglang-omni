# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend

from sglang_omni.models.easymagpie_tts import compile_support, engine_builder
from sglang_omni.models.easymagpie_tts.config import EasyMagpieTTSPipelineConfig
from sglang_omni.models.easymagpie_tts.engine_builder import (
    PREFILL_GRAPH_MAX_TOKENS,
    EasyMagpieTTSEngineBuilder,
    decode_graph_batch_sizes,
)
from sglang_omni.models.easymagpie_tts.payload_types import MAX_TEXT_TOKENS
from sglang_omni.models.easymagpie_tts.speakers import SPEAKER_SUBDIR


@pytest.mark.parametrize(
    ("max_batch", "sizes"),
    [
        (1, [1]),
        (6, [1, 2, 4, 6]),
        (8, [1, 2, 4, 8]),
        (128, [1, 2, 4, 8, 16, 32, 64, 128]),
    ],
)
def test_decode_graph_sizes_end_at_the_running_limit(max_batch, sizes) -> None:
    assert decode_graph_batch_sizes(max_batch) == sizes


def test_decode_and_prefill_graphs_are_on_by_default(monkeypatch) -> None:
    monkeypatch.setattr(engine_builder, "sglang_captures_mamba_prefill", lambda: True)
    builder = EasyMagpieTTSEngineBuilder(max_running_requests=8)
    defaults = builder.generation_defaults(dtype="float16")
    assert defaults["disable_cuda_graph"] is False
    assert defaults["cuda_graph_max_bs"] == 8
    assert defaults["cuda_graph_bs"] == [1, 2, 4, 8]
    assert defaults["cuda_graph_backend_prefill"] == CudaGraphBackend.BREAKABLE
    assert defaults["cuda_graph_bs_prefill"][-1] == PREFILL_GRAPH_MAX_TOKENS
    assert "disable_prefill_cuda_graph" not in defaults
    assert builder.supports_breakable_prefill_cuda_graph is True
    eager = EasyMagpieTTSEngineBuilder(cuda_graph=False).generation_defaults(
        dtype="float16"
    )
    assert eager["disable_cuda_graph"] is True
    assert eager["disable_prefill_cuda_graph"] is True


def test_prefill_runs_eagerly_with_one_warning_on_an_older_sglang(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(engine_builder, "sglang_captures_mamba_prefill", lambda: False)
    with caplog.at_level(logging.WARNING, logger=engine_builder.__name__):
        defaults = EasyMagpieTTSEngineBuilder().generation_defaults(dtype="float16")
        EasyMagpieTTSEngineBuilder(cuda_graph=False).generation_defaults(
            dtype="float16"
        )
    assert defaults["disable_prefill_cuda_graph"] is True
    assert "cuda_graph_backend_prefill" not in defaults
    assert defaults["disable_cuda_graph"] is False
    assert len(caplog.records) == 1
    assert "prefill runs eagerly" in caplog.records[0].getMessage()


def test_the_prefill_graph_captures_the_nemotron_h_backbone(talker) -> None:
    assert talker.language_model is talker.backbone


def test_graph_buckets_follow_a_stage_running_limit_override(sglang_fixes) -> None:
    builder = EasyMagpieTTSEngineBuilder()
    overrides = {"max_running_requests": 4}
    builder.adjust_overrides(overrides)
    assert overrides["cuda_graph_max_bs"] == 4
    assert overrides["cuda_graph_bs"] == [1, 2, 4]
    with pytest.raises(ValueError, match="tp_size"):
        builder.adjust_overrides({"tp_size": 2})


@pytest.fixture
def sglang_fixes(monkeypatch):
    """Which compile fixes the installed SGLang is missing."""
    missing: list[str] = []
    monkeypatch.setattr(compile_support, "missing_compile_fixes", lambda: missing)
    return missing


def test_talker_compiles_every_decode_graph_batch_size(sglang_fixes) -> None:
    builder = EasyMagpieTTSEngineBuilder(max_running_requests=8)
    overrides = builder.generation_defaults(dtype="float16")
    builder.adjust_overrides(overrides)
    assert overrides["enable_torch_compile"] is True
    assert overrides["torch_compile_max_bs"] == 8


@pytest.mark.parametrize(
    "overrides",
    [{"enable_torch_compile": False}, {"disable_cuda_graph": True}],
)
def test_talker_compile_follows_stage_overrides(sglang_fixes, overrides) -> None:
    EasyMagpieTTSEngineBuilder().adjust_overrides(overrides)
    assert overrides["enable_torch_compile"] is False


def test_talker_stays_eager_without_the_sglang_fixes(sglang_fixes, caplog) -> None:
    sglang_fixes.append("Nemotron-H MoE runs serially under compile")
    overrides = {"enable_torch_compile": True}
    EasyMagpieTTSEngineBuilder().adjust_overrides(overrides)
    assert overrides["enable_torch_compile"] is False
    assert "Nemotron-H MoE runs serially" in caplog.text


def test_the_installed_sglang_reports_each_missing_fix() -> None:
    missing = compile_support.missing_compile_fixes()
    assert set(missing) <= {
        "Triton launches pass PDL constexprs on GPUs without PDL",
        "Mamba2 decode state updates stay in place under compile",
        "Nemotron-H MoE runs serially under compile",
        "native MoE applies routed_scaling_factor",
    }


def test_pipeline_waits_for_a_cold_compile() -> None:
    assert EasyMagpieTTSPipelineConfig.startup_timeout_s == 3600.0


def test_setup_model_sizes_decode_state_and_loads_voices(talker, tmp_path) -> None:
    voices = tmp_path / SPEAKER_SUBDIR
    voices.mkdir()
    torch.save(torch.ones(3, 8), voices / "eng.pt")
    torch.save({"speaker_encoding": torch.zeros(2, 8)}, voices / "alt.pt")
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=talker, req_to_token_pool=SimpleNamespace(size=9)
        )
    )
    EasyMagpieTTSEngineBuilder(max_running_requests=6).setup_model(
        model_worker=worker,
        checkpoint_dir=str(tmp_path),
        device="cpu",
        gpu_id=0,
        server_args=SimpleNamespace(max_running_requests=3),
    )
    state = talker.decode_state
    assert (state.max_batch, state.num_slots) == (6, 9)
    assert state.text.shape[1] == MAX_TEXT_TOKENS
    speakers = talker.speaker_table
    assert speakers.spans == {"alt": (0, 2), "eng": (2, 3)}
    assert speakers.rows.dtype == next(talker.parameters()).dtype
    torch.testing.assert_close(
        speakers.rows.sum(dim=1), torch.tensor([0.0] * 2 + [8.0] * 3)
    )


def test_async_decode_is_on_from_batch_one_by_default() -> None:
    kwargs = EasyMagpieTTSEngineBuilder().extra_scheduler_kwargs()
    assert kwargs["enable_async_decode"] is True
    assert kwargs["async_decode_min_batch_size"] == 1
    sync = EasyMagpieTTSEngineBuilder(enable_async_decode=False)
    assert sync.extra_scheduler_kwargs()["enable_async_decode"] is False
