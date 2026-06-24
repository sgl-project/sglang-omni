# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the CUDA graph batch validator (mocked objects, no GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import sglang_omni.utils.cuda_graph_batch_validator as cgv
from sglang_omni.utils.cuda_graph_batch_validator import (
    CudaGraphBatchReport,
    CustomGraphReport,
    NoGraphReport,
    evaluate_cuda_graph_batch_sizing,
    read_captured_bs,
    read_model_buffer_capacity,
    sglang_stage_names,
    validate_stage,
    validate_stage_scheduler,
    validate_stages,
)


class _FakeTensor:
    """Minimal stand-in for a torch tensor exposing only ``.shape``."""

    def __init__(self, first_dim: int):
        self.shape = (first_dim, 8)


def _fake_runner(
    *,
    model=None,
    max_running_requests=64,
    cuda_graph_max_bs=64,
    capture_bs=None,
    request_slots=64,
    has_graph_runner=True,
):
    graph_runner = SimpleNamespace(capture_bs=capture_bs) if has_graph_runner else None
    return SimpleNamespace(
        server_args=SimpleNamespace(
            max_running_requests=max_running_requests,
            cuda_graph_max_bs=cuda_graph_max_bs,
        ),
        req_to_token_pool=SimpleNamespace(size=request_slots),
        graph_runner=graph_runner,
        model=model,
    )


# --- pure verdict logic ---------------------------------------------------


def test_consistent_sizing_is_ok():
    report = evaluate_cuda_graph_batch_sizing(
        stage="tts_engine",
        max_running_requests=64,
        cuda_graph_max_bs=64,
        captured_bs=[1, 2, 4, 8, 16, 32, 64],
        request_slots=64,
        buffer_capacity=65,
    )
    assert report.ok
    assert report.max_captured_bs == 64


def test_captured_exceeds_buffer_is_flagged():
    report = evaluate_cuda_graph_batch_sizing(
        stage="tts_engine",
        max_running_requests=64,
        cuda_graph_max_bs=128,
        captured_bs=[1, 16, 64, 128],
        request_slots=128,
        buffer_capacity=65,
    )
    assert not report.ok
    assert any("overruns the model buffer" in f for f in report.findings)


def test_buffer_below_admission_limit_flagged():
    report = evaluate_cuda_graph_batch_sizing(
        stage="tts_engine",
        max_running_requests=64,
        cuda_graph_max_bs=64,
        captured_bs=[1, 16, 32],
        request_slots=64,
        buffer_capacity=32,
    )
    assert not report.ok
    assert any("cannot be served" in f for f in report.findings)


def test_clamped_cap_is_not_a_failure():
    report = evaluate_cuda_graph_batch_sizing(
        stage="talker_ar",
        max_running_requests=16,
        cuda_graph_max_bs=64,
        captured_bs=[1, 2, 4, 8, 16],
        request_slots=16,
        buffer_capacity=17,
    )
    assert report.ok
    assert not any("clamped" in f for f in report.findings)


def test_missing_buffer_validates_partial():
    report = evaluate_cuda_graph_batch_sizing(
        stage="voxtral_tts",
        max_running_requests=16,
        cuda_graph_max_bs=16,
        captured_bs=[1, 8, 16],
        request_slots=16,
        buffer_capacity=None,
        buffer_source="no buffer probe registered",
    )
    assert report.ok
    assert any("model-side buffer not read" in f for f in report.findings)


# --- captured-bs reader ---------------------------------------------------


def test_read_captured_bs_sorts_and_ints():
    runner = _fake_runner(capture_bs=[64, 1, 16, 4])
    assert read_captured_bs(runner) == [1, 4, 16, 64]


def test_read_captured_bs_missing_capture_returns_none():
    runner = _fake_runner(has_graph_runner=True, capture_bs=None)
    assert read_captured_bs(runner) is None


def test_read_captured_bs_no_graph_runner_returns_none():
    runner = _fake_runner(has_graph_runner=False)
    assert read_captured_bs(runner) is None


def test_read_captured_bs_malformed_element_degrades_to_none():
    runner = _fake_runner(capture_bs=[1, 16, "auto"])
    assert read_captured_bs(runner) is None


def test_read_captured_bs_empty_list_is_none():
    runner = _fake_runner(capture_bs=[])
    assert read_captured_bs(runner) is None


# --- model-side buffer reader ---------------------------------------------


def _as_named(_unused, clsname, **attrs):
    """Build an object whose ``type().__name__ == clsname`` carrying attrs."""
    obj = type(clsname, (), {})()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def test_read_buffer_higgs_sampler_pool():
    model = _as_named(
        None,
        "HiggsTTSModel",
        _sampler_pool=SimpleNamespace(seeds=_FakeTensor(65)),
    )
    cap, source = read_model_buffer_capacity(model)
    assert cap == 65
    assert "_sampler_pool.seeds.shape[0]" in source


def test_read_buffer_returns_minimum_across_registered_buffers():
    model = _as_named(
        None,
        "HiggsTTSModel",
        _sampler_pool=SimpleNamespace(seeds=_FakeTensor(65)),
        _cg_codes_BN=_FakeTensor(40),
        _cg_active_last_codes=_FakeTensor(65),
    )
    cap, source = read_model_buffer_capacity(model)
    assert cap == 40
    assert "_cg_codes_BN.shape[0]" in source


def test_read_buffer_qwen3_tts_feedback():
    model = _as_named(None, "Qwen3TTSTalker", _feedback_buffer=_FakeTensor(64))
    cap, source = read_model_buffer_capacity(model)
    assert cap == 64
    assert "_feedback_buffer.shape[0]" in source


def test_read_buffer_inner_submodule_fallback():
    inner = _as_named(None, "Inner", _feedback_buffer=_FakeTensor(32))
    model = _as_named(None, "Qwen3OmniTalker", model=inner)
    cap, source = read_model_buffer_capacity(model)
    assert cap == 32
    assert source.startswith("model.model.")


def test_read_buffer_qwen3_omni_prefers_top_level_alias():
    inner = _as_named(None, "TextModel", _feedback_buffer=_FakeTensor(48))
    model = _as_named(
        None, "Qwen3OmniTalker", model=inner, _feedback_buffer=inner._feedback_buffer
    )
    cap, source = read_model_buffer_capacity(model)
    assert cap == 48
    assert source == "model._feedback_buffer.shape[0]"


def test_read_buffer_unregistered_model():
    model = _as_named(None, "TotallyUnknownModel")
    cap, source = read_model_buffer_capacity(model)
    assert cap is None
    assert "no buffer probe registered" in source


def test_read_buffer_registered_but_unallocated():
    model = _as_named(None, "S2ProSGLangTextModel")
    cap, source = read_model_buffer_capacity(model)
    assert cap is None
    assert "none of its buffers resolved" in source


def test_read_buffer_none_model():
    cap, source = read_model_buffer_capacity(None)
    assert cap is None
    assert "no model object" in source


# --- per-stage validation (auto buffer read) ------------------------------


def test_validate_stage_auto_reads_buffer_and_passes():
    model = _as_named(None, "Qwen3TTSTalker", _feedback_buffer=_FakeTensor(64))
    runner = _fake_runner(model=model, capture_bs=[1, 2, 4, 8, 16, 32, 64])
    report = validate_stage("tts_engine", runner)
    assert report.buffer_capacity == 64
    assert report.stage == "tts_engine (Qwen3TTSTalker)"
    assert report.ok


def test_validate_stage_detects_undersized_buffer_end_to_end():
    model = _as_named(
        None, "VoxtralSGLangTTSModel", _decode_input_embed_buffer=_FakeTensor(64)
    )
    runner = _fake_runner(
        model=model,
        max_running_requests=128,
        cuda_graph_max_bs=128,
        capture_bs=[1, 16, 64, 128],
        request_slots=128,
    )
    report = validate_stage("tts_generation", runner)
    assert report.buffer_capacity == 64
    assert not report.ok
    assert any("overruns the model buffer" in f for f in report.findings)


def test_validate_stage_caller_override_buffer():
    model = _as_named(None, "TotallyUnknownModel")
    runner = _fake_runner(model=model, capture_bs=[1, 16, 64])
    report = validate_stage("tts_engine", runner, buffer_capacity=65)
    assert report.buffer_capacity == 65
    assert report.buffer_source == "caller-provided"


def test_validate_stage_unregistered_model_partial_report():
    model = _as_named(None, "TotallyUnknownModel")
    runner = _fake_runner(model=model, capture_bs=[1, 16, 64])
    report = validate_stage("tts_engine", runner)
    assert report.buffer_capacity is None
    assert report.ok
    assert any("model-side buffer not read" in f for f in report.findings)


def test_validate_stage_names_the_stage():
    model = _as_named(None, "Qwen3TTSTalker", _feedback_buffer=_FakeTensor(64))
    runner = _fake_runner(model=model, capture_bs=[1, 16, 64])
    out = validate_stage("talker_ar", runner).format()
    assert "Stage: talker_ar (Qwen3TTSTalker)" in out
    assert "VERDICT:" in out
    assert "model-side buffers:" in out


# --- per-stage enumeration -------------------------------------------------


def _stage(name, factory):
    return SimpleNamespace(name=name, factory=factory)


def test_enumerate_single_sglang_stage_tts():
    cfg = SimpleNamespace(
        stages=[
            _stage("preprocessing", "pkg.stages.create_preprocessing_executor"),
            _stage("tts_engine", "pkg.stages.create_sglang_tts_engine_executor"),
            _stage("vocoder", "pkg.stages.create_vocoder_executor"),
        ]
    )
    assert sglang_stage_names(cfg) == ["tts_engine"]


def test_enumerate_qwen3_omni_has_two_sglang_stages():
    cfg = SimpleNamespace(
        stages=[
            _stage("preprocessing", "pkg.stages.create_preprocessing_executor"),
            _stage("audio_encoder", "pkg.stages.create_audio_encoder_executor"),
            _stage("thinker", "pkg.stages.create_sglang_thinker_executor_from_config"),
            _stage("decode", "pkg.stages.create_decode_executor"),
            _stage("talker_ar", "pkg.stages.create_talker_ar_executor_from_config"),
            _stage(
                "code2wav",
                "pkg.components.code2wav_scheduler.create_code2wav_scheduler",
            ),
        ]
    )
    assert sglang_stage_names(cfg) == ["thinker", "talker_ar"]


def test_enumerate_voxtral_generation_stage():
    cfg = SimpleNamespace(
        stages=[
            _stage("preprocessing", "pkg.stages.create_preprocessing_executor"),
            _stage("tts_generation", "pkg.pipeline.stages.create_generation_executor"),
            _stage("vocoder", "pkg.stages.create_vocoder_executor"),
        ]
    )
    assert sglang_stage_names(cfg) == ["tts_generation"]


def test_enumerate_empty_when_no_sglang_stage():
    cfg = SimpleNamespace(
        stages=[
            _stage("preprocessing", "pkg.stages.create_preprocessing_executor"),
            _stage("vocoder", "pkg.stages.create_vocoder_executor"),
        ]
    )
    assert sglang_stage_names(cfg) == []


# --- runtime scheduler guard ----------------------------------------------


def test_validate_scheduler_runs_for_sglang_stage():
    model = _as_named(None, "Qwen3TTSTalker", _feedback_buffer=_FakeTensor(64))
    runner = _fake_runner(model=model, capture_bs=[1, 16, 64])
    scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
    report = validate_stage_scheduler("tts_engine", scheduler)
    assert isinstance(report, CudaGraphBatchReport)
    assert report.stage == "tts_engine (Qwen3TTSTalker)"


def test_validate_scheduler_plain_stage_gets_no_graph_report():
    scheduler = SimpleNamespace(process_one=lambda x: x)
    report = validate_stage_scheduler("preprocessing", scheduler)
    assert isinstance(report, NoGraphReport)
    assert report.ok
    assert "no CUDA graph" in report.format()


def test_validate_scheduler_fish_wrapped_runner_resolves():
    model = _as_named(None, "S2ProSGLangTextModel", _vq_codes=_FakeTensor(64))
    runner = _fake_runner(model=model, capture_bs=[1, 16, 64])
    fish_scheduler = type("FishScheduler", (), {})()
    fish_scheduler._model_runner = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=runner)
    )
    report = validate_stage_scheduler("tts_engine", fish_scheduler)
    assert isinstance(report, CudaGraphBatchReport)
    assert report.buffer_capacity == 64
    assert "S2ProSGLangTextModel" in report.stage


def test_validate_scheduler_sglang_graphs_disabled_gets_no_graph_report():
    model = _as_named(None, "Qwen3TTSTalker", _feedback_buffer=_FakeTensor(64))
    runner = _fake_runner(model=model, capture_bs=None)
    scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
    report = validate_stage_scheduler("tts_engine", scheduler)
    assert isinstance(report, NoGraphReport)
    assert "Qwen3TTSTalker" in report.stage
    assert report.ok


# --- custom-graph (non-SGLang) stage coverage -----------------------------


def _fake_moss_vocoder_scheduler(*, captured, batch_size, has_runner=True):
    session = SimpleNamespace(
        has_cuda_graph_runner=lambda: has_runner,
        captured_frames=lambda: list(captured),
        _batch_size=batch_size,
    )
    sched = type("MossTTSLocalStreamingVocoderScheduler", (), {})()
    sched._session = session
    return sched


def test_custom_graph_stage_reported():
    sched = _fake_moss_vocoder_scheduler(captured=[100, 25, 5], batch_size=16)
    report = validate_stage_scheduler("vocoder", sched)
    assert isinstance(report, CustomGraphReport)
    assert report.captured_frames == [5, 25, 100]
    assert report.slot_capacity == 16
    assert "vocoder (MossTTSLocalStreamingVocoderScheduler)" in report.format()
    assert any("frame count T" in f for f in report.findings)


def test_custom_graph_stage_no_capture_gets_no_graph_report():
    sched = _fake_moss_vocoder_scheduler(captured=[], batch_size=16, has_runner=False)
    report = validate_stage_scheduler("vocoder", sched)
    assert isinstance(report, NoGraphReport)
    assert report.ok


def test_custom_graph_format_is_labeled():
    sched = _fake_moss_vocoder_scheduler(captured=[1, 50], batch_size=8)
    out = validate_stage_scheduler("vocoder", sched).format()
    assert "custom graph" in out
    assert "captured frames (T):" in out
    assert "slot capacity:" in out


# --- model-level driver: one independent report per stage -----------------


def test_validate_stages_covers_every_stage_independently():
    model = _as_named(None, "Qwen3TTSTalker", _feedback_buffer=_FakeTensor(64))
    sglang_sched = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=_fake_runner(model=model, capture_bs=[1, 16, 64])
        )
    )
    preproc_sched = SimpleNamespace(process_one=lambda x: x)
    vocoder_sched = _fake_moss_vocoder_scheduler(captured=[5, 50], batch_size=16)

    reports = validate_stages(
        [
            ("preprocessing", preproc_sched),
            ("tts_engine", sglang_sched),
            ("vocoder", vocoder_sched),
        ]
    )

    assert len(reports) == 3
    assert isinstance(reports[0], NoGraphReport)
    assert isinstance(reports[1], CudaGraphBatchReport)
    assert isinstance(reports[2], CustomGraphReport)
    assert all(hasattr(r, "ok") for r in reports)
    assert "preprocessing" in reports[0].stage
    assert "tts_engine" in reports[1].stage
    assert "vocoder" in reports[2].stage


def test_validate_stages_qwen3_omni_two_sglang_stages():
    thinker_model = _as_named(None, "Qwen3OmniTalker", _feedback_buffer=_FakeTensor(16))
    talker_model = _as_named(None, "Qwen3OmniTalker", _feedback_buffer=_FakeTensor(16))
    thinker = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=_fake_runner(
                model=thinker_model,
                max_running_requests=16,
                cuda_graph_max_bs=16,
                capture_bs=[1, 8, 16],
                request_slots=16,
            )
        )
    )
    talker = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=_fake_runner(
                model=talker_model,
                max_running_requests=16,
                cuda_graph_max_bs=16,
                capture_bs=[1, 8, 16],
                request_slots=16,
            )
        )
    )
    reports = validate_stages([("thinker", thinker), ("talker_ar", talker)])
    assert [type(r).__name__ for r in reports] == [
        "CudaGraphBatchReport",
        "CudaGraphBatchReport",
    ]
    assert "thinker" in reports[0].stage
    assert "talker_ar" in reports[1].stage


# --- probe registry covers every audited model ----------------------------


def test_every_generation_model_has_a_probe():
    expected = {
        "HiggsTTSModel",
        "Qwen3TTSTalker",
        "MossTTSDelaySGLangModel",
        "MossTTSLocalSGLangModel",
        "S2ProSGLangTextModel",
        "VoxtralSGLangTTSModel",
        "Qwen3OmniTalker",
    }
    assert expected <= set(cgv._BUFFER_PROBES)
