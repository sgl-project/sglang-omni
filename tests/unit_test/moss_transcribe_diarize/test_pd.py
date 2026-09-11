# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from array import array
from pathlib import Path

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.config import resolve_stage_factory_args
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_transcribe_diarize.pd import request_builders
from sglang_omni.models.moss_transcribe_diarize.pd.config import (
    DECODE_STAGE,
    PREFILL_STAGE,
    MossTranscribeDiarizePDPipelineConfig,
)
from sglang_omni.models.moss_transcribe_diarize.pd.engine_builder import (
    MossTranscribeDiarizePDEngineBuilder,
)
from sglang_omni.models.moss_transcribe_diarize.pd.request_builders import (
    MOSS_TD_PD_RESUME_SCHEMA,
    make_state_adapters,
)
from sglang_omni.models.moss_transcribe_diarize.pd.stages import (
    create_sglang_moss_transcribe_diarize_decode_executor,
    create_sglang_moss_transcribe_diarize_prefill_executor,
)
from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    MossTranscribeDiarizeRequestData,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pd_utils import (
    DecodeContinuation,
    ReservedKV,
    continuation_from_req,
    req_from_continuation,
)
from sglang_omni.serve.openai_errors import is_bad_request_error


def _builder(pd_role: str) -> MossTranscribeDiarizePDEngineBuilder:
    return MossTranscribeDiarizePDEngineBuilder(
        max_running_requests=16,
        max_new_tokens=None,
        context_length=None,
        mem_fraction_static=0.8,
        mm_embedding_cache_size_bytes=0,
        encoder_cache_size_bytes=0,
        enable_torch_compile=False,
        torch_compile_max_bs=4,
        enable_async_decode=True,
        async_decode_min_batch_size=1,
        prefill_coalesce_requests=4,
        prefill_coalesce_wait_ms=12.0,
        prefill_coalesce_when_idle=True,
        prefill_coalesce_requires_pending_builds=True,
        prefill_coalesce_after_builds_during_decode=True,
        encoder_chunk_buckets=[1],
        encoder_torch_compile=False,
        encoder_max_batch_size=2,
        request_build_max_workers=1,
        request_build_max_pending=1,
        stream_emit_interval_s=0.05,
        pd_role=pd_role,
    )


def _prefill_req() -> Req:
    sampling = SamplingParams(
        max_new_tokens=16,
        temperature=0.0,
        top_p=0.95,
        top_k=50,
        stop_token_ids={2},
    )
    sampling.normalize(None)
    req = Req(
        rid="moss-pd-request",
        origin_input_text="",
        origin_input_ids=array("q", [10, 11, 12]),
        sampling_params=sampling,
        vocab_size=128,
        eos_token_ids={2},
    )
    req.output_ids.append(42)
    req.multimodal_inputs = object()
    payload = StagePayload(
        request_id=req.rid,
        request=OmniRequest(
            inputs={"audio_bytes": b"raw-audio-must-not-cross-pd"},
            params={"stream": False, "language": "zh"},
            metadata={"audio": b"raw-audio-metadata"},
        ),
        data={"prefill_only": torch.ones(1)},
    )
    req._omni_data = MossTranscribeDiarizeRequestData(
        input_ids=torch.tensor([10, 11, 12]),
        output_ids=req.output_ids,
        req=req,
        prompt_token_ids=[10, 11, 12],
        max_new_tokens=16,
        audio_duration_s=3.25,
        language="zh",
        engine_start_s=12.5,
        stage_payload=payload,
    )
    return req


def test_moss_pd_config_wires_two_explicit_engine_stages() -> None:
    config = MossTranscribeDiarizePDPipelineConfig(model_path="moss-td")

    assert config.resolved_entry_stage == PREFILL_STAGE
    assert config.terminal_stages == [DECODE_STAGE]
    assert config.gpu_placement == {PREFILL_STAGE: 0, DECODE_STAGE: 1}
    assert [stage.process for stage in config.stages] == [PREFILL_STAGE, DECODE_STAGE]
    assert config.stage_named(PREFILL_STAGE).factory_path.endswith(
        "create_sglang_moss_transcribe_diarize_prefill_executor"
    )
    assert config.stage_named(DECODE_STAGE).factory_path.endswith(
        "create_sglang_moss_transcribe_diarize_decode_executor"
    )

    for stage in config.stages:
        args = resolve_stage_factory_args(stage, config)
        assert "pd_role" not in args
        assert args["gpu_id"] == stage.gpu
        assert args["server_args_overrides"]["disable_radix_cache"] is True
        assert args["server_args_overrides"]["page_size"] == 1


def test_moss_pd_example_and_variant_are_discoverable() -> None:
    config_path = Path(__file__).parents[3] / "examples/configs/moss_td_pd_h100.yaml"
    config = ConfigManager.from_file(str(config_path)).config

    assert isinstance(config, MossTranscribeDiarizePDPipelineConfig)
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(
            "MossTranscribeDiarizePDPipelineConfig"
        )
        is MossTranscribeDiarizePDPipelineConfig
    )


@pytest.mark.parametrize(
    ("factory", "role"),
    [
        (create_sglang_moss_transcribe_diarize_prefill_executor, "prefill"),
        (create_sglang_moss_transcribe_diarize_decode_executor, "decode"),
    ],
)
def test_moss_pd_stage_factories_keep_role_out_of_config_surface(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    role: str,
) -> None:
    built: list[tuple[str, str, int | None]] = []

    def build(builder, model_path, *, gpu_id=None, **_kwargs):
        built.append((builder.pd_role, model_path, gpu_id))
        return object()

    monkeypatch.setattr(MossTranscribeDiarizePDEngineBuilder, "build", build)
    scheduler = factory("moss-td", gpu_id=2)

    assert scheduler is not None
    assert built == [(role, "moss-td", 2)]


def test_moss_pd_continuation_strips_audio_and_restores_result_state() -> None:
    state_builder, state_restorer = make_state_adapters()
    continuation = continuation_from_req(_prefill_req(), "moss-transfer", state_builder)
    encoded = continuation.encode()

    assert b"raw-audio-must-not-cross-pd" not in encoded
    assert b"raw-audio-metadata" not in encoded
    decoded = DecodeContinuation.decode(encoded)
    assert decoded.stage_payload["request"]["inputs"] is None
    assert decoded.stage_payload["data"] is None
    assert decoded.multimodal_resume["schema"] == MOSS_TD_PD_RESUME_SCHEMA

    pool = ReqToTokenPool(
        size=2,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
    )
    rebuilt = req_from_continuation(
        decoded,
        ReservedKV(
            slots=torch.tensor([7, 8, 9]),
            page_indices=(7, 8, 9),
            seq_len=3,
        ),
        req_to_token_pool=pool,
        state_restorer=state_restorer,
    )

    assert list(rebuilt.output_ids) == [42]
    assert rebuilt.multimodal_inputs is None
    assert rebuilt.tokenizer is None
    assert rebuilt._omni_data.prompt_token_ids == [10, 11, 12]
    assert rebuilt._omni_data.audio_duration_s == 3.25
    assert rebuilt._omni_data.language == "zh"
    assert rebuilt._omni_data.engine_start_s == 12.5
    assert rebuilt._omni_data.output_ids is rebuilt.output_ids


def test_moss_pd_rejects_streaming_before_building_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def request_builder(_payload: StagePayload) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(
        request_builders,
        "make_moss_transcribe_diarize_scheduler_adapters",
        lambda **_kwargs: (request_builder, object()),
    )
    guarded_builder, _ = request_builders.make_scheduler_adapters()
    payload = StagePayload(
        request_id="streaming-request",
        request=OmniRequest(inputs=b"audio", params={"stream": True}),
        data=None,
    )
    with pytest.raises(ValueError, match="requires stream=false"):
        guarded_builder(payload)

    assert calls == 0
    assert is_bad_request_error(RuntimeError("model requires stream=false"))


@pytest.mark.parametrize(
    ("role", "expected_stage", "expected_partner"),
    [
        ("prefill", "asr_prefill", "asr_decode"),
        ("decode", "asr_decode", None),
    ],
)
def test_moss_pd_builder_selects_explicit_scheduler_role(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    expected_stage: str,
    expected_partner: str | None,
) -> None:
    from sglang_omni.scheduling import pd_scheduler

    calls: list[tuple[str, dict]] = []

    def prefill_scheduler(**kwargs):
        calls.append(("prefill", kwargs))
        return object()

    def decode_scheduler(**kwargs):
        calls.append(("decode", kwargs))
        return object()

    state_builder = object()
    state_restorer = object()
    monkeypatch.setattr(pd_scheduler, "OmniPrefillScheduler", prefill_scheduler)
    monkeypatch.setattr(pd_scheduler, "OmniDecodeScheduler", decode_scheduler)
    monkeypatch.setattr(
        request_builders,
        "make_state_adapters",
        lambda: (state_builder, state_restorer),
    )

    builder = _builder(role)
    scheduler = builder._make_scheduler(
        model_worker=object(),
        tree_cache=object(),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        server_args=object(),
        model_config=object(),
        model_runner=object(),
        request_builder=object(),
        result_adapter=object(),
        extra_scheduler_kwargs={},
    )

    assert scheduler is not None
    assert calls[0][0] == role
    kwargs = calls[0][1]
    assert kwargs["stage_name"] == expected_stage
    if expected_partner is not None:
        assert kwargs["partner_stage"] == expected_partner
        assert kwargs["state_builder"] is state_builder
    else:
        assert kwargs["resume_schema"] == MOSS_TD_PD_RESUME_SCHEMA
        assert kwargs["state_restorer"] is state_restorer
