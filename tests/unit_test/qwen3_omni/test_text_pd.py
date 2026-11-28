# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Qwen3-Omni text-only disaggregated Prefill/Decode."""

from __future__ import annotations

import asyncio
import json
import tempfile
from array import array
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, MultimodalInputs, Req
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.config import build_stage_placement_plan
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_factory_signature_args,
    resolve_stage_factory_arg_defaults,
)
from sglang_omni.config.schema import EndpointsConfig, EngineArgs, ProcessConfig
from sglang_omni.config.topology import (
    compile_logical_processes,
    resolve_tracked_binding_processes,
)
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni import bootstrap as qwen_bootstrap
from sglang_omni.models.qwen3_omni import stages as qwen_stages
from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    build_decode_result,
)
from sglang_omni.models.qwen3_omni.config import (
    THINKER_DECODE_STAGE,
    THINKER_PREFILL_STAGE,
    Qwen3OmniTextPDPipelineConfig,
    Variants,
    build_server_warmup_request,
    talker_stage,
)
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    QWEN3_OMNI_MROPE_RESUME_SCHEMA,
    QWEN3_OMNI_TEXT_PD_AUDIO_ERROR,
    THINKER_STAGE,
    build_qwen3_omni_pd_state,
    make_qwen3_omni_pd_state_restorer,
    make_thinker_scheduler_adapters,
    make_thinker_stream_output_builder,
    project_thinker_to_decode,
    resolve_preprocessing_next_stages_text_pd,
    validate_text_pd_output_modalities,
)
from sglang_omni.models.qwen3_omni.thinker_model_runner import (
    Qwen3OmniThinkerModelRunner,
)
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
from sglang_omni.scheduling.pd_receiver import DecodeContinuation, ReservedKV
from sglang_omni.scheduling.pd_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
)
from sglang_omni.scheduling.pd_utils import (
    continuation_from_req,
    request_from_continuation,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.serve.openai_errors import generation_error_status_code
from tests.unit_test.fixtures.pd_fakes import (
    FakeRequestPool,
    make_decode_scheduler_stub,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class SimpleTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Signature matches HuggingFace PreTrainedTokenizerBase.encode.
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        # Signature matches HuggingFace PreTrainedTokenizerBase.decode.
        return " ".join(f"tok{token_id}" for token_id in token_ids)


def assert_contains_no_tensors(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for item in value.values():
            assert_contains_no_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_contains_no_tensors(item)


def make_prefill_req(
    *,
    prompt_ids: list[int],
    salted_ids: list[int] | None = None,
    vocab_size: int = 152064,
    mrope_delta: torch.Tensor | None = None,
    metadata: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
    max_new_tokens: int = 16,
    stop: list[str] | None = None,
    return_logprob: bool = False,
) -> Req:
    effective_ids = salted_ids if salted_ids is not None else prompt_ids
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop=stop,
    )
    sampling_params.normalize(SimpleTokenizer() if stop else None)
    req = Req(
        rid="req-mm-1",
        origin_input_text="",
        origin_input_ids=array("q", effective_ids),
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )
    req.output_ids.append(42)
    if mrope_delta is not None:
        mm_inputs = MultimodalInputs(mm_items=[])
        mm_inputs.mrope_positions = torch.zeros((3, len(prompt_ids)), dtype=torch.int64)
        mm_inputs.mrope_position_delta = mrope_delta
        req.multimodal_inputs = mm_inputs
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor(prompt_ids, dtype=torch.long)},
        thinker_inputs={
            "model_inputs": {
                "image_embeds": torch.randn(4, 16),
                "deepstack_visual_embeds": [torch.randn(4, 16)],
                "audio_embeds": torch.randn(2, 16),
            }
        },
        stream_state={"cursor": 0},
    )
    payload = StagePayload(
        request_id="req-mm-1",
        request=OmniRequest(
            inputs={"messages": [{"role": "user", "content": "hi"}]},
            params=params or {"stream": True, "max_new_tokens": max_new_tokens},
            metadata=metadata if metadata is not None else {"sdk_path": Path("/tmp/a")},
        ),
        data=state.to_dict(),
    )
    req.omni_data = SGLangARRequestData(
        input_ids=torch.tensor(effective_ids, dtype=torch.long),
        output_ids=req.output_ids,
        req=req,
        stage_payload=payload,
        max_new_tokens=max_new_tokens,
        return_logprob=return_logprob,
        output_token_logprobs=[(-0.25, 42)] if return_logprob else [],
    )
    return req


@pytest.mark.parametrize(
    "config_name",
    [
        "qwen3_omni_text_pd_h100_bf16.yaml",
        "qwen3_omni_text_pd_h200_bf16.yaml",
    ],
)
def test_text_pd_yaml_configs_load_validate_and_track_only_decode(
    config_name: str,
) -> None:
    assert Variants["text-pd"] is Qwen3OmniTextPDPipelineConfig
    config_path = REPO_ROOT / "examples" / "configs" / config_name
    config = ConfigManager.from_file(str(config_path)).config
    assert isinstance(config, Qwen3OmniTextPDPipelineConfig)
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        THINKER_PREFILL_STAGE,
        THINKER_DECODE_STAGE,
        "decode",
    ]
    placement = build_stage_placement_plan(config)
    logical_plan, _ = compile_logical_processes(config)
    assert placement.stages[THINKER_PREFILL_STAGE].gpu_ids == (0,)
    assert placement.stages[THINKER_DECODE_STAGE].gpu_ids == (1,)
    assert config.stage_named("image_encoder").process == "image_encoder"
    assert config.stage_named("audio_encoder").process == "audio_encoder"
    assert config.env_defaults["OMP_NUM_THREADS"] == "8"
    assert config.env_defaults["TOKENIZERS_PARALLELISM"] == "false"
    assert resolve_tracked_binding_processes(config, logical_plan) == {
        THINKER_DECODE_STAGE
    }

    warmup = build_server_warmup_request(config)
    assert warmup.modalities == ["text"]
    assert warmup.talker_max_new_tokens is None


@pytest.mark.parametrize(
    ("processes", "expected_prefill_gpus", "expected_decode_gpus"),
    [
        (
            {
                THINKER_DECODE_STAGE: ProcessConfig(
                    num_replicas=3, replica_devices=[1, 2, 3]
                )
            },
            [(THINKER_PREFILL_STAGE, 0)],
            [
                ("thinker_decode@r0", 1),
                ("thinker_decode@r1", 2),
                ("thinker_decode@r2", 3),
            ],
        ),
        (
            {
                THINKER_PREFILL_STAGE: ProcessConfig(
                    num_replicas=2, replica_devices=[0, 1]
                ),
                THINKER_DECODE_STAGE: ProcessConfig(
                    num_replicas=2, replica_devices=[2, 3]
                ),
            },
            [("thinker_prefill@r0", 0), ("thinker_prefill@r1", 1)],
            [("thinker_decode@r0", 2), ("thinker_decode@r1", 3)],
        ),
    ],
)
def test_text_pd_replica_expansion_and_stage_instance_pool_wiring(
    monkeypatch: pytest.MonkeyPatch,
    processes: dict[str, ProcessConfig],
    expected_prefill_gpus: list[tuple[str, int]],
    expected_decode_gpus: list[tuple[str, int]],
) -> None:
    monkeypatch.setattr(
        "sglang_omni.pipeline.runtime_config.visible_device_count",
        lambda: 4,
    )
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="pd_") as short_ipc_root:
        config = Qwen3OmniTextPDPipelineConfig(
            model_path="dummy",
            processes=processes,
            endpoints=EndpointsConfig(base_path=short_ipc_root),
        )
        config.stage_named("image_encoder").gpu_memory_fraction = 0.03
        config.stage_named("audio_encoder").gpu_memory_fraction = 0.03
        config.stage_named(THINKER_PREFILL_STAGE).gpu_memory_fraction = 0.90
        config.stage_named(THINKER_DECODE_STAGE).gpu_memory_fraction = 0.90
        prep = prepare_pipeline_runtime(config)
        try:
            assert resolve_tracked_binding_processes(
                config, prep.logical_process_plan
            ) == {THINKER_DECODE_STAGE}
            stage_by_name = {stage.name: stage for stage in prep.stages_cfg}
            role_checks = (
                (
                    THINKER_PREFILL_STAGE,
                    expected_prefill_gpus,
                    qwen_stages.create_sglang_thinker_prefill_executor_from_config,
                ),
                (
                    THINKER_DECODE_STAGE,
                    expected_decode_gpus,
                    qwen_stages.create_sglang_thinker_decode_executor_from_config,
                ),
            )
            for logical_stage, expected_gpus, factory in role_checks:
                by_name = {
                    inst.stage_name: inst
                    for inst in prep.placement_plan.instances_of(logical_stage)
                }
                for instance_name, expected_gpu in expected_gpus:
                    assert by_name[instance_name].gpu_ids == (expected_gpu,)
                    defaults = resolve_stage_factory_arg_defaults(
                        stage_by_name[instance_name],
                        config,
                        gpu_id=by_name[instance_name].gpu_ids[0],
                    )
                    resolved = resolve_factory_signature_args(
                        factory, {}, defaults=defaults
                    )
                    assert "stage_instance_name" not in resolved
                    assert resolved["gpu_id"] == expected_gpu
        finally:
            prep.runtime_dir.close()


def test_text_pd_placement_normalizes_explicit_engine_defaults() -> None:
    config = Qwen3OmniTextPDPipelineConfig(model_path="dummy")
    config.stage_named(THINKER_PREFILL_STAGE).engine = EngineArgs(
        page_size=1, kv_cache_dtype="auto", dtype="auto"
    )
    build_stage_placement_plan(config)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda cfg: (
                setattr(cfg.stage_named(THINKER_PREFILL_STAGE), "tp_size", 2),
                setattr(cfg.stage_named(THINKER_PREFILL_STAGE), "gpu", [0, 2]),
            ),
            "tp_size == 1",
        ),
        (
            lambda cfg: setattr(cfg.stage_named(THINKER_DECODE_STAGE), "gpu", 0),
            "distinct GPUs",
        ),
        (
            lambda cfg: setattr(cfg.stage_named(THINKER_DECODE_STAGE), "gpu", None),
            "requires GPU placement",
        ),
        (
            lambda cfg: setattr(
                cfg.stage_named(THINKER_PREFILL_STAGE),
                "engine",
                EngineArgs(page_size=16),
            ),
            r"matching engine\.page_size",
        ),
        (
            lambda cfg: setattr(
                cfg.stage_named(THINKER_DECODE_STAGE),
                "engine",
                EngineArgs(kv_cache_dtype="fp8_e4m3"),
            ),
            r"matching engine\.kv_cache_dtype",
        ),
        (
            lambda cfg: setattr(
                cfg.stage_named(THINKER_DECODE_STAGE),
                "engine",
                EngineArgs(dtype="float16"),
            ),
            r"matching engine\.dtype",
        ),
        (
            lambda cfg: setattr(
                cfg.stage_named(THINKER_PREFILL_STAGE),
                "engine",
                EngineArgs(disable_radix_cache=False),
            ),
            "disable_radix_cache=True",
        ),
        (
            lambda cfg: setattr(
                cfg.stage_named(THINKER_DECODE_STAGE),
                "engine",
                EngineArgs(enable_mixed_chunk=True),
            ),
            "enable_mixed_chunk=False",
        ),
        (
            lambda cfg: setattr(
                cfg.stage_named(THINKER_DECODE_STAGE),
                "engine",
                EngineArgs(speculative_algorithm="EAGLE3"),
            ),
            "speculative decoding",
        ),
        (
            lambda cfg: cfg.stages.append(
                talker_stage(gpu=2, process="talker_ar", partial_start_min_chunks=5)
            ),
            "does not support speech stages",
        ),
    ],
)
def test_text_pd_placement_validation_rejects_invalid_configurations(
    mutate: Callable[[Qwen3OmniTextPDPipelineConfig], object],
    match: str,
) -> None:
    config = Qwen3OmniTextPDPipelineConfig(model_path="dummy")
    mutate(config)
    with pytest.raises(ValueError, match=match):
        build_stage_placement_plan(config)


@pytest.mark.parametrize(
    ("encoder_inputs", "expected_targets"),
    [
        ({}, [THINKER_PREFILL_STAGE]),
        (
            {"image_encoder": {"pixel_values": torch.zeros(1)}},
            ["image_encoder", THINKER_PREFILL_STAGE],
        ),
        (
            {"audio_encoder": {"input_features": torch.zeros(1)}},
            ["audio_encoder", THINKER_PREFILL_STAGE],
        ),
        (
            {
                "image_encoder": {"pixel_values": torch.zeros(1)},
                "audio_encoder": {"input_features": torch.zeros(1)},
            },
            ["image_encoder", "audio_encoder", THINKER_PREFILL_STAGE],
        ),
    ],
)
def test_preprocessing_next_stages_text_pd_routes_by_active_modalities(
    encoder_inputs: dict[str, object],
    expected_targets: list[str],
) -> None:
    state = Qwen3OmniPipelineState(encoder_inputs=encoder_inputs)
    payload = StagePayload(
        request_id="req-route",
        request=OmniRequest(inputs=None),
        data=state.to_dict(),
    )
    assert (
        resolve_preprocessing_next_stages_text_pd("req-route", payload)
        == expected_targets
    )


def test_text_pd_audio_rejection_fails_fast_in_preprocessing_and_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_text_pd_output_modalities(None)
    validate_text_pd_output_modalities(OmniRequest(inputs=None, metadata={}))
    validate_text_pd_output_modalities(
        OmniRequest(inputs=None, metadata={"output_modalities": ["text"]})
    )
    for bad_modalities in (["audio"], ["text", "audio"], "audio"):
        with pytest.raises(
            ValueError, match="text-pd variant supports only text"
        ) as exc_info:
            validate_text_pd_output_modalities(
                OmniRequest(inputs=None, metadata={"output_modalities": bad_modalities})
            )
        assert generation_error_status_code(exc_info.value) == 400

    for error in (
        NotImplementedError("PD does not support structured-output sampling"),
        NotImplementedError("PD does not support custom logit processors"),
        NotImplementedError("PD does not transfer retained streaming-session state"),
        ValueError("PD cannot transfer an empty prompt"),
        ValueError(QWEN3_OMNI_TEXT_PD_AUDIO_ERROR),
    ):
        assert generation_error_status_code(error) == 400

    preprocessor_call = AsyncMock(side_effect=lambda payload: payload)
    monkeypatch.setattr(
        qwen_stages,
        "Qwen3OmniPreprocessor",
        lambda **kwargs: preprocessor_call,
    )
    scheduler = qwen_stages.create_preprocessing_executor(
        "dummy-model", reject_audio_output_modality=True
    )
    audio_payload = StagePayload(
        request_id="req-audio",
        request=OmniRequest(
            inputs=None, metadata={"output_modalities": ["text", "audio"]}
        ),
        data={},
    )
    with pytest.raises(ValueError, match="text-pd variant supports only text"):
        asyncio.run(scheduler.fn(audio_payload))
    preprocessor_call.assert_not_awaited()

    default_payload = StagePayload(
        request_id="req-default",
        request=OmniRequest(inputs=None, metadata={}),
        data={},
    )
    assert asyncio.run(scheduler.fn(default_payload)) is default_payload
    preprocessor_call.assert_awaited_once_with(default_payload)

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=SimpleTokenizer(),
        vocab_size=152064,
        capture_prompt_hidden_states=False,
        speech_enabled=False,
        reject_audio_output_modality=True,
    )
    with pytest.raises(ValueError, match="text-pd variant supports only text"):
        request_builder(audio_payload)


@pytest.mark.parametrize(
    ("raw_delta", "expected_resume_delta"),
    [
        (torch.tensor([[-12.0]], dtype=torch.float32), -12),
        (torch.tensor([[7]], dtype=torch.int64), 7),
        (torch.tensor([[0.0]], dtype=torch.float32), None),
        (None, None),
    ],
)
def test_multimodal_state_roundtrip_strips_tensors_unsalts_ids_and_restores_mrope(
    raw_delta: torch.Tensor | None,
    expected_resume_delta: int | None,
) -> None:
    clean_ids = [101, 151655, 151655, 202]
    salted_ids = [101, 152064 + 987654321, 152064 + 987654321, 202]
    prefill_req = make_prefill_req(
        prompt_ids=clean_ids,
        salted_ids=salted_ids,
        mrope_delta=raw_delta,
        stop=["END"],
        return_logprob=True,
    )

    continuation = continuation_from_req(
        prefill_req, "transfer-1", build_qwen3_omni_pd_state
    )
    assert_contains_no_tensors(continuation.stage_payload)
    assert continuation.origin_input_ids == clean_ids
    assert continuation.origin_input_ids_unpadded is None
    assert continuation.stage_payload["request"]["metadata"] == {}
    assert continuation.stage_payload["request"]["params"] == {"stream": True}
    assert continuation.stage_payload["data"]["prompt"] == {"prompt_tokens": 4}
    if expected_resume_delta is None:
        assert continuation.multimodal_resume is None
    else:
        assert continuation.multimodal_resume == {
            "schema": QWEN3_OMNI_MROPE_RESUME_SCHEMA,
            "mrope_position_delta": expected_resume_delta,
        }

    decoded = DecodeContinuation.decode(continuation.encode())
    tokenizer = SimpleTokenizer()
    restorer = make_qwen3_omni_pd_state_restorer(tokenizer=tokenizer)
    req_pool = FakeRequestPool(capacity=4)
    restored = request_from_continuation(
        decoded,
        ReservedKV(
            slots=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            page_indices=(0, 1, 2, 3),
            sequence_length=4,
        ),
        req_to_token_pool=req_pool,
        state_restorer=restorer,
    )
    assert list(restored.origin_input_ids) == clean_ids
    assert list(restored.origin_input_ids_unpadded) == clean_ids
    assert all(
        0 <= token_id < prefill_req.vocab_size for token_id in restored.origin_input_ids
    )
    assert restored.sampling_params.stop_strs == ["END"]
    assert restored.omni_data.return_logprob is True
    assert restored.omni_data.output_token_logprobs == [[-0.25, 42]]
    assert restored.tokenizer is tokenizer
    assert restored.omni_model_inputs is None
    assert (
        restored._omni_consumed is None
    )  # noqa: leading-underscore  # upstream Req field
    assert (
        restored._codec_suppress_tokens is None
    )  # noqa: leading-underscore  # upstream Req field
    assert (
        restored._omni_mm_positions is None
    )  # noqa: leading-underscore  # upstream Req field
    if expected_resume_delta is None:
        assert restored.multimodal_inputs is None
    else:
        assert restored.multimodal_inputs is not None
        assert restored.multimodal_inputs.mrope_positions is None
        assert restored.multimodal_inputs.mrope_position_delta.dtype == torch.int64
        assert restored.multimodal_inputs.mrope_position_delta.tolist() == [
            [expected_resume_delta]
        ]

    # Verify Decode result adapter -> project_thinker_to_decode -> StreamingDetokenizer parity.
    restored.output_ids.append(43)
    _, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=prefill_req.vocab_size,
        capture_prompt_hidden_states=False,
        speech_enabled=False,
    )
    decode_payload = project_thinker_to_decode(result_adapter(restored.omni_data))
    final_result = build_decode_result(
        decode_payload,
        tokenizer=tokenizer,
        eos_token_id=None,
        is_streaming=False,
    )
    assert final_result["text"] == "tok42 tok43"
    assert final_result["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }


def test_decode_stream_output_emits_transferred_first_token_for_single_token_reply() -> (
    None
):
    tokenizer = SimpleTokenizer()
    prefill_req = make_prefill_req(
        prompt_ids=[10, 20, 30],
        max_new_tokens=1,
        params={"stream": True, "max_new_tokens": 1},
    )
    continuation = DecodeContinuation.decode(
        continuation_from_req(
            prefill_req, "transfer-1tok", build_qwen3_omni_pd_state
        ).encode()
    )
    req_pool = FakeRequestPool(capacity=4)
    restored = request_from_continuation(
        continuation,
        ReservedKV(
            slots=torch.tensor([0, 1, 2], dtype=torch.int64),
            page_indices=(0, 1, 2),
            sequence_length=3,
        ),
        req_to_token_pool=req_pool,
        state_restorer=make_qwen3_omni_pd_state_restorer(tokenizer=tokenizer),
    )
    restored.finished_reason = FINISH_LENGTH(1)

    _, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=prefill_req.vocab_size,
        stage_name=THINKER_STAGE,
        capture_prompt_hidden_states=False,
        speech_enabled=False,
    )
    scheduler = make_decode_scheduler_stub()
    scheduler.emit_transferred_token = True
    scheduler.result_adapter = result_adapter
    scheduler.stream_output_builder = make_thinker_stream_output_builder(
        speech_enabled=False,
        prompt_hidden_store=None,
    )
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.model_path_end_done = set()
    scheduler.pending_stream_ingress = {}
    scheduler.request_finished_callback = None
    scheduler.active_stream_targets = {}
    scheduler.emitted_spec_tokens = {}
    scheduler.enable_plain_fallback = False
    with get_context().override_server_args(weight_version="default"):
        scheduler.stream_output([restored])

    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    stream_messages = [message for message in messages if message.type == "stream"]
    result_messages = [message for message in messages if message.type == "result"]
    assert len(stream_messages) == 1
    assert stream_messages[0].target == "decode"
    assert stream_messages[0].metadata["token_id"] == 42
    assert len(result_messages) == 1
    assert messages.index(stream_messages[0]) < messages.index(result_messages[0])

    projected = project_thinker_to_decode(result_messages[0].data)
    final_result = build_decode_result(
        projected,
        tokenizer=tokenizer,
        eos_token_id=None,
        is_streaming=True,
    )
    assert final_result["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_state_builder_and_restorer_reject_malformed_mrope_and_ids() -> None:
    bad_delta_req = make_prefill_req(
        prompt_ids=[1, 2],
        mrope_delta=torch.tensor([[-1.5]], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="integer mrope_position_delta"):
        build_qwen3_omni_pd_state(bad_delta_req)

    salted_in_prompt = make_prefill_req(
        prompt_ids=[1, 152064 + 7],
        salted_ids=[1, 152064 + 7],
    )
    with pytest.raises(ValueError, match="out of vocab range"):
        build_qwen3_omni_pd_state(salted_in_prompt)

    mismatched_len = make_prefill_req(prompt_ids=[1, 2], salted_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="prompt token length mismatch"):
        build_qwen3_omni_pd_state(mismatched_len)

    restorer = make_qwen3_omni_pd_state_restorer(tokenizer=SimpleTokenizer())
    dummy_req = make_prefill_req(prompt_ids=[1, 2])
    with pytest.raises(ValueError, match="Unsupported multimodal resume schema"):
        restorer(
            dummy_req,
            dummy_req.omni_data,
            {"schema": "wrong_v1", "mrope_position_delta": 1},
        )
    with pytest.raises(ValueError, match="Expected int mrope_position_delta"):
        restorer(
            dummy_req,
            dummy_req.omni_data,
            {"schema": QWEN3_OMNI_MROPE_RESUME_SCHEMA, "mrope_position_delta": True},
        )


def test_pd_thinker_factories_resolve_real_server_args_and_reject_bad_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "tiny_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 152064,
                "max_position_embeddings": 32768,
            }
        ),
        encoding="utf-8",
    )
    model_path = str(model_dir)
    prefill_calls: list[dict[str, object]] = []
    decode_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        qwen_stages,
        "resolve_concrete_device",
        lambda device, gpu_id: torch.device("cpu", gpu_id or 0),
    )
    monkeypatch.setattr(
        "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
        lambda device: 81559,
    )
    monkeypatch.setattr("sglang_omni.platforms.current_platform.is_cuda", lambda: False)
    monkeypatch.setattr("sglang_omni.platforms.current_platform.is_npu", lambda: False)
    monkeypatch.setattr(
        "sglang_omni.platforms.current_platform.enable_thinker_decode_graph",
        lambda: True,
    )
    monkeypatch.setattr(
        qwen_stages,
        "create_thinker_prefill_scheduler",
        lambda server_args, gpu_id, **kwargs: (
            prefill_calls.append(
                {"server_args": server_args, "gpu_id": gpu_id, **kwargs}
            )
            or Mock()
        ),
    )
    monkeypatch.setattr(
        qwen_stages,
        "create_thinker_decode_scheduler",
        lambda server_args, gpu_id, **kwargs: (
            decode_calls.append(
                {"server_args": server_args, "gpu_id": gpu_id, **kwargs}
            )
            or Mock()
        ),
    )

    qwen_stages.create_sglang_thinker_prefill_executor_from_config(
        model_path,
        gpu_id=0,
        total_gpu_memory_fraction=0.92,
    )
    prefill_cfg = resolved_view(prefill_calls[-1]["server_args"])
    assert prefill_cfg.disable_radix_cache is True
    assert prefill_cfg.enable_mixed_chunk is False
    assert prefill_cfg.disable_decode_cuda_graph is True
    assert prefill_cfg.chunked_prefill_size == 8192
    assert prefill_cfg.mem_fraction_static == 0.87
    assert prefill_calls[-1]["stage_name"] == THINKER_PREFILL_STAGE
    assert prefill_calls[-1]["partner_stage"] is None
    assert prefill_calls[-1]["total_gpu_memory_fraction"] == pytest.approx(0.87)
    assert prefill_calls[-1]["max_in_flight_handoffs"] is None

    qwen_stages.create_sglang_thinker_decode_executor_from_config(
        model_path,
        gpu_id=1,
        total_gpu_memory_fraction=0.90,
    )
    decode_cfg = resolved_view(decode_calls[-1]["server_args"])
    assert decode_cfg.disable_radix_cache is True
    assert decode_cfg.enable_mixed_chunk is False
    assert decode_cfg.chunked_prefill_size == -1
    assert decode_cfg.max_prefill_tokens == 2048
    assert decode_cfg.max_running_requests == 128
    assert decode_cfg.cuda_graph_config.prefill.backend == CudaGraphBackend.DISABLED
    assert decode_cfg.mem_fraction_static == 0.90
    assert decode_calls[-1]["stage_name"] == THINKER_DECODE_STAGE
    assert decode_calls[-1]["enable_async_decode"] is True
    assert decode_calls[-1]["decode_runway_tokens"] is None

    # Verify unbudgeted Decode on 80GB H100 does not drop mem_fraction_static to ~0.69.
    qwen_stages.create_sglang_thinker_decode_executor_from_config(
        model_path,
        gpu_id=1,
        total_gpu_memory_fraction=None,
    )
    unbudgeted_decode_cfg = resolved_view(decode_calls[-1]["server_args"])
    assert unbudgeted_decode_cfg.max_prefill_tokens == 2048
    assert unbudgeted_decode_cfg.mem_fraction_static >= 0.85

    for factory in (
        qwen_stages.create_sglang_thinker_prefill_executor_from_config,
        qwen_stages.create_sglang_thinker_decode_executor_from_config,
    ):
        with pytest.raises(ValueError, match="tp_size == 1"):
            factory(model_path, gpu_id=0, tp_size=2)
        with pytest.raises(ValueError, match="disable_radix_cache=True"):
            factory(
                model_path,
                gpu_id=0,
                server_args_overrides={"disable_radix_cache": False},
            )
        with pytest.raises(ValueError, match="enable_mixed_chunk=False"):
            factory(
                model_path,
                gpu_id=0,
                server_args_overrides={"enable_mixed_chunk": True},
            )
        with pytest.raises(ValueError, match="speculative decoding"):
            factory(
                model_path,
                gpu_id=0,
                server_args_overrides={"speculative_algorithm": "EAGLE3"},
            )


def test_bootstrap_pd_schedulers_wire_roles_adapters_and_replicated_stage_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_schedulers: list[tuple[type, dict[str, object]]] = []
    created_runners: list[tuple[type, dict[str, object]]] = []

    fake_model_config = SimpleNamespace(
        model_path="dummy-model",
        vocab_size=152064,
        hf_config=SimpleNamespace(thinker_config=SimpleNamespace()),
    )
    fake_infra = (
        SimpleNamespace(model_runner=SimpleNamespace()),
        object(),
        object(),
        object(),
        fake_model_config,
    )

    def fake_create_infra(*args: object, **kwargs: object) -> tuple[object, ...]:
        return fake_infra

    def fake_prefill_cls(**kwargs: object) -> SimpleNamespace:
        created_schedulers.append((OmniPrefillScheduler, dict(kwargs)))
        return SimpleNamespace(**kwargs)

    def fake_decode_cls(**kwargs: object) -> SimpleNamespace:
        created_schedulers.append((OmniDecodeScheduler, dict(kwargs)))
        return SimpleNamespace(**kwargs)

    def fake_prefill_runner(*args: object, **kwargs: object) -> SimpleNamespace:
        created_runners.append((Qwen3OmniThinkerModelRunner, dict(kwargs)))
        return SimpleNamespace(prompt_hidden_store=None)

    def fake_decode_runner(*args: object, **kwargs: object) -> SimpleNamespace:
        created_runners.append((ThinkerModelRunner, dict(kwargs)))
        return SimpleNamespace(prompt_hidden_store=None)

    monkeypatch.setattr(
        "sglang_omni.scheduling.bootstrap.create_sglang_infrastructure",
        fake_create_infra,
    )
    monkeypatch.setattr(
        "sglang_omni.scheduling.generation_batch_policy.get_prefill_cuda_graph_backend",
        lambda server_args: CudaGraphBackend.DISABLED,
    )
    monkeypatch.setattr(
        "sglang.srt.utils.hf_transformers_utils.get_tokenizer",
        lambda *args, **kwargs: SimpleTokenizer(),
    )
    monkeypatch.setattr(
        "sglang_omni.scheduling.pd_scheduler.OmniPrefillScheduler",
        fake_prefill_cls,
    )
    monkeypatch.setattr(
        "sglang_omni.scheduling.pd_scheduler.OmniDecodeScheduler",
        fake_decode_cls,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.thinker_model_runner.Qwen3OmniThinkerModelRunner",
        fake_prefill_runner,
    )
    monkeypatch.setattr(
        "sglang_omni.model_runner.thinker_model_runner.ThinkerModelRunner",
        fake_decode_runner,
    )

    qwen_bootstrap.create_thinker_prefill_scheduler(
        SimpleNamespace(),
        gpu_id=0,
    )
    prefill_cls, prefill_kwargs = created_schedulers[-1]
    assert prefill_cls is OmniPrefillScheduler
    assert prefill_kwargs["stage_name"] == THINKER_PREFILL_STAGE
    assert prefill_kwargs["partner_stage"] is None
    assert prefill_kwargs["state_builder"] is build_qwen3_omni_pd_state
    assert prefill_kwargs["result_adapter"] is None
    assert prefill_kwargs["stream_output_builder"] is None
    assert "max_in_flight_handoffs" not in prefill_kwargs
    assert created_runners[-1] == (
        Qwen3OmniThinkerModelRunner,
        {"capture_prompt_hidden_states": False},
    )

    qwen_bootstrap.create_thinker_decode_scheduler(
        SimpleNamespace(),
        gpu_id=1,
        enable_async_decode=False,
    )
    decode_cls, decode_kwargs = created_schedulers[-1]
    assert decode_cls is OmniDecodeScheduler
    assert decode_kwargs["stage_name"] == THINKER_DECODE_STAGE
    assert decode_kwargs["resume_schema"] == QWEN3_OMNI_MROPE_RESUME_SCHEMA
    assert decode_kwargs["emit_transferred_token"] is True
    assert decode_kwargs["request_builder"] is None
    assert decode_kwargs["stream_output_builder"] is not None
    assert "decode_runway_tokens" not in decode_kwargs
    assert created_runners[-1] == (
        ThinkerModelRunner,
        {"capture_prompt_hidden_states": False},
    )
    # Verify Decode result_adapter writes thinker_out and THINKER_DECODE_STAGE without stage spoofing.
    dummy_state = Qwen3OmniPipelineState(prompt={"prompt_tokens": 2})
    dummy_payload = StagePayload(
        request_id="req-r1",
        request=OmniRequest(inputs=None),
        data=dummy_state.to_dict(),
    )
    dummy_result = SimpleNamespace(
        stage_payload=dummy_payload,
        output_ids=[42],
        extra_model_outputs={},
        spec_tokens_details=None,
    )
    adapted = decode_kwargs["result_adapter"](dummy_result)
    assert THINKER_DECODE_STAGE in adapted.data["engine_outputs"]
    assert adapted.data["thinker_out"]["output_ids"] == [42]
    projected = project_thinker_to_decode(adapted)
    decoded_final = build_decode_result(
        projected,
        tokenizer=SimpleTokenizer(),
        eos_token_id=None,
        is_streaming=False,
    )
    assert decoded_final["text"] == "tok42"
    assert decoded_final["usage"]["prompt_tokens"] == 2
