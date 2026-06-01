# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import typer

import sglang_omni.models.qwen3_omni.stages as qwen_stages
from sglang_omni.cli.serve import (
    apply_encoder_mem_reserve_cli_override,
    apply_mem_fraction_cli_overrides,
    apply_parallelism_cli_overrides,
)
from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    build_process_topology_plan,
    build_stage_placement_plan,
    resolve_stage_factory_args,
)
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.models.qwen3_omni.merge import decode_events, merge_for_thinker
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    build_sglang_thinker_request,
    project_preprocessing_to_mm_aggregate,
    project_talker_to_code2wav,
    project_thinker_to_decode,
    resolve_mm_aggregate_wait_sources,
    resolve_preprocessing_next_stages,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    apply_encoder_mem_reserve,
    build_sglang_server_args,
)
from sglang_omni.utils.imports import import_string
from tests.unit_test.fixtures.qwen_fakes import (
    FakeQwenTokenizer,
    make_qwen_payload,
    make_qwen_state,
)


def _stage(config: PipelineConfig, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _server_args_overrides(config: PipelineConfig, name: str) -> dict[str, object]:
    return _stage(config, name).factory_args.get("server_args_overrides", {})


def _runtime_mem_fraction_static(config, name: str) -> float | None:
    return _stage(config, name).runtime.sglang_server_args.mem_fraction_static


def test_qwen_pipeline_config_and_state_contracts() -> None:
    """Preserves Qwen text/speech topology and Qwen3OmniPipelineState coercion behavior."""
    text_config = Qwen3OmniPipelineConfig(model_path="model")
    speech_config = Qwen3OmniSpeechPipelineConfig(model_path="model")
    colocated_config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="model")

    assert [stage.name for stage in text_config.stages] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert speech_config.terminal_stages == ["decode", "code2wav"]
    assert (
        speech_config.terminal_stages_fn
        == "sglang_omni.models.qwen3_omni.request_builders.resolve_terminal_stages"
    )
    speech_thinker = _stage(speech_config, "thinker")
    speech_talker = _stage(speech_config, "talker_ar")
    text_thinker = _stage(text_config, "thinker")
    preprocessing = _stage(speech_config, "preprocessing")
    aggregate = _stage(speech_config, "mm_aggregate")
    # Speech-mode thinker streams hidden states to talker_ar AND text-token
    # ids to decode (for the streaming detokenizer); text-mode thinker
    # streams only to decode. Lock both so a regression here can't silently
    # disable per-token streaming for either path.
    request_builders_path = "sglang_omni.models.qwen3_omni.request_builders"
    assert preprocessing.next == ["image_encoder", "audio_encoder", "mm_aggregate"]
    assert preprocessing.route_fn == (
        f"{request_builders_path}.resolve_preprocessing_next_stages"
    )
    assert aggregate.wait_for == ["preprocessing", "image_encoder", "audio_encoder"]
    assert aggregate.wait_for_fn == (
        f"{request_builders_path}.resolve_mm_aggregate_wait_sources"
    )
    assert aggregate.route_fn == (
        f"{request_builders_path}.resolve_mm_aggregate_next_stages"
    )
    assert speech_thinker.stream_to == ["talker_ar", "decode"]
    assert speech_thinker.route_fn == (
        f"{request_builders_path}.resolve_thinker_next_stages"
    )
    assert speech_thinker.stream_done_to_fn == (
        f"{request_builders_path}.resolve_thinker_stream_done_targets"
    )
    assert speech_thinker.project_payload["decode"] == (
        f"{request_builders_path}.project_thinker_to_decode"
    )
    assert text_thinker.project_payload["decode"] == (
        f"{request_builders_path}.project_thinker_to_decode"
    )
    assert speech_talker.project_payload["code2wav"] == (
        f"{request_builders_path}.project_talker_to_code2wav"
    )
    assert text_thinker.stream_to == ["decode"]
    assert _stage(text_config, "decode").can_accept_stream_before_payload
    assert _stage(speech_config, "decode").can_accept_stream_before_payload
    assert _stage(speech_config, "talker_ar").can_accept_stream_before_payload
    assert _stage(speech_config, "code2wav").can_accept_stream_before_payload
    assert text_config.env_defaults == {"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"}
    assert speech_config.env_defaults == {"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"}
    assert colocated_config.env_defaults == {"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"}

    # Early-submit wiring (issue #473): the talker stage receives its
    # new_request from mm_aggregate so it can enter its deferred state
    # before the thinker finishes streaming. The thinker therefore only
    # routes its final payload to decode; its hidden-state stream to
    # talker_ar is preserved via stream_to (locked above).
    speech_aggregate = _stage(speech_config, "mm_aggregate")
    assert speech_aggregate.next == ["thinker", "talker_ar"]
    assert speech_aggregate.project_payload is not None
    assert "talker_ar" in speech_aggregate.project_payload
    assert _stage(speech_config, "thinker").next == "decode"

    text_aggregate = _stage(text_config, "mm_aggregate")
    assert text_aggregate.next == "thinker"
    assert _stage(text_config, "thinker").next == "decode"

    state = Qwen3OmniPipelineState.from_dict(
        {
            "prompt": {"input_ids": torch.tensor([1, 2]), "prompt_text": "hi"},
            "mm_inputs": "bad",
            "encoder_inputs": {"image_encoder": {"cache_key": "img"}},
            "thinker_out": {"output_ids": [3], "is_final": True},
        }
    )
    assert torch.equal(state.prompt["input_ids"], torch.tensor([1, 2]))
    assert state.mm_inputs == {}
    assert state.encoder_inputs["image_encoder"]["cache_key"] == "img"
    assert state.thinker_out["is_final"] is True


def test_qwen_thinker_to_decode_projection_drops_multimodal_tensors() -> None:
    audio_embeds = torch.ones(2, 3, device="cpu")
    hidden_states = torch.ones(4, device="cpu")
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs="hi"),
        data={
            "prompt": {"input_ids": torch.tensor([1, 2]), "prompt_text": "hi"},
            "thinker_inputs": {
                "model_inputs": {
                    "audio_embeds": audio_embeds,
                    "audio_feature_lengths": torch.tensor([2]),
                }
            },
            "thinker_out": {
                "output_ids": [3],
                "step": 1,
                "is_final": True,
                "extra_model_outputs": {"hidden_states": hidden_states},
            },
            "engine_outputs": {
                "thinker": {
                    "output_ids": [3],
                    "extra_model_outputs": {"hidden_states": hidden_states},
                }
            },
        },
    )

    projected = project_thinker_to_decode(payload)
    state = Qwen3OmniPipelineState.from_dict(projected.data)

    assert state.thinker_inputs == {}
    assert state.thinker_out["output_ids"] == [3]
    assert state.thinker_out["extra_model_outputs"] == {}
    assert state.engine_outputs["thinker"]["output_ids"] == [3]
    assert state.engine_outputs["thinker"]["extra_model_outputs"] == {}


def test_qwen_thinker_to_decode_projection_isolates_stream_state() -> None:
    stream_state = {"token_ids": [1, 2], "text": "hi", "emitted_text": ""}
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs="hi"),
        data={
            "prompt": {"input_ids": torch.tensor([1, 2]), "prompt_text": "hi"},
            "stream_state": stream_state,
            "thinker_out": {"output_ids": [3], "is_final": False},
        },
    )

    projected = project_thinker_to_decode(payload)

    assert projected.data["stream_state"] == stream_state
    assert projected.data["stream_state"] is not stream_state
    assert projected.data["stream_state"]["token_ids"] is not stream_state["token_ids"]


def test_qwen_talker_to_code2wav_projection_keeps_only_request_latch() -> None:
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs="hi", params={"stream": False}),
        data={
            "prompt": {"input_ids": torch.tensor([1, 2]), "prompt_text": "hi"},
            "thinker_inputs": {
                "model_inputs": {
                    "audio_embeds": torch.ones(2, 3),
                }
            },
            "thinker_out": {
                "extra_model_outputs": {"hidden_states": torch.ones(4)},
            },
        },
    )

    projected = project_talker_to_code2wav(payload)

    assert projected.request_id == payload.request_id
    assert projected.request is payload.request
    assert projected.data == {}


def test_qwen_speech_config_wires_request_granular_active_subgraph() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="model")
    aggregate = _stage(config, "mm_aggregate")
    thinker = _stage(config, "thinker")
    aggregate_route_fn = import_string(aggregate.route_fn)
    route_fn = import_string(thinker.route_fn)
    stream_done_to_fn = import_string(thinker.stream_done_to_fn)
    terminal_stages_fn = import_string(config.terminal_stages_fn)

    text_payload = StagePayload(
        request_id="text",
        request=OmniRequest(inputs=[], metadata={"output_modalities": ["text"]}),
        data={},
    )
    audio_payload = StagePayload(
        request_id="audio",
        request=OmniRequest(inputs=[], metadata={"output_modalities": ["audio"]}),
        data={},
    )
    default_payload = StagePayload(
        request_id="default",
        request=OmniRequest(inputs=[]),
        data={},
    )

    assert aggregate_route_fn("text", text_payload) == "thinker"
    assert route_fn("text", text_payload) == "decode"
    assert stream_done_to_fn("text", text_payload) == ["decode"]
    assert terminal_stages_fn(text_payload.request) == ["decode"]

    assert aggregate_route_fn("audio", audio_payload) == ["thinker", "talker_ar"]
    assert route_fn("audio", audio_payload) == "decode"
    assert stream_done_to_fn("audio", audio_payload) == ["talker_ar", "decode"]
    assert terminal_stages_fn(audio_payload.request) == ["decode", "code2wav"]

    assert aggregate_route_fn("default", default_payload) == ["thinker", "talker_ar"]
    assert route_fn("default", default_payload) == "decode"
    assert stream_done_to_fn("default", default_payload) == ["talker_ar", "decode"]
    assert terminal_stages_fn(default_payload.request) == ["decode", "code2wav"]


def test_qwen_preprocessing_routes_only_active_encoder_branches() -> None:
    def _payload(encoder_inputs):
        return make_qwen_payload(make_qwen_state(encoder_inputs=encoder_inputs))

    cases = [
        (
            {
                "image_encoder": {"_skip": True, "_result": {}},
                "audio_encoder": {"_skip": True, "_result": {}},
            },
            ["mm_aggregate"],
            ["preprocessing"],
        ),
        (
            {"audio_encoder": {}},
            ["mm_aggregate"],
            ["preprocessing"],
        ),
        (
            {"audio_encoder": {"cache_key": "audio-cache"}},
            ["mm_aggregate"],
            ["preprocessing"],
        ),
        (
            {
                "audio_encoder": {
                    "_active": False,
                    "input_features": torch.ones((1, 2, 3)),
                }
            },
            ["mm_aggregate"],
            ["preprocessing"],
        ),
        (
            {"image_encoder": {}},
            ["mm_aggregate"],
            ["preprocessing"],
        ),
        (
            {"audio_encoder": {"input_features": torch.ones((1, 2, 3))}},
            ["audio_encoder", "mm_aggregate"],
            ["preprocessing", "audio_encoder"],
        ),
        (
            {"image_encoder": {"pixel_values": torch.ones((1, 3))}},
            ["image_encoder", "mm_aggregate"],
            ["preprocessing", "image_encoder"],
        ),
        (
            {"image_encoder": {"pixel_values_videos": torch.ones((1, 3))}},
            ["image_encoder", "mm_aggregate"],
            ["preprocessing", "image_encoder"],
        ),
        (
            {
                "image_encoder": {"pixel_values": torch.ones((1, 3))},
                "audio_encoder": {"input_features": torch.ones((1, 2, 3))},
            },
            ["image_encoder", "audio_encoder", "mm_aggregate"],
            ["preprocessing", "image_encoder", "audio_encoder"],
        ),
    ]

    for encoder_inputs, expected_next, expected_wait in cases:
        payload = _payload(encoder_inputs)
        assert resolve_preprocessing_next_stages(payload.request_id, payload) == (
            expected_next
        )
        aggregate_payload = project_preprocessing_to_mm_aggregate(payload)
        assert (
            resolve_mm_aggregate_wait_sources(
                aggregate_payload.request_id,
                "preprocessing",
                aggregate_payload,
            )
            == expected_wait
        )
        assert (
            resolve_mm_aggregate_wait_sources(
                aggregate_payload.request_id,
                "audio_encoder",
                aggregate_payload,
            )
            is None
        )


def test_qwen_aggregate_wait_sources_accept_projected_active_metadata() -> None:
    payload = make_qwen_payload(
        make_qwen_state(
            encoder_inputs={
                "audio_encoder": {"cache_key": "audio-cache", "_active": True}
            }
        )
    )

    assert resolve_mm_aggregate_wait_sources(
        payload.request_id,
        "preprocessing",
        payload,
    ) == ["preprocessing", "audio_encoder"]


def test_qwen_aggregate_projection_marks_uncached_active_encoder_inputs() -> None:
    state = make_qwen_state(
        encoder_inputs={
            "audio_encoder": {"input_features": torch.ones((1, 2, 3))},
            "image_encoder": {"_skip": True, "_result": {}},
        }
    )

    projected = project_preprocessing_to_mm_aggregate(make_qwen_payload(state))
    projected_state = Qwen3OmniPipelineState.from_dict(projected.data)

    assert projected_state.encoder_inputs == {
        "audio_encoder": {"_active": True},
        "image_encoder": {"_skip": True},
    }
    assert resolve_mm_aggregate_wait_sources(
        projected.request_id,
        "preprocessing",
        projected,
    ) == ["preprocessing", "audio_encoder"]


def test_qwen_builder_omits_mem_fraction_static_by_default() -> None:
    server_args = build_sglang_server_args(
        "dummy",
        context_length=8192,
        tp_size=2,
        random_seed=777,
    )

    assert server_args.mem_fraction_static is None
    assert server_args.context_length == 8192
    assert server_args.tp_size == 2
    assert server_args.random_seed == 777


def test_qwen_builder_forwards_explicit_mem_fraction_static() -> None:
    server_args = build_sglang_server_args(
        "dummy",
        context_length=4096,
        mem_fraction_static=0.82,
        dtype="bfloat16",
    )

    assert server_args.mem_fraction_static == 0.82
    assert server_args.dtype == "bfloat16"


def test_qwen_encoder_mem_reserve_applies_only_to_valid_auto_values() -> None:
    server_args = SimpleNamespace(mem_fraction_static=0.929)

    apply_encoder_mem_reserve(server_args, 0.05)

    assert server_args.mem_fraction_static == 0.879

    apply_encoder_mem_reserve(server_args, 0.0)
    assert server_args.mem_fraction_static == 0.879

    with pytest.raises(ValueError, match="below the safe floor"):
        apply_encoder_mem_reserve(SimpleNamespace(mem_fraction_static=0.15), 0.10)

    for invalid_reserve in (-0.01, 1.0):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            apply_encoder_mem_reserve(
                SimpleNamespace(mem_fraction_static=0.929),
                invalid_reserve,
            )


def test_qwen_cli_global_and_specific_mem_fraction_target_only_ar_stages() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=0.70,
        talker_mem_fraction_static=0.65,
    )

    assert _runtime_mem_fraction_static(config, "thinker") == 0.70
    assert _runtime_mem_fraction_static(config, "talker_ar") == 0.65
    for non_ar_stage in ("image_encoder", "audio_encoder", "code2wav"):
        assert "server_args_overrides" not in _stage(config, non_ar_stage).factory_args


def test_qwen_cli_per_role_mem_fraction_overrides_global_when_all_three_passed() -> (
    None
):
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=0.70,
        talker_mem_fraction_static=0.65,
    )

    assert _runtime_mem_fraction_static(config, "thinker") == 0.70
    assert _runtime_mem_fraction_static(config, "talker_ar") == 0.65


def test_qwen_cli_global_mem_fraction_applies_when_no_per_role_override() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )

    assert _runtime_mem_fraction_static(config, "thinker") == 0.80
    assert _runtime_mem_fraction_static(config, "talker_ar") == 0.80


def test_qwen_cli_partial_per_role_falls_back_to_global_for_unspecified_role() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=0.70,
        talker_mem_fraction_static=None,
    )

    assert _runtime_mem_fraction_static(config, "thinker") == 0.70
    assert _runtime_mem_fraction_static(config, "talker_ar") == 0.80


def test_qwen_cli_talker_per_role_overrides_global_thinker_falls_back() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=0.65,
    )

    assert _runtime_mem_fraction_static(config, "thinker") == 0.80
    assert _runtime_mem_fraction_static(config, "talker_ar") == 0.65


def test_qwen_cli_mem_fraction_static_survives_runtime_overrides_overlay() -> None:
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        runtime_overrides={
            "thinker": {"server_args_overrides": {"disable_cuda_graph": True}}
        },
    )

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )

    resolved = resolve_stage_factory_args(_stage(config, "thinker"), config)
    assert resolved["server_args_overrides"]["mem_fraction_static"] == 0.80
    assert resolved["server_args_overrides"]["disable_cuda_graph"] is True


@pytest.mark.parametrize(
    (
        "speech_enabled",
        "expected_infrastructure_graph_disabled",
        "expected_capture_hidden_layers",
        "expected_init_graph_calls",
    ),
    [
        (False, False, None, 0),
        (True, True, [0, 24], 1),
    ],
)
def test_qwen_thinker_cuda_graph_capture_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    speech_enabled: bool,
    expected_infrastructure_graph_disabled: bool,
    expected_capture_hidden_layers: list[int] | None,
    expected_init_graph_calls: int,
) -> None:
    from sglang.srt.utils import hf_transformers_utils

    from sglang_omni.model_runner import thinker_model_runner
    from sglang_omni.models.qwen3_omni import bootstrap, request_builders
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling import omni_scheduler, sglang_backend

    server_args = SimpleNamespace(disable_cuda_graph=False)
    infrastructure_saw_graph_disabled: list[bool] = []
    capture_hidden_layers_seen: list[list[int] | None] = []
    init_graph_calls = 0

    class FakeModelRunner:
        model = object()

        def init_device_graphs(self) -> None:
            nonlocal init_graph_calls
            init_graph_calls += 1
            assert server_args.disable_cuda_graph is False

    model_config = SimpleNamespace(
        model_path="model",
        vocab_size=10,
        hf_config=SimpleNamespace(thinker_config=object()),
    )
    model_worker = SimpleNamespace(
        model_runner=FakeModelRunner(),
        model_config=model_config,
    )

    def fake_create_infrastructure(*args, **kwargs):
        infrastructure_saw_graph_disabled.append(bool(args[0].disable_cuda_graph))
        capture_hidden_layers_seen.append(kwargs.get("capture_hidden_layers"))
        return (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            model_config,
        )

    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        fake_create_infrastructure,
    )
    monkeypatch.setattr(
        hf_transformers_utils, "get_tokenizer", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        request_builders,
        "make_thinker_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(request_builders, "make_thinker_stream_output_builder", object)
    monkeypatch.setattr(
        request_builders, "should_generate_audio_output", lambda payload: False
    )
    monkeypatch.setattr(
        sglang_backend, "SGLangOutputProcessor", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        thinker_model_runner,
        "ThinkerModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        omni_scheduler,
        "OmniScheduler",
        SimpleNamespace,
    )

    scheduler = bootstrap.create_thinker_scheduler(
        server_args, speech_enabled=speech_enabled
    )

    assert infrastructure_saw_graph_disabled == [expected_infrastructure_graph_disabled]
    assert capture_hidden_layers_seen == [expected_capture_hidden_layers]
    assert init_graph_calls == expected_init_graph_calls
    assert getattr(server_args, "enable_return_hidden_states", False) is speech_enabled
    assert server_args.disable_cuda_graph is False
    assert scheduler.server_args is server_args


def test_qwen_cli_mem_fraction_static_rejects_runtime_override_duplicate() -> None:
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        runtime_overrides={
            "thinker": {"server_args_overrides": {"mem_fraction_static": 0.70}}
        },
    )

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )

    with pytest.raises(ValueError, match="mem_fraction_static"):
        resolve_stage_factory_args(_stage(config, "thinker"), config)


def test_qwen_cli_rejects_talker_override_on_text_only_qwen_without_partial_write() -> (
    None
):
    config = Qwen3OmniPipelineConfig(model_path="dummy")
    original = config.model_dump()

    with pytest.raises(typer.BadParameter, match="talker"):
        apply_mem_fraction_cli_overrides(
            config,
            mem_fraction_static=None,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=0.65,
        )

    assert config.model_dump() == original


def test_qwen_cli_rejects_invalid_mem_fraction_without_partial_write() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    original = config.model_dump()

    with pytest.raises(typer.BadParameter, match="must be > 0 and < 1"):
        apply_mem_fraction_cli_overrides(
            config,
            mem_fraction_static=1.0,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=None,
        )

    assert config.model_dump() == original


def test_qwen_cli_rejects_global_mem_fraction_when_pipeline_has_no_supported_roles() -> (
    None
):
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory=(
                    "sglang_omni.models.qwen3_omni.stages."
                    "create_preprocessing_executor"
                ),
                terminal=True,
            )
        ],
    )

    with pytest.raises(typer.BadParameter, match="supported"):
        apply_mem_fraction_cli_overrides(
            config,
            mem_fraction_static=0.80,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=None,
        )


def test_qwen_cli_encoder_mem_reserve_routes_as_thinker_factory_arg() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_encoder_mem_reserve_cli_override(
        config,
        encoder_mem_reserve=0.15,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
    )

    thinker_args = _stage(config, "thinker").factory_args
    assert thinker_args["encoder_mem_reserve"] == 0.15
    assert "encoder_mem_reserve" not in thinker_args.get("server_args_overrides", {})
    assert "encoder_mem_reserve" not in _stage(config, "talker_ar").factory_args


def test_qwen_cli_encoder_mem_reserve_is_exclusive_with_thinker_auto_path_pins() -> (
    None
):
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=0.80,
            thinker_mem_fraction_static=None,
        )

    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=None,
            thinker_mem_fraction_static=0.70,
        )


def test_qwen_cli_encoder_mem_reserve_rejects_config_pinned_thinker_mem_fraction() -> (
    None
):
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    thinker_args = _stage(config, "thinker").factory_args
    thinker_args["server_args_overrides"] = {"mem_fraction_static": 0.70}

    with pytest.raises(typer.BadParameter, match="not explicitly pinned"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=None,
            thinker_mem_fraction_static=None,
        )


def test_qwen_cli_encoder_mem_reserve_rejects_runtime_pinned_thinker_mem_fraction() -> (
    None
):
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        runtime_overrides={
            "thinker": {"server_args_overrides": {"mem_fraction_static": 0.70}}
        },
    )

    with pytest.raises(typer.BadParameter, match="not explicitly pinned"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=None,
            thinker_mem_fraction_static=None,
        )


def test_qwen_cli_encoder_mem_reserve_rejects_typed_pinned_thinker_mem_fraction() -> (
    None
):
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    _stage(config, "thinker").runtime.sglang_server_args.mem_fraction_static = 0.70

    with pytest.raises(typer.BadParameter, match="not explicitly pinned"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=None,
            thinker_mem_fraction_static=None,
        )


def test_qwen_cli_encoder_mem_reserve_survives_runtime_overrides_overlay() -> None:
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        runtime_overrides={"thinker": {"encoder_mem_reserve": 0.10}},
    )

    apply_encoder_mem_reserve_cli_override(
        config,
        encoder_mem_reserve=0.15,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
    )

    resolved = resolve_stage_factory_args(_stage(config, "thinker"), config)

    assert resolved["encoder_mem_reserve"] == 0.15


def test_qwen_cli_thinker_tp_override_keeps_parallelism_alias_in_sync() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=2,
        thinker_gpus="0,1",
        talker_gpu=None,
        code2wav_gpu=None,
    )

    thinker = _stage(config, "thinker")
    assert thinker.tp_size == 2
    assert thinker.parallelism.tp == 2
    assert thinker.gpu == [0, 1]


def test_qwen_text_thinker_tp_builds_topology_without_memory_fractions() -> None:
    config = Qwen3OmniPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.82,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )
    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=2,
        thinker_gpus="0,1",
        talker_gpu=None,
        code2wav_gpu=None,
    )

    placement = build_stage_placement_plan(config)
    build_process_topology_plan(config, placement)

    thinker = _stage(config, "thinker")
    assert thinker.tp_size == 2
    assert thinker.gpu == [0, 1]
    assert _stage(config, "thinker").runtime.resources.total_gpu_memory_fraction is None


def test_qwen_thinker_auto_path_applies_encoder_reserve() -> None:
    server_args = SimpleNamespace(mem_fraction_static=0.929)

    applied = qwen_stages._apply_qwen_thinker_encoder_reserve(
        server_args,
        has_explicit_mem_fraction_static=False,
        encoder_mem_reserve=0.05,
    )

    assert applied is True
    assert server_args.mem_fraction_static == 0.879


def test_qwen_thinker_explicit_pin_bypasses_encoder_reserve() -> None:
    server_args = SimpleNamespace(mem_fraction_static=0.70)

    applied = qwen_stages._apply_qwen_thinker_encoder_reserve(
        server_args,
        has_explicit_mem_fraction_static=True,
        encoder_mem_reserve=0.20,
    )

    assert applied is False
    assert server_args.mem_fraction_static == 0.70


def test_qwen_thinker_encoder_reserve_rejects_below_safe_floor() -> None:
    with pytest.raises(ValueError, match="below the safe floor"):
        qwen_stages._apply_qwen_thinker_encoder_reserve(
            SimpleNamespace(mem_fraction_static=0.15),
            has_explicit_mem_fraction_static=False,
            encoder_mem_reserve=0.10,
        )


def test_qwen_factory_signatures_keep_reserve_thinker_only() -> None:
    thinker_sig = inspect.signature(
        qwen_stages.create_sglang_thinker_executor_from_config
    )
    talker_sig = inspect.signature(qwen_stages.create_talker_ar_executor_from_config)

    assert thinker_sig.parameters["encoder_mem_reserve"].default == 0.05
    assert "encoder_mem_reserve" not in talker_sig.parameters


def test_qwen_mm_aggregate_keeps_lightweight_inputs_and_prunes_after_merge() -> None:
    """Preserves lightweight fan-in payloads and prunes consumed encoder tensors."""
    state = make_qwen_state(
        mm_inputs={
            "image": {
                "pixel_values": torch.ones((2, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
            "audio": {
                "feature_attention_mask": torch.ones((1, 2), dtype=torch.long),
                "audio_feature_lengths": torch.tensor([2]),
            },
        },
        encoder_inputs={
            "image_encoder": {
                "cache_key": "image-cache",
                "pixel_values": torch.ones((2, 3)),
            },
            "audio_encoder": {
                "cache_key": "audio-cache",
                "input_features": torch.ones((1, 2, 3)),
            },
        },
    )

    projected = project_preprocessing_to_mm_aggregate(make_qwen_payload(state))
    projected_state = Qwen3OmniPipelineState.from_dict(projected.data)
    assert "pixel_values" not in projected_state.mm_inputs["image"]
    assert projected_state.encoder_inputs == {
        "image_encoder": {"cache_key": "image-cache", "_active": True},
        "audio_encoder": {"cache_key": "audio-cache", "_active": True},
    }

    image_state = Qwen3OmniPipelineState(
        encoder_outs={"image_encoder": {"image_embeds": torch.ones((2, 2))}}
    )
    audio_state = Qwen3OmniPipelineState(
        encoder_outs={
            "audio_encoder": {
                "audio_embeds": torch.ones((2, 2)),
                "audio_feature_lengths": torch.tensor([2]),
            }
        }
    )
    merged = merge_for_thinker(
        {
            "preprocessing": projected,
            "image_encoder": make_qwen_payload(image_state),
            "audio_encoder": make_qwen_payload(audio_state),
        }
    )
    merged_state = Qwen3OmniPipelineState.from_dict(merged.data)
    assert merged_state.encoder_inputs == {}
    assert merged_state.encoder_outs == {}
    assert "image_embeds" in merged_state.thinker_inputs["model_inputs"]
    assert "audio_embeds" in merged_state.thinker_inputs["model_inputs"]
    assert "pixel_values" not in merged_state.mm_inputs["image"]
    assert "input_features" not in merged_state.mm_inputs["audio"]
    assert merged_state.thinker_inputs["media_cache_keys"] == {
        "image": "image:image-cache",
        "video": "video:image-cache",
        "audio": "audio:audio-cache",
    }


def test_qwen_thinker_request_and_decode_contracts() -> None:
    """Preserves incremental text deltas, replacement-char suppression, and final text."""
    stream_state = Qwen3OmniPipelineState()
    tokenizer = FakeQwenTokenizer(pieces={1: "A", 2: "\ufffd", 3: "B"})
    first = list(
        decode_events(
            thinker_out={"output_ids": [1]},
            state=stream_state,
            tokenizer=tokenizer,
            eos_token_id=99,
            step=1,
        )
    )
    dropped = list(
        decode_events(
            thinker_out={"output_ids": [2]},
            state=stream_state,
            tokenizer=tokenizer,
            eos_token_id=99,
            step=2,
        )
    )
    final = list(
        decode_events(
            thinker_out={"output_ids": [1, 3, 99], "is_final": True},
            state=stream_state,
            tokenizer=FakeQwenTokenizer(pieces={1: "A", 3: "B"}),
            eos_token_id=99,
            step=3,
        )
    )
    assert first[0].payload == {"text": "A"}
    assert dropped == []
    assert final[0].type == "text_final"
    assert final[0].payload == {"text": "AB"}


def test_qwen_sglang_request_hashes_media_tokens_without_changing_mrope_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves hashed media pad tokens while M-RoPE still sees original ids."""
    captured: dict[str, torch.Tensor] = {}

    def fake_mrope(input_ids, model_inputs, thinker_config):
        del model_inputs, thinker_config
        captured["input_ids"] = input_ids.clone()
        return torch.zeros((3, input_ids.numel()), dtype=torch.long), torch.tensor(0)

    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.request_builders._compute_mrope_positions",
        fake_mrope,
    )

    audio_token_id = 77
    input_ids = torch.tensor([10, audio_token_id, 11], dtype=torch.long)
    state = make_qwen_state(
        prompt={"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        thinker_inputs={
            "model_inputs": {"audio_embeds": torch.ones((1, 4))},
            "media_cache_keys": {"audio": "audio:cache"},
        },
    )
    req_data = build_sglang_thinker_request(
        state,
        params={"max_new_tokens": 3, "seed": 123},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        request_id="rid-1",
        thinker_config=SimpleNamespace(
            image_token_id=55,
            video_token_id=66,
            audio_token_id=audio_token_id,
        ),
    )

    pad_values = req_data.req.omni_model_inputs["pad_values"]
    assert pad_values["audio"] >= 256
    assert int(req_data.input_ids[1]) == pad_values["audio"]
    assert captured["input_ids"].tolist() == input_ids.tolist()
