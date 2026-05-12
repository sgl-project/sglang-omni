# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import typer

import sglang_omni_v1.models.qwen3_omni.stages as qwen_stages
from sglang_omni_v1.cli.serve import (
    apply_encoder_mem_reserve_cli_override,
    apply_mem_fraction_cli_overrides,
)
from sglang_omni_v1.config import PipelineConfig, StageConfig
from sglang_omni_v1.config.compiler import _resolve_factory_args
from sglang_omni_v1.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni_v1.models.qwen3_omni.merge import decode_events, merge_for_thinker
from sglang_omni_v1.models.qwen3_omni.payload_types import PipelineState
from sglang_omni_v1.models.qwen3_omni.request_builders import (
    build_sglang_thinker_request,
    project_preprocessing_to_mm_aggregate,
)
from sglang_omni_v1.scheduling.sglang_backend.server_args_builder import (
    apply_encoder_mem_reserve,
    build_sglang_server_args,
)
from tests.unit_test.fixtures.qwen_fakes import (
    FakeQwenTokenizer,
    make_qwen_payload,
    make_qwen_state,
)


def _stage(config: PipelineConfig, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _server_args_overrides(config: PipelineConfig, name: str) -> dict[str, object]:
    return _stage(config, name).factory_args.get("server_args_overrides", {})


def test_qwen_pipeline_config_and_state_contracts() -> None:
    """Preserves Qwen text/speech topology and PipelineState coercion behavior."""
    text_config = Qwen3OmniPipelineConfig(model_path="model")
    speech_config = Qwen3OmniSpeechPipelineConfig(model_path="model")

    assert [stage.name for stage in text_config.stages] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert speech_config.terminal_stages == ["decode", "code2wav"]
    assert {stage.name: stage for stage in speech_config.stages}[
        "thinker"
    ].stream_to == ["talker_ar"]

    state = PipelineState.from_dict(
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

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.70
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.65
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

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.70
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.65


def test_qwen_cli_global_mem_fraction_applies_when_no_per_role_override() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.80
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.80


def test_qwen_cli_partial_per_role_falls_back_to_global_for_unspecified_role() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=0.70,
        talker_mem_fraction_static=None,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.70
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.80


def test_qwen_cli_talker_per_role_overrides_global_thinker_falls_back() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=0.65,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.80
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.65


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

    resolved = _resolve_factory_args(_stage(config, "thinker"), config)
    assert resolved["server_args_overrides"]["mem_fraction_static"] == 0.80
    assert resolved["server_args_overrides"]["disable_cuda_graph"] is True


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
                factory=(
                    "sglang_omni_v1.models.qwen3_omni.stages."
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

    resolved = _resolve_factory_args(_stage(config, "thinker"), config)

    assert resolved["encoder_mem_reserve"] == 0.15


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
            "audio": {"audio_feature_lengths": torch.tensor([1])},
        },
        encoder_inputs={
            "image_encoder": {"cache_key": "image-cache"},
            "audio_encoder": {"cache_key": "audio-cache"},
        },
    )

    projected = project_preprocessing_to_mm_aggregate(make_qwen_payload(state))
    projected_state = PipelineState.from_dict(projected.data)
    assert "pixel_values" not in projected_state.mm_inputs["image"]
    assert projected_state.encoder_inputs == {
        "image_encoder": {"cache_key": "image-cache"},
        "audio_encoder": {"cache_key": "audio-cache"},
    }

    image_state = PipelineState(
        encoder_outs={"image_encoder": {"image_embeds": torch.ones((2, 2))}}
    )
    merged = merge_for_thinker(
        {
            "preprocessing": make_qwen_payload(state),
            "image_encoder": make_qwen_payload(image_state),
        }
    )
    merged_state = PipelineState.from_dict(merged.data)
    assert merged_state.encoder_inputs == {}
    assert merged_state.encoder_outs == {}
    assert "image_embeds" in merged_state.thinker_inputs["model_inputs"]
    assert merged_state.thinker_inputs["media_cache_keys"] == {
        "image": "image:image-cache",
        "video": "video:image-cache",
        "audio": "audio:audio-cache",
    }


def test_qwen_thinker_request_and_decode_contracts() -> None:
    """Preserves incremental text deltas, replacement-char suppression, and final text."""
    stream_state = PipelineState()
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
        "sglang_omni_v1.models.qwen3_omni.request_builders._compute_mrope_positions",
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
