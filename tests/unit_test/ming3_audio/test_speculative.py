# SPDX-License-Identifier: Apache-2.0
"""Ming 3 audio thinker with a separate DFlash draft."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import JsonValue
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

import sglang_omni.platforms as platforms
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.ming3_audio import stages
from sglang_omni.models.ming3_audio.config import Ming3AudioPipelineConfig
from sglang_omni.modules.ming_audio.hf_config import Ming3AudioConfig
from sglang_omni.modules.ming_audio.thinker import Ming3AudioThinker
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.speculative.dflash_draft import ReducedVocabularyDFlashWorker

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolate_model_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stages.ModelRegistry, "models", dict(stages.ModelRegistry.models)
    )


def write_dflash_checkpoint(directory: Path) -> None:
    source_config = {
        "speculators_model_type": "dflash",
        "sample_from_anchor": False,
        "sliding_window_non_causal": False,
        "dtype": "bfloat16",
        "draft_vocab_size": 48000,
        "aux_hidden_state_layer_ids": [2, 21, 39],
        "mask_token_id": 156910,
        "block_size": 16,
        "transformer_layer_config": {"hidden_size": 2560, "vocab_size": 157184},
    }
    (directory / "config.json").write_text(json.dumps(source_config))
    (directory / "model.safetensors").touch()


def test_dflash_draft_is_served_synchronously_from_a_loadable_copy(
    tmp_path: Path,
) -> None:
    write_dflash_checkpoint(tmp_path)
    overrides: dict[str, JsonValue] = {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": str(tmp_path),
        "speculative_draft_model_revision": "unused-for-a-local-path",
    }
    draft_checkpoint = stages.prepare_draft_checkpoint(overrides)
    copy = Path(draft_checkpoint.name)
    config = json.loads((copy / "config.json").read_text())
    assert config["architectures"] == ["ReducedVocabularyDFlashDraftModel"]
    # note (wilsonzheng0327): speculators aux id i is the stream entering layer i, the output of layer i - 1.
    assert config["dflash_config"]["target_layer_ids"] == [1, 20, 38]
    assert overrides["speculative_draft_model_path"] == str(copy)
    assert overrides["speculative_draft_model_revision"] is None
    assert overrides["disable_overlap_schedule"] is True
    assert overrides["enable_mixed_chunk"] is False
    draft_checkpoint.cleanup()
    assert not copy.exists()


def test_dflash_draft_keeps_its_own_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The thinker's architecture must not reach the draft's config, and the draft copy goes once loaded."""
    checkpoint = tmp_path / "thinker"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text((FIXTURES / "config.json").read_text())
    draft = tmp_path / "draft"
    draft.mkdir()
    write_dflash_checkpoint(draft)
    draft_copies: list[Path] = []
    draft_configs: list[dict[str, JsonValue]] = []

    class InfrastructureReached(Exception):
        pass

    def server_args(
        model_path: str, context_length: int | None, **overrides: JsonValue
    ) -> dict[str, JsonValue]:
        # note (wilsonzheng0327): SGLang refuses DFLASH on a CPU device, so record what the stage asks for.
        draft_copy = Path(str(overrides["speculative_draft_model_path"]))
        draft_copies.append(draft_copy)
        draft_configs.append(json.loads((draft_copy / "config.json").read_text()))
        return overrides

    infrastructure = Mock(side_effect=InfrastructureReached)
    monkeypatch.setattr(stages, "build_sglang_server_args", server_args)
    monkeypatch.setattr(stages, "create_sglang_infrastructure", infrastructure)
    manager = ConfigManager(Ming3AudioPipelineConfig(model_path=str(checkpoint)))
    pipeline = manager.merge_config(
        manager.parse_extra_args(
            [
                "--thinker.factory.device",
                "cpu",
                "--thinker.engine.device",
                "cpu",
                "--thinker.engine.speculative_algorithm",
                "DFLASH",
                "--thinker.engine.speculative_draft_model_path",
                str(draft),
            ]
        )
    )
    with pytest.raises(InfrastructureReached):
        stages.create_thinker_executor(
            **resolve_stage_factory_args(pipeline.stage_named("thinker"), pipeline)
        )
    server_overrides = infrastructure.call_args.args[0]
    assert "json_model_override_args" not in server_overrides
    assert server_overrides["speculative_algorithm"] == "DFLASH"
    assert server_overrides["disable_overlap_schedule"] is True
    [draft_config] = draft_configs
    assert draft_config["architectures"] == ["ReducedVocabularyDFlashDraftModel"]
    assert (
        infrastructure.call_args.kwargs["speculative_worker_class"]
        is ReducedVocabularyDFlashWorker
    )
    assert infrastructure.call_args.kwargs["supports_speculative_decoding"] is True
    [draft_copy] = draft_copies
    assert not draft_copy.exists()


@pytest.mark.parametrize(
    ("params", "refused"),
    [
        ({"temperature": 0.7, "seed": 42, "max_new_tokens": 32}, True),
        ({"temperature": 0.0, "seed": 42, "max_new_tokens": 32}, False),
        ({"temperature": 0.7, "seed": 42, "max_new_tokens": 1}, False),
    ],
    ids=["sampled", "greedy", "one-token"],
)
def test_dflash_admission_refuses_a_seed_the_verify_would_ignore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, JsonValue],
    refused: bool,
) -> None:
    """A seeded sampled request was accepted under DFLASH, though the verify draws every later token without its seed."""
    write_dflash_checkpoint(tmp_path)
    monkeypatch.setattr(
        platforms.current_platform, "get_device", lambda index: "cpu", raising=False
    )
    monkeypatch.setattr(stages, "build_sglang_server_args", Mock())
    monkeypatch.setattr(
        stages,
        "create_sglang_infrastructure",
        Mock(
            return_value=(
                Mock(gpu_id=0),
                None,
                None,
                None,
                SimpleNamespace(vocab_size=157184),
            )
        ),
    )
    monkeypatch.setattr(
        stages, "get_tokenizer", Mock(return_value=SimpleNamespace(eos_token_id=1))
    )
    scheduler_factory = Mock()
    monkeypatch.setattr(stages, "OmniScheduler", scheduler_factory)
    stages.create_thinker_executor(
        "unused",
        max_seq_len=8192,
        enable_async_decode=False,
        device="cpu",
        server_args_overrides={
            "speculative_algorithm": "DFLASH",
            "speculative_draft_model_path": str(tmp_path),
        },
    )
    scheduler_kwargs = scheduler_factory.call_args.kwargs
    request_data = scheduler_kwargs["request_builder"](
        StagePayload(
            request_id="audio-chat",
            request=OmniRequest(inputs="", params=params),
            data={
                "input_ids": torch.tensor([3, 4, 5]),
                "audio_positions": torch.tensor([1]),
                "audio_embeddings": torch.zeros(1, 2560),
            },
        )
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.draft_worker = Mock()
    scheduler.spec_algorithm = scheduler.draft_algorithm = SpeculativeAlgorithm.DFLASH
    scheduler.model_runner = scheduler_kwargs["model_runner"]

    error = scheduler.speculative_request_error(request_data.req, request_data)

    if refused:
        assert error is not None and "seed" in error
    else:
        assert error is None


def test_dflash_capture_reuses_the_layer_output_capture() -> None:
    captured: list[list[int]] = []
    thinker = SimpleNamespace(set_dspark_layers_to_capture=captured.append)
    Ming3AudioThinker.set_dflash_layers_to_capture(thinker, [1, 20, 38])
    assert captured == [[1, 20, 38]]


def test_config_names_the_thinker_architecture_not_the_checkpoint_wrapper() -> None:
    config = Ming3AudioConfig(
        text_config={"hidden_size": 2560, "num_hidden_layers": 42},
        architectures=["BailingMoeV3AudioForConditionalGeneration"],
    )
    assert config.architectures == ["BailingMoeV3ForCausalLM"]
