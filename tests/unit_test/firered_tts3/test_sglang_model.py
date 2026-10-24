# SPDX-License-Identifier: Apache-2.0
"""Backbone feedback, text logits, and checkpoint loading contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

import sglang_omni.models.firered_tts3.engine_builder as engine_builder_module
from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel

HIDDEN_SIZE = 8
BATCH_SIZE = 3


def build_model(
    *, use_lm_head: bool, with_buffer: bool
) -> tuple[FireRedTTS3SGLangModel, dict[str, torch.Tensor | bool]]:
    """A FireRedTTS3SGLangModel shell with a recording stub backbone."""
    model = FireRedTTS3SGLangModel.__new__(FireRedTTS3SGLangModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(use_lm_head=use_lm_head)
    model.llm_hidden_size = HIDDEN_SIZE
    recorded: dict[str, torch.Tensor | bool] = {}

    def backbone_model(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input_embeds is not None
        recorded["input_embeds"] = input_embeds
        return input_embeds

    real_logits = torch.full((BATCH_SIZE, 151936), 7.0)

    def logits_processor(
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: torch.nn.Module,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        recorded["logits_processor_called"] = True
        return LogitsProcessorOutput(
            next_token_logits=real_logits, hidden_states=hidden_states
        )

    model.backbone = SimpleNamespace(
        model=backbone_model,
        logits_processor=logits_processor,
        lm_head=torch.nn.Identity(),
    )
    model.graph_feedback_buffer = (
        torch.arange(BATCH_SIZE * HIDDEN_SIZE, dtype=torch.bfloat16).reshape(
            BATCH_SIZE, HIDDEN_SIZE
        )
        if with_buffer
        else None
    )
    model.text_logits_requested = False
    return model, recorded


def forward_batch_stub(*, decode: bool) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode, is_extend=lambda: not decode
        ),
        extend_seq_lens=torch.full((BATCH_SIZE,), 5) if not decode else None,
        input_embeds=torch.full((BATCH_SIZE, HIDDEN_SIZE), -1.0),
    )


@pytest.mark.parametrize(
    "use_lm_head,with_buffer,decode,requested_flags,logit_widths",
    [
        (False, True, True, [False], [1]),
        (False, False, True, [False], [1]),
        (True, True, True, [False], [151936]),
        (True, False, True, [False, True], [1, 151936]),
        (True, True, False, [False, True], [1, 151936]),
    ],
)
def test_forward_preserves_feedback_and_text_logits(
    use_lm_head: bool,
    with_buffer: bool,
    decode: bool,
    requested_flags: list[bool],
    logit_widths: list[int],
) -> None:
    """Captured decode must expose text logits; eager execution must honor each step's sampling flag."""
    model, recorded = build_model(use_lm_head=use_lm_head, with_buffer=with_buffer)
    batch = forward_batch_stub(decode=decode)
    for requested, width in zip(requested_flags, logit_widths, strict=True):
        recorded.clear()
        model.text_logits_requested = requested
        output = model.forward(
            torch.zeros(BATCH_SIZE, dtype=torch.long), torch.arange(BATCH_SIZE), batch
        )
        if with_buffer and decode:
            assert torch.equal(
                output.hidden_states, model.graph_feedback_buffer[:BATCH_SIZE]
            )
        else:
            assert output.hidden_states is batch.input_embeds
        assert recorded.get("logits_processor_called", False) is (width > 1)
        assert output.next_token_logits.shape == (BATCH_SIZE, width)


@pytest.fixture
def engine_builder() -> FireRedTTS3EngineBuilder:
    pipeline = FireRedTTS3PipelineConfig(model_path="model")
    return FireRedTTS3EngineBuilder(
        max_running_requests=pipeline.generation_admission_defaults()[
            "max_running_requests"
        ],
        **pipeline.stage_factory_kwargs("talker"),
    )


def test_generation_defaults_turn_the_decode_graph_on(
    engine_builder: FireRedTTS3EngineBuilder,
) -> None:
    builder = engine_builder
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is False
    # adjust_overrides runs on the merged dict (defaults + yaml overrides).
    overrides = dict(defaults)
    builder.adjust_overrides(overrides)
    # The graph is captured with FULL hidden states; the runner requests the
    # matching mode per step.
    assert overrides["enable_return_hidden_states"] is True
    eager_overrides = dict(defaults, disable_cuda_graph=True)
    builder.adjust_overrides(eager_overrides)
    assert "enable_return_hidden_states" not in eager_overrides


@pytest.fixture
def loadable_model() -> FireRedTTS3SGLangModel:
    class Backbone(torch.nn.Linear):
        def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
            with torch.no_grad():
                for name, tensor in weights:
                    self.get_parameter(name).copy_(tensor)

    model = FireRedTTS3SGLangModel.__new__(FireRedTTS3SGLangModel)
    torch.nn.Module.__init__(model)
    model.backbone = Backbone(2, 2, bias=False)
    model.weights_locked = False
    for name in ("patch_encoder", "dit", "dit_head", "stop_head"):
        setattr(model, name, torch.nn.Linear(2, 2, bias=False))
    model.spk_proj_llm = model.spk_proj_dit = None
    return model


def test_weight_loading_finishes_before_execution_is_locked(
    loadable_model: FireRedTTS3SGLangModel,
) -> None:
    model = loadable_model
    assert model.weight_update_blocked_reason is None
    initial = torch.ones_like(model.dit.weight)
    assert model.load_weights([("dit.weight", initial)]) == {"dit.weight"}
    assert torch.equal(model.dit.weight, initial)
    model.weights_locked = True
    updates = iter([("dit.weight", torch.zeros_like(initial))])
    with pytest.raises(RuntimeError, match="restart the worker"):
        model.load_weights(updates)
    assert next(updates)[0] == "dit.weight"
    assert torch.equal(model.dit.weight, initial)


@pytest.mark.parametrize("compile_fails", [False, True])
def test_compile_locks_updates_before_creating_execution(
    loadable_model: FireRedTTS3SGLangModel,
    engine_builder: FireRedTTS3EngineBuilder,
    monkeypatch: pytest.MonkeyPatch,
    compile_fails: bool,
) -> None:
    model = loadable_model
    builder = engine_builder

    def prepare(
        dit: torch.nn.Module, mode: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        assert dit is model.dit and mode == builder.flow_mode
        with pytest.raises(RuntimeError, match="restart the worker"):
            model.load_weights([])
        if compile_fails:
            raise RuntimeError("compile failed")
        else:
            return dit, dit

    monkeypatch.setattr(engine_builder_module, "prepare_flow_dit", prepare)
    if compile_fails:
        with pytest.raises(RuntimeError, match="compile failed"):
            builder.compile_model(model, None)
    else:
        builder.compile_model(model, None)
    assert model.weight_update_blocked_reason is not None


def test_disabled_flow_graph_does_not_lock_weight_loading(
    loadable_model: FireRedTTS3SGLangModel,
) -> None:
    pipeline = FireRedTTS3PipelineConfig(model_path="model")
    builder = FireRedTTS3EngineBuilder(
        max_running_requests=pipeline.generation_admission_defaults()[
            "max_running_requests"
        ],
        **{**pipeline.stage_factory_kwargs("talker"), "flow_cuda_graph": False},
    )
    builder.compile_model(loadable_model, None)
    assert loadable_model.weight_update_blocked_reason is None
    replacement = torch.zeros_like(loadable_model.dit.weight)
    loadable_model.load_weights([("dit.weight", replacement)])
    assert torch.equal(loadable_model.dit.weight, replacement)


@pytest.mark.parametrize(
    "backbone_prefix", ["backbone_llm.", "backbone_llm.model."], ids=["base", "chatml"]
)
def test_checkpoint_routing_loads_backbone_and_side_path(
    loadable_model: FireRedTTS3SGLangModel,
    backbone_prefix: Literal["backbone_llm.", "backbone_llm.model."],
) -> None:
    weights = [
        (f"{backbone_prefix}weight", torch.full((2, 2), 2.0)),
        ("dit.weight", torch.full((2, 2), 3.0, dtype=torch.bfloat16)),
        ("stop_head.weight", torch.full((2, 2), 4.0)),
        ("backbone_llm.lm_head.weight", torch.full((2, 2), -1.0)),
    ]
    loaded = loadable_model.load_weights(weights)
    assert loaded == {"dit.weight", "stop_head.weight"}
    inputs = torch.ones(1, 2)
    assert torch.equal(loadable_model.backbone(inputs), torch.full((1, 2), 4.0))
    assert torch.equal(loadable_model.dit(inputs), torch.full((1, 2), 6.0))
    assert torch.equal(loadable_model.stop_head(inputs), torch.full((1, 2), 8.0))
    assert loadable_model.dit.weight.dtype == torch.float32
    with pytest.raises(ValueError, match="unexpected FireRedTTS3 checkpoint key"):
        loadable_model.load_weights([("unknown.weight", torch.ones(2, 2))])
