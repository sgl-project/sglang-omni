# SPDX-License-Identifier: Apache-2.0
"""Public talker contract: condition embeddings and prefill CUDA graph wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sglang.srt.model_loader.utils import resolve_language_model
from torch import nn

from sglang_omni.models.minicpm_o import stages
from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
    MiniCPMTTSProjector,
)
from sglang_omni.platforms import current_platform

HIDDEN = 8
LLM_DIM = 16
NUM_TEXT = 20
TEXT_EOS = 5
AUDIO_BOS = 6


def bare_model() -> MiniCPMOTalkerForCausalLM:
    model = object.__new__(MiniCPMOTalkerForCausalLM)
    nn.Module.__init__(model)
    model.text_eos_token_id = TEXT_EOS
    model.audio_bos_token_id = AUDIO_BOS
    model.normalize_projected_hidden = True
    model.emb_text = nn.Embedding(NUM_TEXT, HIDDEN)
    model.projector_semantic = MiniCPMTTSProjector(LLM_DIM, HIDDEN)
    return model


def test_condition_matches_reference_math():
    model = bare_model()
    tokens = torch.tensor([3, 7, 1], dtype=torch.long)
    hidden = torch.randn(3, LLM_DIM)

    condition = model.build_condition_embeddings(tokens, hidden)

    # reference: emb_text(t) + l2norm(projector(h)), then [text_eos, audio_bos]
    ref = model.emb_text(tokens) + F.normalize(
        model.projector_semantic(hidden), p=2, dim=-1
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, torch.cat([ref, boundary], dim=0))
    assert condition.shape == (5, HIDDEN)


def test_condition_empty_span_is_boundary_only():
    model = bare_model()
    condition = model.build_condition_embeddings(
        torch.empty(0, dtype=torch.long), torch.empty(0, LLM_DIM)
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, boundary)


def test_condition_length_mismatch_raises():
    model = bare_model()
    with pytest.raises(ValueError, match="length mismatch"):
        model.build_condition_embeddings(torch.tensor([1, 2]), torch.randn(3, LLM_DIM))


def test_prefill_graphs_resolve_the_talker_decoder():
    model = bare_model()
    model.llama = SimpleNamespace(model=nn.Identity())
    assert resolve_language_model(model) is model.llama.model


@pytest.mark.parametrize(
    ("server_args_overrides", "is_cuda", "backend", "operator_selected"),
    [
        ({}, True, "breakable", False),
        ({"cuda_graph_backend_prefill": "disabled"}, True, "disabled", True),
        ({}, False, "disabled", False),
    ],
)
def test_talker_stage_defaults_breakable_prefill_graphs_on_nvidia(
    monkeypatch: pytest.MonkeyPatch,
    server_args_overrides: dict[str, object],
    is_cuda: bool,
    backend: str,
    operator_selected: bool,
) -> None:
    built: dict[str, object] = {}
    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(current_platform, "enable_talker_graph", lambda: True)
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "register_minicpm_o_hf_config", lambda: None)
    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        lambda model_path, context_length, **overrides: built.update(overrides),
    )
    monkeypatch.setattr(
        stages,
        "resolved_view",
        lambda server_args: SimpleNamespace(
            mem_fraction_static=0.5, max_running_requests=32, max_total_tokens=None
        ),
    )
    monkeypatch.setattr(stages, "validate_generation_batch_policy", lambda **_: None)
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda gpu_id: 0)
    monkeypatch.setattr(
        stages,
        "create_talker_scheduler",
        lambda server_args, gpu_id, **kwargs: built.update(scheduler=kwargs),
    )

    stages.create_sglang_talker_executor_from_config(
        "model", server_args_overrides=server_args_overrides
    )

    assert built["cuda_graph_backend_prefill"] == backend
    assert (
        max(built["cuda_graph_bs_prefill"])
        == stages.TALKER_PREFILL_CUDA_GRAPH_MAX_TOKENS
    )
    assert built["scheduler"]["operator_selected_prefill_backend"] is operator_selected
