# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.fun_asr.stages as fun_asr_stages
from sglang_omni.models.fun_asr.sglang_model import (
    FunAsrNanoAdaptor,
    FunAsrNanoAudioEncoder,
)


def _tiny_model() -> SimpleNamespace:
    encoder = FunAsrNanoAudioEncoder(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
    )
    projector = FunAsrNanoAdaptor(
        encoder_dim=8,
        llm_dim=8,
        ffn_dim=16,
        num_layers=1,
        attention_heads=2,
    )
    return SimpleNamespace(
        audio_tower=encoder,
        multi_modal_projector=projector,
        config=SimpleNamespace(encoder_config=SimpleNamespace(input_size=8)),
    )


def _stub_torch_compile_config(monkeypatch) -> None:
    # The helper calls sglang's set_torch_compile_config, which mutates global
    # dynamo/inductor config; keep unit tests side-effect free.
    import sglang.srt.model_executor.cuda_graph_runner as cuda_graph_runner

    monkeypatch.setattr(cuda_graph_runner, "set_torch_compile_config", lambda: None)


def test_compile_fun_asr_audio_encoder_compiles_forwards_with_dynamic_shapes(
    monkeypatch,
) -> None:
    _stub_torch_compile_config(monkeypatch)
    model = _tiny_model()
    original_tower_forward = model.audio_tower.forward
    original_projector_forward = model.multi_modal_projector.forward
    tower_param_names = set(dict(model.audio_tower.named_parameters()))

    compile_calls = []
    forward_shapes = []

    def _fake_compile(fn, dynamic=None):
        compile_calls.append({"fn": fn, "dynamic": dynamic})

        def _wrapped(xs):
            forward_shapes.append(tuple(xs.shape))
            return fn(xs)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    fun_asr_stages._compile_fun_asr_audio_encoder(model, warmup_lfr_frames=16)

    assert [call["dynamic"] for call in compile_calls] == [True, True]
    assert compile_calls[0]["fn"] == original_tower_forward
    assert compile_calls[1]["fn"] == original_projector_forward
    # Bound-method compile must leave the module tree and parameter names
    # intact — load_weights and weight updates match checkpoint names against
    # named_parameters, so no _orig_mod prefixes may appear.
    assert set(dict(model.audio_tower.named_parameters())) == tower_param_names
    # Warmup ran through both compiled forwards: encoder input [1, T, 560-dim
    # equivalent], then the encoder's output into the projector.
    assert forward_shapes == [(1, 16, 8), (1, 16, 8)]


def test_compile_fun_asr_audio_encoder_warmup_matches_service_grad_mode(
    monkeypatch,
) -> None:
    # The pre-LM encoder service encodes under torch.inference_mode, and
    # Dynamo guards on the input's dispatch key set. Inference-mode tensors
    # lack ADInplaceOrView, so the warmup tensor must be *allocated* inside
    # the context — merely calling inside it leaves a normal tensor whose
    # graph the service's tensors fail, costing a ~20 s recompile on the
    # first real request.
    _stub_torch_compile_config(monkeypatch)
    model = _tiny_model()
    modes = []

    def _fake_compile(fn, dynamic=None):
        def _wrapped(xs):
            modes.append((torch.is_inference_mode_enabled(), torch.is_inference(xs)))
            return fn(xs)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    fun_asr_stages._compile_fun_asr_audio_encoder(model, warmup_lfr_frames=16)
    assert modes == [(True, True), (True, True)]

    modes.clear()
    model = _tiny_model()
    fun_asr_stages._compile_fun_asr_audio_encoder(
        model, warmup_lfr_frames=16, warmup_inference_mode=False
    )
    assert modes == [(False, False), (False, False)]


def test_compile_fun_asr_audio_encoder_rejects_degenerate_warmup_length(
    monkeypatch,
) -> None:
    _stub_torch_compile_config(monkeypatch)
    model = _tiny_model()

    def _fail_compile(fn, dynamic=None):
        raise AssertionError("torch.compile must not run for invalid warmup")

    monkeypatch.setattr(torch, "compile", _fail_compile)

    with pytest.raises(ValueError, match="warmup_lfr_frames"):
        fun_asr_stages._compile_fun_asr_audio_encoder(model, warmup_lfr_frames=1)
