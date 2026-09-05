# SPDX-License-Identifier: Apache-2.0
"""DFlash admission and bootstrap contracts, without allocating model/GPU state."""

from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen3_omni import bootstrap, request_builders
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.mark.parametrize(
    "algorithm, options, error",
    [
        ("EAGLE3", {}, "only DFLASH"),
        ("DFLASH", {"speech_enabled": False}, "requires speech_enabled"),
        ("DFLASH", {"talker_stream_token_only": False}, "requires speech_enabled"),
        ("DFLASH", {"capture_speech_hidden_states": True}, "requires speech_enabled"),
    ],
)
def test_unsupported_speculation_fails_before_model_initialization(
    algorithm, options, error
):
    args = SimpleNamespace(
        speculative_algorithm=algorithm, speculative_draft_model_path="draft"
    )
    values = dict(
        speech_enabled=True,
        talker_stream_token_only=True,
        capture_speech_hidden_states=False,
    )
    values.update(options)
    with pytest.raises(ValueError, match=error):
        bootstrap._configure_thinker_speculation(args, **values)


@pytest.mark.parametrize("enable_prefill_graphs", [False, True])
def test_dflash_bootstrap_selects_native_worker_and_disables_lookahead(
    monkeypatch, enable_prefill_graphs
):
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.utils import hf_transformers_utils

    from sglang_omni.model_runner import thinker_model_runner
    from sglang_omni.models.qwen3_omni import thinker_model_runner as qwen_runner
    from sglang_omni.scheduling import bootstrap as infrastructure
    from sglang_omni.scheduling import omni_scheduler, sglang_backend
    from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
    from sglang_omni.vendor.sglang import server_args as server_args_module

    args = SimpleNamespace(
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="draft",
        disable_cuda_graph=not enable_prefill_graphs,
        enable_return_hidden_states=False,
        disable_overlap_schedule=False,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(
                backend=(
                    CudaGraphBackend.BREAKABLE
                    if enable_prefill_graphs
                    else CudaGraphBackend.DISABLED
                ),
                bs=[32, 64],
            )
        ),
    )
    config = SimpleNamespace(
        model_path="target",
        vocab_size=100,
        hf_config=SimpleNamespace(thinker_config=object()),
    )
    prefill_runner = object.__new__(PrefillCudaGraphRunner)
    prefill_runner.prefill_backend_name = args.cuda_graph_config.prefill.backend
    prefill_runner.capture_num_tokens = [32, 64]
    prefill_runner.buffer_registry = SimpleNamespace(has_slot=lambda name: False)
    target = SimpleNamespace(
        model_runner=SimpleNamespace(prefill_cuda_graph_runner=prefill_runner)
    )
    draft = object()
    calls = []

    def create(*unused, **kwargs):
        calls.append(kwargs)
        return target, object(), object(), object(), config, draft

    monkeypatch.setattr(
        infrastructure,
        "create_sglang_infrastructure",
        create,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("DFlash must bypass scalar model runners")

    monkeypatch.setattr(thinker_model_runner, "ThinkerModelRunner", forbidden)
    monkeypatch.setattr(qwen_runner, "Qwen3OmniThinkerModelRunner", forbidden)
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", forbidden)
    monkeypatch.setattr(
        hf_transformers_utils, "get_tokenizer", lambda *a, **kw: object()
    )
    monkeypatch.setattr(omni_scheduler, "OmniScheduler", SimpleNamespace)
    monkeypatch.setattr(
        server_args_module,
        "override_server_args",
        lambda obj, reason, **kwargs: [
            setattr(obj, key, value) for key, value in kwargs.items()
        ],
    )
    adapters = []
    monkeypatch.setattr(
        request_builders,
        "make_thinker_scheduler_adapters",
        lambda **kw: adapters.append(kw) or (object(), object()),
    )
    scheduler = bootstrap.create_thinker_scheduler(
        args,
        speech_enabled=True,
        talker_stream_token_only=True,
        capture_speech_hidden_states=False,
        enable_async_decode=True,
    )
    assert scheduler.tp_worker is target and scheduler.speculative_worker is draft
    assert scheduler.model_runner is None and scheduler.enable_async_decode is False
    assert args.disable_overlap_schedule is True
    assert args.enable_multimodal is False
    assert calls[0]["capture_hidden_layers"] is None
    assert calls[0]["speculative"] is True
    assert calls[0]["enable_prefill_input_embeds"] is False
    assert adapters[0]["require_dflash_compatible_request"] is True
    assert adapters[0]["require_token_only_talker_input"] is True


@pytest.mark.parametrize(
    "params, model_inputs, capture_keys, error",
    [
        ({"temperature": 0.7}, {}, (), "greedy"),
        ({"return_logprob": True}, {}, (), "logprobs"),
        ({"logprobs": True}, {}, (), "logprobs"),
        ({"return_hidden_states": True}, {}, (), "hidden"),
        ({}, {"audio_embeds": "audio"}, (), "text-input"),
        ({}, {}, ("hidden_states",), "text-input"),
    ],
)
def test_dflash_rejects_unsupported_request_before_build(
    monkeypatch, params, model_inputs, capture_keys, error
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "unsupported request must fail before allocation/admission"
        )

    monkeypatch.setattr(request_builders, "build_sglang_thinker_request", forbidden)
    build, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=100,
        require_dflash_compatible_request=True,
    )
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(
            inputs=[], params=params, metadata={"output_modalities": ["text"]}
        ),
        data={
            "thinker_inputs": {
                "model_inputs": model_inputs,
                "capture_model_output_keys": capture_keys,
            }
        },
    )
    with pytest.raises(ValueError, match=error):
        build(payload)


def test_dflash_admits_greedy_text_and_preserves_request_parameters(monkeypatch):
    data = SimpleNamespace(stage_payload=None)
    seen = []
    monkeypatch.setattr(
        request_builders,
        "build_sglang_thinker_request",
        lambda state, **kw: seen.append(kw) or data,
    )
    build, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=100,
        require_dflash_compatible_request=True,
    )
    params = {"temperature": 0, "max_new_tokens": 64, "stop_token_ids": [7]}
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs=[], params=params),
        data={"thinker_inputs": {"model_inputs": {}}},
    )
    assert build(payload) is data
    assert seen[0]["params"] == params and data.stage_payload is payload


def test_native_prefill_attestation_keeps_backend_and_bucket_checks():
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
    from sglang_omni.utils.cuda_graph_batch_validator import attest_prefill_cuda_graphs

    runner = object.__new__(PrefillCudaGraphRunner)
    runner.prefill_backend_name = CudaGraphBackend.BREAKABLE
    runner.capture_num_tokens = [32, 64]
    runner.buffer_registry = SimpleNamespace(has_slot=lambda name: False)
    model_runner = SimpleNamespace(prefill_cuda_graph_runner=runner)
    args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=CudaGraphBackend.BREAKABLE, bs=[32, 64])
        )
    )
    with pytest.raises(RuntimeError, match="no input_embeds slot"):
        attest_prefill_cuda_graphs(model_runner, args)
    attest_prefill_cuda_graphs(model_runner, args, require_input_embeds=False)

    runner.prefill_backend_name = CudaGraphBackend.DISABLED
    with pytest.raises(RuntimeError, match="backend mismatch"):
        attest_prefill_cuda_graphs(model_runner, args, require_input_embeds=False)
    runner.prefill_backend_name = CudaGraphBackend.BREAKABLE
    runner.capture_num_tokens = [32]
    with pytest.raises(RuntimeError, match="capture shapes differ"):
        attest_prefill_cuda_graphs(model_runner, args, require_input_embeds=False)
