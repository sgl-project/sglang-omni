import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    get_server_return_hidden_states_mode,
)
from sglang.srt.runtime_context import get_context, publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import hf_transformers_utils

from sglang_omni.models.minicpm_o import bootstrap as minicpm_bootstrap
from sglang_omni.models.minicpm_o import request_builders
from sglang_omni.scheduling import bootstrap
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.vendor.sglang.server_args import override_server_args


@pytest.fixture
def dependencies(monkeypatch):
    model = SimpleNamespace(
        thinker=SimpleNamespace(model=SimpleNamespace(embed_tokens=object())),
        num_audio_tokens=1026,
        codec_eos_id=1025,
    )
    worker = SimpleNamespace(
        gpu_id=0,
        model_runner=SimpleNamespace(model=model, sampler=object()),
    )
    state = SimpleNamespace(worker=worker, events=[], fail_at=None)
    infrastructure_signature = inspect.signature(bootstrap.create_sglang_infrastructure)
    scheduler_signature = inspect.signature(OmniScheduler.__init__)

    def init_graphs(model_worker):
        assert model_worker is worker
        state.events.append(
            (
                "graphs",
                get_context().config_leaf("disable_cuda_graph"),
                get_server_return_hidden_states_mode(),
            )
        )
        if state.fail_at == "graphs":
            raise RuntimeError("graph initialization failed")

    def create_infrastructure(server_args, gpu_id, **kwargs):
        infrastructure_signature.bind(server_args, gpu_id, **kwargs)
        state.infrastructure_kwargs = kwargs
        if state.fail_at == "before_publish":
            raise RuntimeError("initialization failed before publish")
        publish(server_args, role="scheduler")
        state.events.append(("publish", get_server_return_hidden_states_mode()))
        if state.fail_at == "after_publish":
            raise RuntimeError("initialization failed after publish")
        if not kwargs.get("defer_cuda_graph_capture"):
            init_graphs(worker)
        return (
            worker,
            "tree",
            "req_pool",
            "kv_pool",
            SimpleNamespace(model_path="dummy", vocab_size=10000),
        )

    def init_scheduler(scheduler, **kwargs):
        scheduler_signature.bind(scheduler, **kwargs)
        scheduler.kwargs = kwargs

    monkeypatch.setattr(
        bootstrap, "create_sglang_infrastructure", create_infrastructure
    )
    monkeypatch.setattr(bootstrap, "init_sglang_cuda_graphs", init_graphs)
    monkeypatch.setattr(OmniScheduler, "__init__", init_scheduler)
    monkeypatch.setattr(
        hf_transformers_utils,
        "get_tokenizer",
        Mock(return_value=SimpleNamespace(convert_tokens_to_ids=lambda token: 1)),
    )
    for stage in ("thinker", "talker"):
        monkeypatch.setattr(
            request_builders,
            f"make_{stage}_scheduler_adapters",
            Mock(return_value=(object(), object())),
        )
    with get_context().override_server_args():
        state.server_args = ServerArgs(model_path="dummy")
        state.server_args.resolve_once()
        yield state


@pytest.mark.parametrize("speech_enabled", [False, True])
@pytest.mark.parametrize("disable_cuda_graph", [False, True])
@pytest.mark.parametrize("hidden_mode", [None, "last", "full"])
def test_thinker_bootstrap_capture_and_scheduler_contract(
    dependencies, speech_enabled, disable_cuda_graph, hidden_mode
):
    state = dependencies
    override_server_args(
        state.server_args,
        "test",
        disable_cuda_graph=disable_cuda_graph,
        enable_return_hidden_states=False,
        return_hidden_states_mode=hidden_mode,
    )
    configured_mode = {
        None: CaptureHiddenMode.NULL,
        "last": CaptureHiddenMode.LAST,
        "full": CaptureHiddenMode.FULL,
    }[hidden_mode]
    deferred = speech_enabled and not disable_cuda_graph
    captured_mode = CaptureHiddenMode.FULL if deferred else configured_mode

    scheduler = minicpm_bootstrap.create_thinker_scheduler(
        state.server_args, speech_enabled=speech_enabled
    )

    assert state.infrastructure_kwargs["defer_cuda_graph_capture"] == deferred
    assert state.events == [
        ("publish", captured_mode),
        ("graphs", disable_cuda_graph, captured_mode),
    ]
    assert get_server_return_hidden_states_mode() == configured_mode
    assert get_context().config_leaf("enable_return_hidden_states") is False
    runner = scheduler.kwargs["model_runner"]
    assert runner._capture_hidden_mode == (
        CaptureHiddenMode.FULL if speech_enabled else configured_mode
    )
    assert scheduler.kwargs["abort_callback"] == runner.reset_request
    assert scheduler.kwargs["model_config"].vocab_size == 10000


@pytest.mark.parametrize("disable_cuda_graph", [False, True])
def test_talker_bootstrap_capture_and_scheduler_contract(
    dependencies, disable_cuda_graph
):
    state = dependencies
    override_server_args(
        state.server_args,
        "test",
        disable_cuda_graph=disable_cuda_graph,
        enable_return_hidden_states=False,
        return_hidden_states_mode=None,
    )

    scheduler = minicpm_bootstrap.create_talker_scheduler(state.server_args)

    assert state.infrastructure_kwargs["defer_cuda_graph_capture"] != disable_cuda_graph
    assert state.events == [
        ("publish", CaptureHiddenMode.NULL),
        ("graphs", disable_cuda_graph, CaptureHiddenMode.NULL),
    ]
    assert get_context().config_leaf("disable_radix_cache") is True
    assert get_context().config_leaf("chunked_prefill_size") == 0
    assert scheduler.kwargs["model_config"].vocab_size == 1026
    assert state.worker.model_runner.model._sampler is state.worker.model_runner.sampler


@pytest.mark.parametrize("fail_at", ["before_publish", "after_publish", "graphs"])
def test_thinker_restores_hidden_configuration_after_failure(dependencies, fail_at):
    state = dependencies
    state.fail_at = fail_at
    override_server_args(
        state.server_args,
        "test",
        disable_cuda_graph=False,
        enable_return_hidden_states=False,
        return_hidden_states_mode="last",
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        minicpm_bootstrap.create_thinker_scheduler(
            state.server_args, speech_enabled=True
        )

    if fail_at == "before_publish":
        cfg = resolved_view(state.server_args)
        assert cfg.enable_return_hidden_states is False
        assert cfg.return_hidden_states_mode == "last"
        assert cfg.disable_cuda_graph is False
    else:
        assert get_server_return_hidden_states_mode() == CaptureHiddenMode.LAST
        assert get_context().config_leaf("enable_return_hidden_states") is False
        assert get_context().config_leaf("disable_cuda_graph") is False
