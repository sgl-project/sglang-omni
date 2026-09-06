# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.unit_test.fakes import FakeServerArgs

AUTOTUNE_MODULE = "sglang_omni.model_runner.moe_prefill_autotune"


def _autotune_module():
    assert importlib.util.find_spec(AUTOTUNE_MODULE) is not None
    return importlib.import_module(AUTOTUNE_MODULE)


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    **attrs: object,
) -> ModuleType:
    module = ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_minimal_autotune_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extend_mode: object,
) -> None:
    _install_fake_module(
        monkeypatch,
        "sglang.srt.model_executor.forward_batch_info",
        ForwardMode=SimpleNamespace(EXTEND=extend_mode),
    )

    @contextlib.contextmanager
    def flashinfer_autotune_context(model_runner: object, *, skip_logits: bool):
        del model_runner
        assert skip_logits is True
        yield

    _install_fake_module(
        monkeypatch,
        "sglang.srt.model_executor.runner.flashinfer_autotune",
        should_run_flashinfer_autotune=lambda model_runner: True,
        flashinfer_autotune_context=flashinfer_autotune_context,
    )
    _install_fake_module(
        monkeypatch, "sglang.srt.layers.cp.padding", get_cp_padding_align_size=lambda: 1
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.runtime_context",
        get_parallel=lambda: SimpleNamespace(attn_tp_size=1),
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.utils.common",
        ceil_align=lambda value, align: ((value + align - 1) // align) * align,
        require_mlp_sync=lambda server_args: False,
    )


class _TalkerFeedbackBackbone:
    """Executable reproduction of the talker codec-feedback input branch."""

    def __init__(self, *, max_running_requests: int, hidden_size: int) -> None:
        self._cp_enabled = True
        self._feedback_mask = torch.ones(max_running_requests, dtype=torch.bool)
        self._feedback_buffer = torch.full(
            (max_running_requests, hidden_size), 2.0, dtype=torch.float32
        )
        self.hidden_size = hidden_size

    def synthetic_extend(self, num_tokens: int) -> torch.Tensor:
        hidden_states = torch.ones(num_tokens, self.hidden_size)
        if self._cp_enabled:
            batch_size = hidden_states.shape[0]
            feedback_mask = self._feedback_mask[:batch_size]
            hidden_states = torch.where(
                feedback_mask.unsqueeze(-1),
                self._feedback_buffer[:batch_size],
                hidden_states,
            )
            self._feedback_mask[:batch_size] = False
        return hidden_states


def test_moe_autotune_is_inert_when_talker_flag_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autotune = _autotune_module()
    monkeypatch.delenv(autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV, raising=False)

    class UntouchableRunner:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(
                f"disabled autotune touched model runner attribute {name}"
            )

    autotune.maybe_autotune_prefill_moe(
        UntouchableRunner(),
        env_var=autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("", 0, id="empty_is_disabled"),
        pytest.param("0", 0, id="zero_is_disabled"),
        pytest.param(" 17 ", 17, id="integer_is_parsed"),
        pytest.param("not-an-int", 0, id="invalid_is_disabled"),
    ],
)
def test_moe_autotune_talker_flag_parsing(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: int,
) -> None:
    autotune = _autotune_module()
    monkeypatch.setenv(autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV, raw)

    assert (
        autotune.moe_autotune_prefill_tokens(autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV)
        == expected
    )


def test_moe_autotune_opt_in_runs_one_aligned_extend_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autotune = _autotune_module()
    monkeypatch.setenv(autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV, "5")
    events: list[object] = []
    extend_mode = object()

    _install_fake_module(
        monkeypatch,
        "sglang.srt.model_executor.forward_batch_info",
        ForwardMode=SimpleNamespace(EXTEND=extend_mode),
    )

    @contextlib.contextmanager
    def flashinfer_autotune_context(model_runner: object, *, skip_logits: bool):
        events.append(("autotune_enter", model_runner, skip_logits))
        yield
        events.append("autotune_exit")

    _install_fake_module(
        monkeypatch,
        "sglang.srt.model_executor.runner.flashinfer_autotune",
        should_run_flashinfer_autotune=lambda model_runner: (
            events.append(("applicable", model_runner)) or True
        ),
        flashinfer_autotune_context=flashinfer_autotune_context,
    )
    _install_fake_module(
        monkeypatch, "sglang.srt.layers.cp.padding", get_cp_padding_align_size=lambda: 4
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.runtime_context",
        get_parallel=lambda: SimpleNamespace(attn_tp_size=2),
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.utils.common",
        ceil_align=lambda value, align: ((value + align - 1) // align) * align,
        require_mlp_sync=lambda server_args: True,
    )

    buffers = object()

    @contextlib.contextmanager
    def canary_context(index: int):
        events.append(("canary_enter", index))
        yield
        events.append(("canary_exit", index))

    eager_runner = SimpleNamespace()

    def alloc_dummy_decode_buffers(tokens: int) -> object:
        events.append(("alloc", tokens))
        return buffers

    def dummy_run(**kwargs: object) -> None:
        with kwargs["run_ctx"]:
            events.append(("dummy", kwargs))

    eager_runner._alloc_dummy_decode_buffers = alloc_dummy_decode_buffers
    eager_runner._dummy_run = dummy_run
    model_runner = SimpleNamespace(
        eager_runner=eager_runner,
        server_args=object(),
        ps=SimpleNamespace(tp_rank=3),
        canary_manager=SimpleNamespace(
            with_active_single_forward_manager=canary_context
        ),
    )

    autotune.maybe_autotune_prefill_moe(
        model_runner,
        env_var=autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV,
    )

    assert events[0] == ("applicable", model_runner)
    assert events[1] == ("alloc", 8)
    assert events[2] == ("autotune_enter", model_runner, True)
    assert events[3] == ("canary_enter", 0)
    assert events[4][0] == "dummy"
    dummy_kwargs = events[4][1]
    assert dummy_kwargs["batch_size"] == 8
    assert dummy_kwargs["forward_mode_override"] is extend_mode
    assert dummy_kwargs["buffers"] is buffers
    assert events[5:] == [("canary_exit", 0), "autotune_exit"]


def test_moe_autotune_talker_oversized_extend_suspends_decode_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autotune = _autotune_module()
    monkeypatch.setenv(autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV, "2048")
    extend_mode = object()
    _install_minimal_autotune_runtime(monkeypatch, extend_mode=extend_mode)

    backbone = _TalkerFeedbackBackbone(max_running_requests=64, hidden_size=4)
    original_mask = backbone._feedback_mask.clone()
    enabled_during_dummy: list[bool] = []
    buffers = object()

    def dummy_run(**kwargs: object) -> None:
        assert kwargs["batch_size"] == 2048
        assert kwargs["forward_mode_override"] is extend_mode
        assert kwargs["buffers"] is buffers
        enabled_during_dummy.append(backbone._cp_enabled)
        hidden_states = backbone.synthetic_extend(2048)
        assert hidden_states.shape == (2048, 4)

    eager_runner = SimpleNamespace(
        _alloc_dummy_decode_buffers=lambda tokens: buffers,
        _dummy_run=dummy_run,
    )
    model_runner = SimpleNamespace(
        eager_runner=eager_runner,
        model=SimpleNamespace(model=backbone, _cp_enabled=True),
        server_args=object(),
        ps=SimpleNamespace(tp_rank=0),
        canary_manager=None,
    )

    autotune.maybe_autotune_prefill_moe(
        model_runner,
        env_var=autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV,
    )

    assert enabled_during_dummy == [False]
    assert backbone._cp_enabled is True
    assert torch.equal(backbone._feedback_mask, original_mask)


def test_moe_autotune_talker_restores_decode_feedback_after_dummy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autotune = _autotune_module()
    monkeypatch.setenv(autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV, "2048")
    _install_minimal_autotune_runtime(monkeypatch, extend_mode=object())

    class DummyFailure(RuntimeError):
        pass

    backbone = _TalkerFeedbackBackbone(max_running_requests=64, hidden_size=4)

    def dummy_run(**kwargs: object) -> None:
        del kwargs
        assert backbone._cp_enabled is False
        raise DummyFailure("synthetic forward failed")

    model_runner = SimpleNamespace(
        eager_runner=SimpleNamespace(
            _alloc_dummy_decode_buffers=lambda tokens: object(),
            _dummy_run=dummy_run,
        ),
        model=SimpleNamespace(model=backbone, _cp_enabled=True),
        server_args=object(),
        ps=SimpleNamespace(tp_rank=0),
        canary_manager=None,
    )

    with pytest.raises(DummyFailure, match="synthetic forward failed"):
        autotune.maybe_autotune_prefill_moe(
            model_runner,
            env_var=autotune.MOE_AUTOTUNE_TALKER_TOKENS_ENV,
        )

    assert backbone._cp_enabled is True


def test_talker_bootstrap_propagates_autotune_flag_after_graph_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni import bootstrap

    class AutotuneReached(Exception):
        pass

    events: list[str] = []
    talker_env = "SGLANG_OMNI_MOE_AUTOTUNE_TALKER_TOKENS"

    _install_fake_module(
        monkeypatch,
        "sglang.srt.utils.hf_transformers_utils",
        get_tokenizer=object,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.request_builders",
        make_talker_scheduler_adapters=object,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.talker_model_runner",
        QwenTalkerModelRunner=object,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.models.qwen3_omni.talker_scheduler",
        QwenTalkerScheduler=object,
        configure_talker_server_args=lambda *args, **kwargs: True,
    )

    model_config = SimpleNamespace(
        vocab_size=100,
        hf_config=SimpleNamespace(
            talker_config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=4096))
        ),
    )
    raw_model_runner = SimpleNamespace(
        model_config=model_config,
        model=SimpleNamespace(_sampler=None),
        sampler=object(),
    )
    model_worker = SimpleNamespace(model_runner=raw_model_runner)

    def create_sglang_infrastructure(*args: object, **kwargs: object):
        events.append("infrastructure")
        return (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            model_config,
        )

    def init_sglang_cuda_graphs(worker: object) -> None:
        assert worker is model_worker
        events.append("graph_init")

    _install_fake_module(
        monkeypatch,
        "sglang_omni.scheduling.bootstrap",
        create_sglang_infrastructure=create_sglang_infrastructure,
        init_sglang_cuda_graphs=init_sglang_cuda_graphs,
    )
    _install_fake_module(
        monkeypatch,
        "sglang_omni.scheduling.sglang_backend",
        SGLangOutputProcessor=object,
    )

    def maybe_autotune_prefill_moe(runner: object, env_var: str) -> None:
        assert runner is raw_model_runner
        assert env_var == talker_env
        events.append("autotune")
        raise AutotuneReached

    _install_fake_module(
        monkeypatch,
        AUTOTUNE_MODULE,
        MOE_AUTOTUNE_TALKER_TOKENS_ENV=talker_env,
        maybe_autotune_prefill_moe=maybe_autotune_prefill_moe,
    )

    server_args = FakeServerArgs(disable_cuda_graph=True)
    with pytest.raises(AutotuneReached):
        bootstrap.create_talker_scheduler(server_args)

    assert events == ["infrastructure", "graph_init", "autotune"]
