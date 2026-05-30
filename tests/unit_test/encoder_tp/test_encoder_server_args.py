# SPDX-License-Identifier: Apache-2.0
"""Tests for ``build_sglang_encoder_server_args`` and the protected-keys reject.

We avoid actually constructing ``ServerArgs`` (which fetches the model
config from HF Hub) by monkey-patching it to a recording stand-in. The
helper must reject protected keys *before* delegating, so the stand-in
never sees a bad call.
"""
from __future__ import annotations

import pytest

from sglang_omni.scheduling.sglang_backend import (
    build_sglang_encoder_server_args,
    server_args_builder,
)


class _FakeServerArgs:
    """Records the kwargs the builder passed without doing HF lookup."""

    captured: dict | None = None

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _stub_server_args(monkeypatch):
    monkeypatch.setattr(server_args_builder, "ServerArgs", _FakeServerArgs)
    _FakeServerArgs.captured = None
    yield


# ---------------------------------------------------------------------------
# Happy path: helper produces ServerArgs kwargs with encoder-only fork on
# ---------------------------------------------------------------------------


def test_helper_sets_encoder_only_fork():
    args = build_sglang_encoder_server_args(
        model_path="dummy/model",
        tp_size=2,
        base_gpu_id=0,
        dist_init_addr="127.0.0.1:29500",
    )
    assert args.encoder_only is True
    assert args.language_only is False
    assert args.mm_enable_dp_encoder is False
    assert args.disable_cuda_graph is True
    assert args.tp_size == 2
    assert args.pp_size == 1
    assert args.base_gpu_id == 0
    assert args.dist_init_addr == "127.0.0.1:29500"
    assert args.trust_remote_code is True


def test_helper_passes_dtype_load_format():
    args = build_sglang_encoder_server_args(
        model_path="dummy/model",
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr="127.0.0.1:29500",
        dtype="float16",
        load_format="auto",
    )
    assert args.dtype == "float16"
    assert args.load_format == "auto"


def test_helper_forwards_loader_overrides():
    """Loader / processor knobs are NOT protected — pass through."""
    args = build_sglang_encoder_server_args(
        model_path="dummy/model",
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr="127.0.0.1:29500",
        model_loader_extra_config={"key": "value"},
        download_dir="/tmp/cache",
    )
    assert args.model_loader_extra_config == {"key": "value"}
    assert args.download_dir == "/tmp/cache"


# ---------------------------------------------------------------------------
# Protected-key rejects (helper level — go through `**overrides`)
# ---------------------------------------------------------------------------

# AR-only knobs that have no meaning for an encoder-only runner.
_AR_ONLY = [
    ("mem_fraction_static", 0.5),
    ("max_running_requests", 32),
    ("max_prefill_tokens", 1024),
    ("chunked_prefill_size", 512),
    ("context_length", 4096),
]

# Parallelism / placement keys that pass through ``**overrides``.
# (``tp_size``, ``base_gpu_id``, ``dist_init_addr`` are direct kwargs of
# the helper; passing them again via `**overrides` is a Python-level
# TypeError before our reject can fire — the runner's
# ``_WORKER_MANAGED_KEYS`` reject catches users who try this through the
# runner's ``server_args_overrides`` dict.)
_PARALLELISM = [
    ("pp_size", 2),
    ("dp_size", 2),
    ("ep_size", 4),
    ("moe_dense_tp_size", 2),
    ("nnodes", 2),
    ("node_rank", 1),
    # RFC v2: rank-topology keys reachable through `**overrides`.
    # `tp_rank`, `gpu_id`, `nccl_port`, `rank`, `world_size` must not
    # be settable through server_args_overrides because the runner
    # already decided them from StageConfig.tp_size + StageConfig.gpu.
    ("tp_rank", 1),
    ("gpu_id", 7),
    ("nccl_port", 29500),
    ("rank", 1),
    ("world_size", 4),
]

# Encoder-only fork knobs.
_ENCODER_FORK = [
    ("encoder_only", False),
    ("language_only", True),
    ("mm_enable_dp_encoder", True),
    ("enable_dp_attention", True),
    ("enable_dp_lm_head", True),
    ("disable_cuda_graph", False),
    ("device", "cpu"),
]


@pytest.mark.parametrize("key,value", _AR_ONLY)
def test_helper_rejects_ar_only_knobs(key, value):
    with pytest.raises(ValueError, match=key):
        build_sglang_encoder_server_args(
            model_path="dummy/model",
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr="127.0.0.1:29500",
            **{key: value},
        )


@pytest.mark.parametrize("key,value", _PARALLELISM)
def test_helper_rejects_parallelism_keys(key, value):
    with pytest.raises(ValueError, match=key):
        build_sglang_encoder_server_args(
            model_path="dummy/model",
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr="127.0.0.1:29500",
            **{key: value},
        )


@pytest.mark.parametrize("key,value", _ENCODER_FORK)
def test_helper_rejects_encoder_fork_keys(key, value):
    with pytest.raises(ValueError, match=key):
        build_sglang_encoder_server_args(
            model_path="dummy/model",
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr="127.0.0.1:29500",
            **{key: value},
        )


# ---------------------------------------------------------------------------
# Runner-level reject — protected keys reachable only through
# ``server_args_overrides`` dict on the runner
# ---------------------------------------------------------------------------


def _stub_specs():
    """Minimal valid spec list for runner validation tests."""
    from types import SimpleNamespace

    return (
        SimpleNamespace(
            name="dummy",
            build_module=lambda hf, qc: None,
            checkpoint_prefixes=("dummy.",),
            checkpoint_rewrites=(),
            stacked_params_mapping=(),
        ),
    )


def test_runner_rejects_runner_managed_keys_in_overrides_dict():
    """SGLangEncoderRunner rejects runner-managed keys *before* the
    helper-level **splat, otherwise Python raises a confusing
    "got multiple values" TypeError. Tested without actually
    instantiating the runner (which would require a real model)."""
    from sglang_omni.model_runner import sglang_encoder_runner as sew

    cls = sew.SGLangEncoderRunner
    inst = cls.__new__(cls)
    with pytest.raises(ValueError, match="runner-managed keys"):
        cls.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=29500,
            encoder_specs=_stub_specs(),
            server_args_overrides={"tp_size": 99},
        )


def test_runner_rejects_invalid_tp_rank():
    from sglang_omni.model_runner.sglang_encoder_runner import SGLangEncoderRunner

    inst = SGLangEncoderRunner.__new__(SGLangEncoderRunner)
    with pytest.raises(ValueError, match="tp_rank"):
        SGLangEncoderRunner.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=2,
            tp_size=2,  # tp_rank must be < tp_size, so 2 is invalid
            nccl_port=None,
            encoder_specs=_stub_specs(),
        )


def test_runner_rejects_invalid_tp_size():
    from sglang_omni.model_runner.sglang_encoder_runner import SGLangEncoderRunner

    inst = SGLangEncoderRunner.__new__(SGLangEncoderRunner)
    with pytest.raises(ValueError, match="tp_size"):
        SGLangEncoderRunner.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=0,
            tp_size=0,
            nccl_port=None,
            encoder_specs=_stub_specs(),
        )


def test_runner_rejects_empty_encoder_specs():
    """Locks the [Upstream Reuse Boundary] contract: the runner MUST
    refuse to fall back to ``get_model()`` on the full
    ``ForConditionalGeneration`` class. An adapter without any
    ``encoder_specs`` is a misconfigured stage, not a "load
    everything" hint."""
    from sglang_omni.model_runner.sglang_encoder_runner import SGLangEncoderRunner

    inst = SGLangEncoderRunner.__new__(SGLangEncoderRunner)
    with pytest.raises(ValueError, match="encoder_specs"):
        SGLangEncoderRunner.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=29500,
            encoder_specs=(),
        )


def test_runner_rejects_missing_nccl_port():
    from sglang_omni.model_runner.sglang_encoder_runner import SGLangEncoderRunner

    inst = SGLangEncoderRunner.__new__(SGLangEncoderRunner)
    with pytest.raises(ValueError, match="nccl_port"):
        SGLangEncoderRunner.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=None,
            encoder_specs=_stub_specs(),
        )


def test_runner_rejects_unknown_tp_parity_mode():
    from sglang_omni.model_runner.sglang_encoder_runner import SGLangEncoderRunner

    inst = SGLangEncoderRunner.__new__(SGLangEncoderRunner)
    with pytest.raises(ValueError, match="tp_parity_mode"):
        SGLangEncoderRunner.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=29500,
            encoder_specs=_stub_specs(),
            tp_parity_mode="nope",
        )


def test_runner_accepts_fp32_linear_tp_parity_mode():
    from sglang_omni.model_runner.sglang_encoder_runner import _resolve_tp_parity_mode

    assert _resolve_tp_parity_mode("fp32_linear") == "fp32_linear"


def test_runner_configures_tp_collective_switches(monkeypatch):
    """Encoder runner bypasses upstream ModelRunner, so it must mirror
    ModelRunner's all-reduce switch plumbing before TP group init."""
    from types import SimpleNamespace

    from sglang_omni.model_runner import sglang_encoder_runner as sew

    calls = {}
    monkeypatch.setattr(
        sew,
        "set_custom_all_reduce",
        lambda enabled: calls.setdefault("custom", enabled),
    )
    monkeypatch.setattr(
        sew,
        "set_mscclpp_all_reduce",
        lambda enabled: calls.setdefault("mscclpp", enabled),
    )
    monkeypatch.setattr(
        sew,
        "set_torch_symm_mem_all_reduce",
        lambda enabled: calls.setdefault("torch_symm_mem", enabled),
    )

    sew._configure_tp_collectives(
        SimpleNamespace(
            disable_custom_all_reduce=True,
            enable_mscclpp=True,
            enable_torch_symm_mem=False,
        )
    )

    assert calls == {
        "custom": False,
        "mscclpp": True,
        "torch_symm_mem": False,
    }
