# SPDX-License-Identifier: Apache-2.0
"""Thinker TP factory configuration with CFG safeguards."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.arg_groups import model_override_base

from sglang_omni import platforms
from sglang_omni.models.llada2_uni import bootstrap, stages
from sglang_omni.models.llada2_uni.cfg_attention_backend import CFG_ATTENTION_BACKEND
from sglang_omni.scheduling import sglang_backend
from sglang_omni.vendor.sglang import server_args as server_args_module


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("algorithm", ["LowConfidence", "LowConfidenceCFG"])
def test_thinker_factory_preserves_tp_and_cfg_configuration(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, tp_rank: int, algorithm: str
) -> None:
    server_args = SimpleNamespace(dllm_algorithm=algorithm, mem_fraction_static=0.5)
    build_args = Mock(return_value=server_args)
    create_scheduler = Mock()
    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", build_args)
    monkeypatch.setattr(bootstrap, "create_dllm_thinker_scheduler", create_scheduler)
    monkeypatch.setattr(bootstrap, "register_llada2_uni_cfg", Mock())
    override_args = Mock()
    monkeypatch.setattr(server_args_module, "override_server_args", override_args)
    monkeypatch.setattr(model_override_base, "resolved_view", lambda args: args)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(device_type="cuda")
    )

    scheduler = stages.create_sglang_dllm_thinker_executor_from_config(
        "model",
        device="cuda",
        gpu_id=tp_rank + 2,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=29500,
        dllm_algorithm=algorithm,
        server_args_overrides={"tp_size": 8, "mem_fraction_static": 0.5},
    )

    assert scheduler is create_scheduler.return_value
    create_scheduler.assert_called_once_with(
        server_args, tp_rank + 2, tp_rank=tp_rank, nccl_port=29500
    )
    settings = build_args.call_args.kwargs
    assert settings["tp_size"] == tp_size
    assert settings["disable_cuda_graph"] is True
    assert settings["mem_fraction_static"] == 0.5
    assert settings["dllm_algorithm"] == algorithm
    if algorithm == "LowConfidenceCFG":
        assert settings["attention_backend"] == CFG_ATTENTION_BACKEND
        assert settings["dllm_fdfo"] is False
        override_args.assert_called_once_with(
            server_args,
            "sglang_omni.llada2_uni.cfg_attention",
            attention_backend=CFG_ATTENTION_BACKEND,
        )
    else:
        assert settings["attention_backend"] == "flashinfer"
