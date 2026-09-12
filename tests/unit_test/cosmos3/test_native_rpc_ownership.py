# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI

from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.models.cosmos3.stages import native_server_kwargs
from sglang_omni.serve.native_media import prepare_native_media_app


@pytest.mark.parametrize("overrides", [None, {"scheduler_rpc_timeout": None}])
def test_generation_keeps_native_request_open_until_work_settles(overrides):
    kwargs = native_server_kwargs("checkpoint", 0, overrides)
    assert kwargs.get("scheduler_rpc_timeout") is None


@pytest.mark.parametrize("timeout", [1, 600])
def test_finite_rpc_deadline_is_rejected_before_native_startup(timeout):
    with pytest.raises(ValueError, match="before work settles"):
        native_server_kwargs("checkpoint", 0, {"scheduler_rpc_timeout": timeout})


def test_native_http_app_and_generation_share_no_rpc_deadline(monkeypatch):
    http_server = ModuleType("sglang.multimodal_gen.runtime.entrypoints.http_server")
    server_args = ModuleType("sglang.multimodal_gen.runtime.server_args")
    app = FastAPI()
    captured = []
    http_server.create_app = lambda args: captured.append(args) or app
    server_args.ServerArgs = SimpleNamespace(
        from_kwargs=lambda **kwargs: SimpleNamespace(
            **({"scheduler_rpc_timeout": None} | kwargs)
        )
    )
    server_args.set_global_server_args = lambda args: None
    monkeypatch.setitem(sys.modules, http_server.__name__, http_server)
    monkeypatch.setitem(sys.modules, server_args.__name__, server_args)
    config = Cosmos3PipelineConfig(model_path="checkpoint")
    config.stages[0].gpu = 0
    assert prepare_native_media_app(config, host="127.0.0.1", port=19000) is app
    assert captured[0].scheduler_rpc_timeout is None
    assert (
        config.stages[0].factory.server_args_overrides.get("scheduler_rpc_timeout")
        is None
    )
