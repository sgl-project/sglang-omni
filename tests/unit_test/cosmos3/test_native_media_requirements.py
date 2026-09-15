# SPDX-License-Identifier: Apache-2.0
"""Refuse a native frontend that cannot observe its scheduler owner."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.serve.native_media import prepare_native_media_app


@pytest.mark.parametrize("supported", [False, True])
def test_native_owner_failure_contract_is_required(monkeypatch, supported):
    from sglang.multimodal_gen.runtime import scheduler_client, server_args
    from sglang.multimodal_gen.runtime.entrypoints import http_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    def legacy_initialize(self, server_args):
        pass

    def supported_initialize(self, server_args, *, worker_failure=None):
        pass

    monkeypatch.setattr(server_args, "set_global_server_args", lambda args: None)
    initialized = []
    app = FastAPI()
    monkeypatch.setattr(
        scheduler_client.AsyncSchedulerClient,
        "initialize",
        supported_initialize if supported else legacy_initialize,
    )
    monkeypatch.setattr(ServerArgs, "from_kwargs", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        http_server, "create_app", lambda args: initialized.append(args) or app
    )
    config = Cosmos3PipelineConfig(model_path="checkpoint")
    if supported:
        assert prepare_native_media_app(config, host="127.0.0.1", port=19000) is app
        assert len(initialized) == 1
    else:
        with pytest.raises(RuntimeError, match="scheduler owner-failure"):
            prepare_native_media_app(config, host="127.0.0.1", port=19000)
        assert initialized == []
