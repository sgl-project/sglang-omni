# SPDX-License-Identifier: Apache-2.0
"""Accept the current native scheduler-client lifecycle contract."""

from types import SimpleNamespace

from fastapi import FastAPI

from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.serve.native_media import prepare_native_media_app


def test_native_frontend_uses_current_scheduler_client_contract(monkeypatch):
    from sglang.multimodal_gen.runtime import scheduler_client, server_args
    from sglang.multimodal_gen.runtime.entrypoints import http_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    def legacy_initialize(self, server_args):
        pass

    monkeypatch.setattr(server_args, "set_global_server_args", lambda args: None)
    initialized = []
    app = FastAPI()
    monkeypatch.setattr(
        scheduler_client.AsyncSchedulerClient,
        "initialize",
        legacy_initialize,
    )
    monkeypatch.setattr(ServerArgs, "from_kwargs", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        http_server, "create_app", lambda args: initialized.append(args) or app
    )
    config = Cosmos3PipelineConfig(model_path="checkpoint")
    assert prepare_native_media_app(config, host="127.0.0.1", port=19000) is app
    assert len(initialized) == 1
