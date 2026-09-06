# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

from sglang.srt.utils import hf_transformers_utils

from sglang_omni.model_runner import base as model_runner_base
from sglang_omni.models.cosmos3 import bootstrap, request_builders
from sglang_omni.models.cosmos3.bootstrap import resolve_transformer_weights_path
from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
from sglang_omni.scheduling import omni_scheduler, sglang_backend


def test_resolves_nested_transformer_weights_directory(tmp_path) -> None:
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (transformer_path / "model.safetensors").touch()

    assert resolve_transformer_weights_path(str(tmp_path)) == str(transformer_path)


def test_remote_resolution_refetches_incomplete_local_snapshot(
    monkeypatch, tmp_path
) -> None:
    partial_snapshot = tmp_path / "partial"
    partial_snapshot.mkdir()
    complete_snapshot = tmp_path / "complete"
    transformer_path = complete_snapshot / "transformer"
    transformer_path.mkdir(parents=True)
    (transformer_path / "model.safetensors").touch()
    calls: list[tuple[bool, str | None]] = []

    def fake_snapshot_download(model_path, **kwargs):
        del model_path
        local_only = bool(kwargs.get("local_files_only"))
        calls.append((local_only, kwargs.get("revision")))
        return str(partial_snapshot if local_only else complete_snapshot)

    monkeypatch.setattr(bootstrap, "snapshot_download", fake_snapshot_download)

    assert resolve_transformer_weights_path(
        "nvidia/Cosmos3-Nano",
        revision="cosmos-revision",
    ) == str(transformer_path)
    assert calls == [
        (True, "cosmos-revision"),
        (False, "cosmos-revision"),
    ]


def test_thinker_loads_tokenizer_from_checkpoint_root(monkeypatch, tmp_path) -> None:
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (transformer_path / "model.safetensors").touch()
    server_args = SimpleNamespace(
        model_path=str(tmp_path),
        revision="cosmos-revision",
    )
    model_config = SimpleNamespace(
        model_path=str(transformer_path),
        vocab_size=10,
        hf_generation_config=SimpleNamespace(),
    )
    tokenizer_calls: list[tuple[str, str | None]] = []
    scheduler_signature = inspect.signature(omni_scheduler.OmniScheduler)
    scheduler_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            SimpleNamespace(),
            object(),
            object(),
            object(),
            model_config,
        ),
    )
    monkeypatch.setattr(
        hf_transformers_utils,
        "get_tokenizer",
        lambda path, **kwargs: (
            tokenizer_calls.append((path, kwargs.get("tokenizer_revision"))) or object()
        ),
    )
    monkeypatch.setattr(model_runner_base, "ModelRunner", lambda *args: object())
    monkeypatch.setattr(
        request_builders,
        "make_text_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        request_builders,
        "make_text_stream_output_builder",
        object,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )

    def fake_scheduler(*args, **kwargs):
        scheduler_signature.bind(*args, **kwargs)
        scheduler_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(omni_scheduler, "OmniScheduler", fake_scheduler)

    bootstrap.create_thinker_scheduler(server_args)

    assert tokenizer_calls == [(str(tmp_path), "cosmos-revision")]
    assert scheduler_kwargs["model_config"] is model_config
