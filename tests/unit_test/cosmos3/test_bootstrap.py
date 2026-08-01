# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.utils import hf_transformers_utils

from sglang_omni.model_runner import base as model_runner_base
from sglang_omni.models.cosmos3 import bootstrap, request_builders
from sglang_omni.models.cosmos3.bootstrap import resolve_transformer_weights_path
from sglang_omni.scheduling import (
    bootstrap as scheduling_bootstrap,
)
from sglang_omni.scheduling import (
    omni_scheduler,
    sglang_backend,
)


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
    local_only_calls: list[bool] = []

    def fake_snapshot_download(model_path, **kwargs):
        del model_path
        local_only = bool(kwargs.get("local_files_only"))
        local_only_calls.append(local_only)
        return str(partial_snapshot if local_only else complete_snapshot)

    monkeypatch.setattr(bootstrap, "snapshot_download", fake_snapshot_download)

    assert resolve_transformer_weights_path("nvidia/Cosmos3-Nano") == str(
        transformer_path
    )
    assert local_only_calls == [True, False]


def test_thinker_loads_tokenizer_from_checkpoint_root(monkeypatch, tmp_path) -> None:
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (transformer_path / "model.safetensors").touch()
    server_args = SimpleNamespace(model_path=str(tmp_path))
    model_config = SimpleNamespace(
        model_path=str(transformer_path),
        vocab_size=10,
        hf_generation_config=SimpleNamespace(),
    )
    tokenizer_paths: list[str] = []

    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            model_config,
        ),
    )
    monkeypatch.setattr(
        hf_transformers_utils,
        "get_tokenizer",
        lambda path, **kwargs: tokenizer_paths.append(path) or object(),
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
        lambda: object(),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(omni_scheduler, "OmniScheduler", SimpleNamespace)

    bootstrap.create_thinker_scheduler(server_args)

    assert tokenizer_paths == [str(tmp_path)]
