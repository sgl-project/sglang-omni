# SPDX-License-Identifier: Apache-2.0
"""CPU tests for weight-IPC configuration, compatibility, and lifecycle."""

from __future__ import annotations

import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from sglang_omni.distributed.weight_ipc import lifecycle, runtime
from sglang_omni.distributed.weight_ipc.compat import validate_weight_ipc_compatibility
from sglang_omni.distributed.weight_ipc.config import (
    apply_weight_ipc_cli_env,
    is_weight_ipc_follower,
    resolve_weight_ipc_config,
)
from sglang_omni.distributed.weight_ipc.export import compute_name_digest
from sglang_omni.distributed.weight_ipc.import_ import (
    WeightIpcImportError,
    validate_bundle,
)
from sglang_omni.distributed.weight_ipc.store import WeightIpcStore
from sglang_omni.distributed.weight_ipc.types import (
    SCHEMA_VERSION,
    IpcTensorMeta,
    WeightIpcBundle,
    WeightIpcConfig,
    WeightIpcRole,
)

_ENV_KEYS = (
    "SGLANG_OMNI_WEIGHT_IPC_ROLE",
    "WEIGHT_IPC_ROLE",
    "SGLANG_OMNI_WEIGHT_IPC_STORE",
    "WEIGHT_IPC_STORE",
    "SGLANG_OMNI_WEIGHT_IPC_TIMEOUT_S",
    "WEIGHT_IPC_TIMEOUT_S",
)
_HIGGS_ARCHITECTURE = "HiggsMultimodalQwen3ForConditionalGeneration"


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.setenv(key, "")


def test_pid_is_alive_rejects_zombie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lifecycle.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        lifecycle.Path,
        "read_text",
        lambda _path, encoding=None: "123 (weight-ipc-leader) Z 1",
    )

    assert not lifecycle.pid_is_alive(123)


def _meta(name: str, shape: tuple[int, ...] = (2, 2)) -> IpcTensorMeta:
    return IpcTensorMeta(
        name=name,
        shape=shape,
        stride=(shape[1], 1) if len(shape) == 2 else (1,),
        dtype="torch.float32",
        nbytes=8,
        device_index=0,
        handle=b"\x00" * 64,
        storage_size_bytes=64,
        storage_offset_bytes=0,
        allocation_offset_bytes=0,
        tensor_offset=0,
        requires_grad=False,
        ref_counter_handle=b"",
        ref_counter_offset=0,
        event_handle=b"",
        event_sync_required=False,
    )


def _bundle() -> WeightIpcBundle:
    tensors = [_meta("weight"), _meta("bias", (2,))]
    return WeightIpcBundle(
        schema_version=SCHEMA_VERSION,
        model_path="model",
        model_revision="revision",
        cuda_driver=None,
        created_unix_s=1.0,
        leader_pid=1234,
        tensors=tensors,
        name_digest=compute_name_digest(tensors),
    )


def test_config_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)

    assert resolve_weight_ipc_config().role == "off"
    assert not is_weight_ipc_follower()


@pytest.mark.parametrize("role", ["leader", "follower"])
def test_active_role_requires_store(
    role: WeightIpcRole, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("WEIGHT_IPC_ROLE", role)

    with pytest.raises(ValueError, match="store"):
        resolve_weight_ipc_config()


def test_cli_config_populates_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)

    apply_weight_ipc_cli_env(role="follower", store=str(tmp_path), timeout_s=30.0)

    assert resolve_weight_ipc_config() == WeightIpcConfig(
        role="follower", store_dir=str(tmp_path), timeout_s=30.0
    )
    assert is_weight_ipc_follower()


def test_compatibility_checks_are_skipped_when_disabled() -> None:
    validate_weight_ipc_compatibility(
        role="off", architectures=None, tp_size=8, pp_size=4
    )


@pytest.mark.parametrize("role", ["leader", "follower"])
def test_higgs_supports_active_roles(role: WeightIpcRole) -> None:
    validate_weight_ipc_compatibility(
        role=role,
        architectures=[_HIGGS_ARCHITECTURE],
        tp_size=1,
        pp_size=1,
    )


@pytest.mark.parametrize(
    ("architectures", "tp_size", "pp_size", "message"),
    [
        (None, 1, 1, "exactly one model architecture"),
        ([""], 1, 1, "exactly one model architecture"),
        (["A", "B"], 1, 1, "exactly one model architecture"),
        (["UnsupportedArchitecture"], 1, 1, "unsupported for architecture"),
        ([_HIGGS_ARCHITECTURE], 2, 1, "tp_size=1"),
        ([_HIGGS_ARCHITECTURE], 1, 2, "pp_size=1"),
    ],
)
def test_compatibility_rejects_unsupported_configuration(
    architectures: list[str] | None,
    tp_size: int,
    pp_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_weight_ipc_compatibility(
            role="leader",
            architectures=architectures,
            tp_size=tp_size,
            pp_size=pp_size,
        )


def test_store_roundtrip_uses_private_files(tmp_path: Path) -> None:
    store = WeightIpcStore(tmp_path / "weight_ipc")
    bundle = _bundle()

    store.write_bundle(bundle)

    assert store.load_bundle() == bundle
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    for path in (
        store.bundle_path,
        store.ready_path,
        store.manifest_path,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_prepare_for_write_removes_stale_publication(tmp_path: Path) -> None:
    store = WeightIpcStore(tmp_path / "weight_ipc")
    store.write_bundle(_bundle())

    store.prepare_for_write()

    assert not any(
        path.exists()
        for path in (
            store.bundle_path,
            store.ready_path,
            store.manifest_path,
        )
    )


def test_store_rejects_group_permissions(tmp_path: Path) -> None:
    store = WeightIpcStore(tmp_path / "weight_ipc")
    store.write_bundle(_bundle())
    os.chmod(store.root, 0o750)

    with pytest.raises(PermissionError, match="group/world"):
        store.load_bundle()


def test_store_rejects_another_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = WeightIpcStore(tmp_path / "weight_ipc")
    store.write_bundle(_bundle())
    monkeypatch.setattr(os, "geteuid", lambda: store.root.stat().st_uid + 1)

    with pytest.raises(PermissionError, match="owned by uid"):
        store.load_bundle()


@pytest.mark.parametrize("entry", ["root", "bundle"])
def test_store_rejects_symlinks(tmp_path: Path, entry: str) -> None:
    if entry == "root":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        store_path = tmp_path / "weight_ipc"
        store_path.symlink_to(target, target_is_directory=True)
        store = WeightIpcStore(store_path)
        message = "real directory"
    else:
        store = WeightIpcStore(tmp_path / "weight_ipc")
        store.prepare()
        payload = tmp_path / "payload.pkl"
        payload.write_bytes(b"not a bundle")
        os.chmod(payload, 0o600)
        store.bundle_path.symlink_to(payload)
        message = "regular file"

    with pytest.raises(PermissionError, match=message):
        store.load_bundle()


@pytest.mark.parametrize(
    ("bundle", "model_path", "message"),
    [
        (replace(_bundle(), name_digest="deadbeef"), "model", "name_digest"),
        (replace(_bundle(), model_path="other"), "model", "model_path"),
    ],
)
def test_validate_bundle_rejects_mismatches(
    bundle: WeightIpcBundle, model_path: str, message: str
) -> None:
    with pytest.raises(WeightIpcImportError, match=message):
        validate_bundle(
            bundle,
            model_path=model_path,
            model_revision="revision",
        )


def test_leader_clears_stale_publication_before_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_dir = tmp_path / "weight_ipc"
    store_dir.mkdir(mode=0o700)
    ready_path = store_dir / "READY"
    ready_path.write_text("stale\n", encoding="utf-8")
    os.chmod(ready_path, 0o600)

    def assert_stale_state_removed(*args, **kwargs):
        assert not ready_path.exists()
        raise RuntimeError("stop after assertion")

    monkeypatch.setattr(runtime, "export_shared_weights", assert_stale_state_removed)

    with pytest.raises(RuntimeError, match="stop after assertion"):
        runtime.export_leader_weights(
            object(),  # type: ignore[arg-type]
            WeightIpcConfig(role="leader", store_dir=str(store_dir)),
            model_path="model",
            model_revision=None,
        )
