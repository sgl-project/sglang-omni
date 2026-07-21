# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the weight-share export/attach file protocol.

The CUDA-IPC layer itself (torch reductions + MultiprocessingSerializer under
MPS) was validated by an on-GPU smoke test; these tests cover *our* protocol:
atomic publish, manifest verification, timeout, alias-not-copy semantics,
buffer coverage (incl. tied/duplicated registrations), the by-value path for
non-aliasable tensors, and post-attach drift detection.

An identity-preserving mock serializer stands in for CUDA IPC so aliasing is
observable with plain CPU tensors (`alias_predicate=lambda t: True`).
"""

from __future__ import annotations

import os
import pickle
import threading
import time

import pytest
import torch
from torch import nn

from sglang_omni.utils import ipc_weights
from sglang_omni.utils.ipc_weights import (
    WeightShareError,
    attach_weights,
    export_weights,
    get_weight_share_config,
    handle_file_for_model,
    verify_attachment,
)


class IdentitySerializer:
    """Mock of the CUDA-IPC serializer: 'deserializing' returns the very same
    tensor objects the exporter passed in, mimicking zero-copy handle opening
    inside a single test process."""

    _store: dict[bytes, dict] = {}
    _next = 0

    @classmethod
    def serialize(cls, obj) -> bytes:
        key = f"identity-{cls._next}".encode()
        cls._next += 1
        cls._store[key] = obj
        return key

    @classmethod
    def deserialize(cls, data: bytes):
        return cls._store[data]


class TinyModel(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.linear = nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn(3, 4, generator=gen))
            self.linear.bias.copy_(torch.randn(3, generator=gen))
        self.head = nn.Linear(4, 3, bias=False)
        # Tied parameter: same Parameter object registered on two modules.
        self.head.weight = self.linear.weight
        self.register_buffer("persistent_buf", torch.randn(2, 2, generator=gen))
        self.register_buffer(
            "cache_buf", torch.randn(5, generator=gen), persistent=False
        )


def _export(model, path, **kw):
    kw.setdefault("serializer", IdentitySerializer)
    kw.setdefault("alias_predicate", lambda t: True)
    # These protocol tests write into pytest's tmp dir (not a private 0700
    # store) with a mock serializer, so they opt out of the fs-trust checks.
    kw.setdefault("validate_secure", False)
    return export_weights(model, path, **kw)


def _attach(model, path, **kw):
    kw.setdefault("serializer", IdentitySerializer)
    kw.setdefault("timeout_s", 5.0)
    kw.setdefault("validate_secure", False)
    return attach_weights(model, path, **kw)


@pytest.fixture()
def handle_path(tmp_path):
    return str(tmp_path / "TinyModel.weights-ipc")


def test_roundtrip_aliases_all_params_and_buffers(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)  # dummy-weight stand-in: different values
    _export(leader, handle_path)
    record = _attach(follower, handle_path)

    # Parameters alias leader storage: same data_ptr, bit-identical values.
    assert follower.linear.weight.data_ptr() == leader.linear.weight.data_ptr()
    assert follower.linear.bias.data_ptr() == leader.linear.bias.data_ptr()
    # Buffers too — persistent and non-persistent.
    assert follower.persistent_buf.data_ptr() == leader.persistent_buf.data_ptr()
    assert follower.cache_buf.data_ptr() == leader.cache_buf.data_ptr()
    assert torch.equal(follower.linear.weight, leader.linear.weight)

    # Alias, not copy: writing through the leader is visible to the follower.
    with torch.no_grad():
        leader.linear.weight[0, 0] = 123.0
    assert follower.linear.weight[0, 0].item() == 123.0

    # Every shared tensor is covered by the attachment record.
    names = set(record)
    assert "linear.weight" in names
    assert "persistent_buf" in names and "cache_buf" in names


def test_tied_parameter_stays_tied_and_shared(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)
    _export(leader, handle_path)
    _attach(follower, handle_path)
    # The tie survives attach (one Parameter object on both modules) and both
    # views alias the leader's storage.
    assert follower.head.weight is follower.linear.weight
    assert follower.head.weight.data_ptr() == leader.linear.weight.data_ptr()


def test_attach_is_assignment_not_inplace_copy(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)
    before_ptr = follower.linear.weight.data_ptr()
    _export(leader, handle_path)
    _attach(follower, handle_path)
    # The follower's original (dummy) storage must have been dropped, not
    # written into.
    assert follower.linear.weight.data_ptr() != before_ptr


def test_value_path_copies_without_aliasing(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)
    # Only alias the parameters; buffers go through the by-value path the way
    # CPU-resident tensors do in production.
    param_ids = {id(p) for p in leader.parameters()}
    _export(leader, handle_path, alias_predicate=lambda t: id(t) in param_ids)
    _attach(follower, handle_path)
    assert torch.equal(follower.persistent_buf, leader.persistent_buf)
    assert follower.persistent_buf.data_ptr() != leader.persistent_buf.data_ptr()
    # Aliased params still share storage.
    assert follower.linear.weight.data_ptr() == leader.linear.weight.data_ptr()


def test_manifest_mismatch_extra_and_missing_names(tmp_path):
    class OtherShape(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)
            self.head = nn.Linear(4, 3, bias=False)
            self.head.weight = self.linear.weight
            self.register_buffer("persistent_buf", torch.randn(2, 2))
            # cache_buf shape differs (6 vs 5) and an extra buffer exists.
            self.register_buffer("cache_buf", torch.randn(6), persistent=False)
            self.register_buffer("extra_buf", torch.randn(1))

    # Same class *name* so the model-class gate passes and the manifest
    # (names + shapes + dtypes) is what rejects the attach.
    OtherShape.__name__ = "TinyModel"

    path = str(tmp_path / "TinyModel.weights-ipc")
    _export(TinyModel(seed=1), path)
    with pytest.raises(WeightShareError, match="manifest mismatch"):
        _attach(OtherShape(), path)


def test_model_class_mismatch_rejected(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    _export(TinyModel(seed=1), path)

    class Different(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

    with pytest.raises(WeightShareError, match="model class"):
        _attach(Different(), path)


def test_attach_timeout_on_missing_file(tmp_path):
    follower = TinyModel()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        _attach(
            follower,
            str(tmp_path / "never-appears.weights-ipc"),
            timeout_s=0.3,
            poll_interval_s=0.05,
        )
    assert time.monotonic() - t0 >= 0.3


def test_attach_waits_for_late_export(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)

    def late_export():
        time.sleep(0.4)
        _export(leader, handle_path)

    thread = threading.Thread(target=late_export)
    thread.start()
    try:
        _attach(follower, handle_path, timeout_s=5.0, poll_interval_s=0.05)
    finally:
        thread.join()
    assert follower.linear.weight.data_ptr() == leader.linear.weight.data_ptr()


def test_export_is_atomic_no_tmp_left_and_readable(handle_path, tmp_path):
    _export(TinyModel(seed=1), handle_path)
    leftovers = [f for f in os.listdir(tmp_path) if ".tmp." in f]
    assert leftovers == []
    with open(handle_path, "rb") as fh:
        payload = pickle.load(fh)
    assert payload["format_version"] == 1
    assert payload["model_class"] == "TinyModel"
    assert isinstance(payload["manifest_hash"], str)


def test_double_export_same_file_rejected(handle_path):
    leader = TinyModel(seed=1)
    _export(leader, handle_path)
    with pytest.raises(WeightShareError, match="already exported"):
        _export(leader, handle_path)


def test_corrupt_handle_file_rejected(handle_path):
    with open(handle_path, "wb") as fh:
        fh.write(pickle.dumps({"format_version": 999}))
    with pytest.raises(WeightShareError, match="format"):
        _attach(TinyModel(), handle_path)


def test_verify_attachment_detects_rebound_storage(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)
    _export(leader, handle_path)
    record = _attach(follower, handle_path)
    verify_attachment(follower, record)  # freshly attached: passes

    # Simulate a post-attach re-initialization (e.g. a stray loader step).
    follower.linear.weight.data = torch.zeros_like(follower.linear.weight)
    with pytest.raises(WeightShareError, match="rebound"):
        verify_attachment(follower, record)


def test_in_place_mutation_keeps_attachment_valid(handle_path):
    # In-place post-load mutation (the truncate_rope_to_bf16 pattern) keeps
    # storage identity, so it must NOT trip the guard and must stay shared.
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)
    _export(leader, handle_path)
    record = _attach(follower, handle_path)
    with torch.no_grad():
        leader.cache_buf.copy_(leader.cache_buf.to(torch.bfloat16).float())
    verify_attachment(follower, record)
    assert torch.equal(follower.cache_buf, leader.cache_buf)


def test_handle_file_for_model_uses_class_name(tmp_path):
    assert handle_file_for_model(str(tmp_path), TinyModel()).endswith(
        "TinyModel.weights-ipc"
    )


def test_validate_weight_share_architecture_allows_and_rejects():
    ipc_weights.validate_weight_share_architecture(
        ["HiggsMultimodalQwen3ForConditionalGeneration"]
    )
    for bad in ([], ["A", "B"], ["MossTTSLocalSGLangModel"], [""], None):
        with pytest.raises(WeightShareError):
            ipc_weights.validate_weight_share_architecture(bad)


def test_is_zombie_parses_state_after_comm(monkeypatch):
    # comm may contain spaces and ")": the state char is read after the LAST
    # ")", so a zombie is detected regardless of the process name.
    monkeypatch.setattr(
        ipc_weights.Path,
        "read_text",
        lambda _self, encoding=None: "42 (weight (leader)) Z 1 2 3\n",
    )
    assert ipc_weights._is_zombie(42)
    monkeypatch.setattr(
        ipc_weights.Path,
        "read_text",
        lambda _self, encoding=None: "42 (weight (leader)) R 1 2 3\n",
    )
    assert not ipc_weights._is_zombie(42)


def test_is_zombie_false_when_stat_unreadable(monkeypatch):
    def _raise(_self, encoding=None):
        raise OSError

    monkeypatch.setattr(ipc_weights.Path, "read_text", _raise)
    assert not ipc_weights._is_zombie(42)


_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX fs-trust checks")


@_POSIX_ONLY
def test_validate_secure_dir_rejects_group_world(tmp_path):
    os.chmod(tmp_path, 0o777)
    with pytest.raises(WeightShareError, match="group/world"):
        ipc_weights._validate_secure_dir(str(tmp_path))


@_POSIX_ONLY
def test_validate_secure_dir_rejects_foreign_owner(tmp_path, monkeypatch):
    os.chmod(tmp_path, 0o700)
    monkeypatch.setattr(ipc_weights.os, "geteuid", lambda: os.stat(tmp_path).st_uid + 1)
    with pytest.raises(WeightShareError, match="owned by"):
        ipc_weights._validate_secure_dir(str(tmp_path))


@_POSIX_ONLY
def test_validate_private_file_rejects_symlink(tmp_path):
    target = tmp_path / "real"
    target.write_bytes(b"x")
    os.chmod(target, 0o600)
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(WeightShareError):
        ipc_weights._validate_private_file(str(link))


def test_check_leader_alive_rejects_dead_pid(monkeypatch):
    monkeypatch.setattr(ipc_weights, "pid_is_alive", lambda pid: False)
    with pytest.raises(WeightShareError, match="not alive"):
        ipc_weights._check_leader_alive({"pid": 4321}, "before attach")
    monkeypatch.setattr(ipc_weights, "pid_is_alive", lambda pid: True)
    ipc_weights._check_leader_alive({"pid": 4321}, "before attach")  # alive: no raise


def test_load_payload_requires_positive_pid(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    with open(path, "wb") as fh:
        pickle.dump({"format_version": 1, "model_class": "TinyModel"}, fh)
    with pytest.raises(WeightShareError, match="invalid leader pid"):
        _attach(TinyModel(), path)


def test_model_path_mismatch_rejected(handle_path):
    _export(TinyModel(seed=1), handle_path, model_path="checkpoint-A")
    with pytest.raises(WeightShareError, match="model_path"):
        _attach(TinyModel(seed=2), handle_path, model_path="checkpoint-B")


def test_get_weight_share_config_parsing():
    assert get_weight_share_config({}) is None
    assert get_weight_share_config({"SGLANG_OMNI_WEIGHT_SHARE": ""}) is None
    assert get_weight_share_config({"SGLANG_OMNI_WEIGHT_SHARE": "  "}) is None

    cfg = get_weight_share_config({"SGLANG_OMNI_WEIGHT_SHARE": "leader:/x/y"})
    assert cfg.role == "leader" and cfg.dir_path == "/x/y"
    assert cfg.attach_timeout_s == ipc_weights.DEFAULT_ATTACH_TIMEOUT_S

    cfg = get_weight_share_config(
        {
            "SGLANG_OMNI_WEIGHT_SHARE": "follower:/x",
            "SGLANG_OMNI_WEIGHT_SHARE_TIMEOUT_S": "42.5",
        }
    )
    assert cfg.role == "follower" and cfg.attach_timeout_s == 42.5

    for bad in ("leader", "boss:/x", "leader:", ":/x"):
        with pytest.raises(ValueError):
            get_weight_share_config({"SGLANG_OMNI_WEIGHT_SHARE": bad})
    with pytest.raises(ValueError):
        get_weight_share_config(
            {
                "SGLANG_OMNI_WEIGHT_SHARE": "leader:/x",
                "SGLANG_OMNI_WEIGHT_SHARE_TIMEOUT_S": "-1",
            }
        )
