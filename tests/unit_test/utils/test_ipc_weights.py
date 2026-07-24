# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the weight-share export/attach file protocol.

The CUDA-IPC layer itself was validated by an on-GPU smoke test; these cover
our file protocol: atomic publish, manifest verification, timeout,
alias-not-copy semantics, buffer coverage (including tied registrations), the
by-value path for non-aliasable tensors, and post-attach drift detection. An
identity-preserving mock serializer stands in for CUDA IPC so aliasing is
observable with plain CPU tensors.
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
        # Note (Jiaxin Deng): tie head to linear so the tied-param dedup path is
        # exercised (one Parameter object shared by two modules).
        self.head.weight = self.linear.weight
        self.register_buffer("persistent_buf", torch.randn(2, 2, generator=gen))
        self.register_buffer(
            "cache_buf", torch.randn(5, generator=gen), persistent=False
        )


def _export(model, path, **kw):
    kw.setdefault("serializer", IdentitySerializer)
    kw.setdefault("alias_predicate", lambda t: True)
    # Note (Jiaxin Deng): protocol tests use a mock serializer in pytest's tmp
    # dir, so opt out of fs-trust.
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

    assert follower.linear.weight.data_ptr() == leader.linear.weight.data_ptr()
    assert follower.linear.bias.data_ptr() == leader.linear.bias.data_ptr()
    assert follower.persistent_buf.data_ptr() == leader.persistent_buf.data_ptr()
    assert follower.cache_buf.data_ptr() == leader.cache_buf.data_ptr()
    assert torch.equal(follower.linear.weight, leader.linear.weight)

    with torch.no_grad():
        leader.linear.weight[0, 0] = 123.0
    assert follower.linear.weight[0, 0].item() == 123.0

    names = set(record)
    assert "linear.weight" in names
    assert "persistent_buf" in names and "cache_buf" in names


def test_tied_parameter_stays_tied_and_shared(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)
    _export(leader, handle_path)
    _attach(follower, handle_path)
    assert follower.head.weight is follower.linear.weight
    assert follower.head.weight.data_ptr() == leader.linear.weight.data_ptr()


def test_attach_is_assignment_not_inplace_copy(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)
    before_ptr = follower.linear.weight.data_ptr()
    _export(leader, handle_path)
    _attach(follower, handle_path)
    assert follower.linear.weight.data_ptr() != before_ptr


def test_value_path_copies_without_aliasing(handle_path):
    leader = TinyModel(seed=1)
    follower = TinyModel(seed=2)
    # Note (Jiaxin Deng): aliasing only the parameters drives buffers through
    # the by-value path, as CPU-resident tensors do in production.
    param_ids = {id(p) for p in leader.parameters()}
    _export(leader, handle_path, alias_predicate=lambda t: id(t) in param_ids)
    _attach(follower, handle_path)
    assert torch.equal(follower.persistent_buf, leader.persistent_buf)
    assert follower.persistent_buf.data_ptr() != leader.persistent_buf.data_ptr()
    assert follower.linear.weight.data_ptr() == leader.linear.weight.data_ptr()


def test_manifest_mismatch_extra_and_missing_names(tmp_path):
    class OtherShape(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)
            self.head = nn.Linear(4, 3, bias=False)
            self.head.weight = self.linear.weight
            self.register_buffer("persistent_buf", torch.randn(2, 2))
            # Note (Jiaxin Deng): diverge cache_buf shape (6 vs 5) and add
            # extra_buf so both the missing-name and extra-name paths fire.
            self.register_buffer("cache_buf", torch.randn(6), persistent=False)
            self.register_buffer("extra_buf", torch.randn(1))

    # Note (Jiaxin Deng): reuse the class name so the model-class gate passes
    # and the manifest (names, shapes, dtypes) is what rejects the attach.
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
    assert payload["format_version"] == ipc_weights._FORMAT_VERSION
    assert payload["model_class"] == "TinyModel"
    assert isinstance(payload["manifest_hash"], str)
    assert payload["private_names"] == []


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

    # Note (Jiaxin Deng): a post-attach re-init (a stray loader step) rebinds
    # storage and must be caught before graph capture.
    follower.linear.weight.data = torch.zeros_like(follower.linear.weight)
    with pytest.raises(WeightShareError, match="rebound"):
        verify_attachment(follower, record)


def test_in_place_mutation_keeps_attachment_valid(handle_path):
    # Note (Jiaxin Deng): in-place post-load mutation (the truncate_rope_to_bf16
    # pattern) keeps storage identity, so it must not trip the guard.
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
    # Note (Jiaxin Deng): exact-set locks so an arch cannot enter or leave
    # either registry without updating the expectations here. Supported means
    # audited plus a passing launcher e2e run; audit-only means audited and
    # still rejected by the gate.
    expected_supported = {
        "HiggsMultimodalQwen3ForConditionalGeneration": frozenset(),
        "MossTTSLocalSGLangModel": frozenset({"_decode_input_embedding.weight"}),
        "MossTTSDelaySGLangModel": frozenset({"_decode_input_embedding.weight"}),
        "MossTranscribeDiarizeForConditionalGeneration": frozenset(),
        "Qwen3ASRForConditionalGeneration": frozenset(),
        "WhisperForConditionalGeneration": frozenset(),
        "FunAsrNanoForConditionalGeneration": frozenset(),
    }
    expected_audit_only = {
        "MingTTSSGLangModel": frozenset({"_decode_input_embedding.weight"}),
        "VoxtralSGLangTTSModel": frozenset(),
        "S2ProSGLangTextModel": frozenset(),
        "LLaDA2MoeModelLM": frozenset(),
        "Qwen3TTSTalker": frozenset({"model._decode_feedback_embedding.weight"}),
        "Qwen3OmniThinkerForCausalLM": frozenset(),
        "Qwen3OmniTalker": frozenset(),
    }
    assert set(ipc_weights.WEIGHT_SHARE_POLICIES) == set(expected_supported)
    assert set(ipc_weights.AUDIT_ONLY_WEIGHT_SHARE_POLICIES) == set(expected_audit_only)
    assert not set(expected_supported) & set(expected_audit_only)
    for arch, private in expected_supported.items():
        policy = ipc_weights.validate_weight_share_architecture([arch])
        assert policy.private_tensor_names == private
    for arch, private in expected_audit_only.items():
        assert (
            ipc_weights.AUDIT_ONLY_WEIGHT_SHARE_POLICIES[arch].private_tensor_names
            == private
        )
        with pytest.raises(WeightShareError, match="support is in progress"):
            ipc_weights.validate_weight_share_architecture([arch])
    # Note (Jiaxin Deng): the Ming-Omni thinker is gate-reachable but only
    # spot-checked, so it must stay rejected until fully audited.
    higgs = "HiggsMultimodalQwen3ForConditionalGeneration"
    for bad in (
        [],
        ["A", "B"],
        ["BailingMoeV2ForCausalLM"],
        [""],
        None,
        [higgs, None],
        [higgs, "   "],
        [None],
    ):
        with pytest.raises(WeightShareError):
            ipc_weights.validate_weight_share_architecture(bad)


def test_is_zombie_parses_state_after_comm(monkeypatch):
    # Note (Jiaxin Deng): comm can contain spaces and ")", so the state char is
    # read only after the last ")".
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
def test_load_payload_rejects_symlinked_handle(tmp_path):
    # Note (Jiaxin Deng): the secure read path opens O_NOFOLLOW, so a symlinked
    # handle is refused.
    real_dir = tmp_path / "real"
    real_dir.mkdir(mode=0o700)
    real_path = str(real_dir / "TinyModel.weights-ipc")
    _export(TinyModel(seed=1), real_path, validate_secure=True)

    link_dir = tmp_path / "link"
    link_dir.mkdir(mode=0o700)
    link_path = link_dir / "TinyModel.weights-ipc"
    os.symlink(real_path, link_path)
    with pytest.raises(WeightShareError):
        _attach(TinyModel(seed=2), str(link_path), validate_secure=True)


def test_check_leader_alive_rejects_dead_pid(monkeypatch):
    monkeypatch.setattr(ipc_weights, "pid_is_alive", lambda pid: False)
    with pytest.raises(WeightShareError, match="not alive"):
        ipc_weights._check_leader_alive(
            {"pid": 4321, "leader_start_time": "1"}, "before attach"
        )
    monkeypatch.setattr(ipc_weights, "pid_is_alive", lambda pid: True)
    monkeypatch.setattr(ipc_weights, "_proc_start_time", lambda pid: "1")
    ipc_weights._check_leader_alive(
        {"pid": 4321, "leader_start_time": "1"}, "before attach"
    )  # alive + matching start time: no raise


@_POSIX_ONLY
def test_check_leader_alive_rejects_recycled_pid():
    # Note (Jiaxin Deng): same live pid but a different recorded start time
    # means the pid was reused.
    with pytest.raises(WeightShareError, match="recycled"):
        ipc_weights._check_leader_alive(
            {"pid": os.getpid(), "leader_start_time": "0"}, "before attach"
        )
    ok = {
        "pid": os.getpid(),
        "leader_start_time": ipc_weights._proc_start_time(os.getpid()),
    }
    ipc_weights._check_leader_alive(ok, "before attach")  # matching start: no raise


def _min_payload(**overrides):
    payload = {
        "format_version": ipc_weights._FORMAT_VERSION,
        "model_class": "TinyModel",
        "manifest_hash": "x",
        "private_names": [],
        "pid": os.getpid(),
        "leader_start_time": ipc_weights._proc_start_time(os.getpid()),
        "gpu_uuid": ipc_weights._gpu_uuid(),
        "ipc_blob": b"",
        "ipc_names": [],
        "value_blobs": {},
    }
    payload.update(overrides)
    return payload


def test_load_payload_requires_positive_pid(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    with open(path, "wb") as fh:
        pickle.dump(_min_payload(pid=0), fh)
    with pytest.raises(WeightShareError, match="invalid leader pid"):
        _attach(TinyModel(), path)


def test_load_payload_rejects_missing_required_field(tmp_path):
    for missing in (
        "model_class",
        "manifest_hash",
        "private_names",
        "ipc_blob",
        "ipc_names",
        "value_blobs",
    ):
        payload = _min_payload()
        del payload[missing]
        path = str(tmp_path / "TinyModel.weights-ipc")
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        with pytest.raises(WeightShareError, match="missing required field"):
            _attach(TinyModel(), path)


def test_load_payload_rejects_wrong_field_type(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    with open(path, "wb") as fh:
        pickle.dump(_min_payload(ipc_names="not-a-list"), fh)
    with pytest.raises(WeightShareError, match="wrong type"):
        _attach(TinyModel(), path)


def test_load_payload_rejects_unreadable_bytes(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    with open(path, "wb") as fh:
        fh.write(b"\x80\x04 truncated not a valid pickle")
    with pytest.raises(WeightShareError, match="not a readable payload"):
        _attach(TinyModel(), path)


def test_alias_rejects_non_tensor_blob(handle_path):
    class BadSerializer:
        @staticmethod
        def serialize(obj):
            return b"x"

        @staticmethod
        def deserialize(data):
            return {"linear.weight": "not-a-tensor"}

    _export(TinyModel(seed=1), handle_path)
    with pytest.raises(WeightShareError, match="tensor mapping"):
        _attach(TinyModel(seed=2), handle_path, serializer=BadSerializer)


def test_model_path_mismatch_rejected(handle_path):
    _export(TinyModel(seed=1), handle_path, model_path="checkpoint-A")
    with pytest.raises(WeightShareError, match="model_path"):
        _attach(TinyModel(seed=2), handle_path, model_path="checkpoint-B")


def test_run_id_mismatch_rejected(handle_path):
    _export(TinyModel(seed=1), handle_path, run_id="run-A")
    with pytest.raises(WeightShareError, match="run"):
        _attach(TinyModel(seed=2), handle_path, run_id="run-B")


def test_check_model_identity_rejects_wrong_gpu(monkeypatch):
    monkeypatch.setattr(ipc_weights, "_gpu_uuid", lambda: "GPU-2222")
    with pytest.raises(WeightShareError, match="GPU"):
        ipc_weights._check_model_identity(
            {"gpu_uuid": "GPU-1111"}, None, None, "handle", run_id=None
        )


@_POSIX_ONLY
def test_claim_namespace_refuses_second_leader(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    ipc_weights._claim_namespace(path, "run-A")  # first leader holds the flock
    with pytest.raises(WeightShareError, match="owns the weight-share"):
        ipc_weights._claim_namespace(path, "run-B")


class ScratchModel(nn.Module):
    """TinyModel plus a registered per-step scratch embedding, mirroring the
    MOSS decode staging pattern the private classification exists for."""

    def __init__(self, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.linear = nn.Linear(4, 3)
        self.scratch = nn.Embedding(4, 3)
        with torch.no_grad():
            self.linear.weight.copy_(torch.randn(3, 4, generator=gen))
            self.linear.bias.copy_(torch.randn(3, generator=gen))
            self.scratch.weight.copy_(torch.randn(4, 3, generator=gen))


_SCRATCH_PRIVATE = frozenset({"scratch.weight"})


def test_private_tensor_keeps_own_storage_and_copies_values(tmp_path):
    path = str(tmp_path / "ScratchModel.weights-ipc")
    leader = ScratchModel(seed=1)
    follower = ScratchModel(seed=2)
    scratch_ptr = follower.scratch.weight.data_ptr()
    _export(leader, path, private_names=_SCRATCH_PRIVATE)
    record = _attach(follower, path, private_names=_SCRATCH_PRIVATE)

    assert follower.linear.weight.data_ptr() == leader.linear.weight.data_ptr()
    assert follower.scratch.weight.data_ptr() == scratch_ptr
    assert follower.scratch.weight.data_ptr() != leader.scratch.weight.data_ptr()
    assert torch.equal(follower.scratch.weight, leader.scratch.weight)
    assert "scratch.weight" in record

    with torch.no_grad():
        leader.scratch.weight.fill_(11.0)
    assert not torch.any(follower.scratch.weight == 11.0)
    with torch.no_grad():
        follower.scratch.weight.fill_(22.0)
    assert not torch.any(leader.scratch.weight == 22.0)
    verify_attachment(follower, record)


def test_private_classification_mismatch_rejected(tmp_path):
    path = str(tmp_path / "ScratchModel.weights-ipc")
    _export(ScratchModel(seed=1), path, private_names=_SCRATCH_PRIVATE)
    with pytest.raises(WeightShareError, match="classification"):
        _attach(ScratchModel(seed=2), path)


def test_private_classification_mismatch_other_direction(tmp_path):
    path = str(tmp_path / "ScratchModel.weights-ipc")
    _export(ScratchModel(seed=1), path)
    with pytest.raises(WeightShareError, match="classification"):
        _attach(ScratchModel(seed=2), path, private_names=_SCRATCH_PRIVATE)


def test_unknown_private_name_fails_closed(tmp_path):
    path = str(tmp_path / "ScratchModel.weights-ipc")
    with pytest.raises(WeightShareError, match="diverged"):
        _export(ScratchModel(seed=1), path, private_names=frozenset({"gone.weight"}))
    _export(ScratchModel(seed=1), path, private_names=_SCRATCH_PRIVATE)
    with pytest.raises(WeightShareError, match="diverged"):
        _attach(ScratchModel(seed=2), path, private_names=frozenset({"gone.weight"}))


def test_ipc_blob_sharing_private_tensor_rejected(tmp_path):
    # Note (Jiaxin Deng): a leader that IPC-shared a policy-private tensor is
    # the corruption vector itself, so the follower must refuse the whole blob.
    path = str(tmp_path / "ScratchModel.weights-ipc")
    tensors = ipc_weights._named_shared_tensors(ScratchModel(seed=1))
    payload = _min_payload(
        model_class="ScratchModel",
        manifest_hash=ipc_weights._manifest_hash(tensors, _SCRATCH_PRIVATE),
        private_names=sorted(_SCRATCH_PRIVATE),
        ipc_blob=IdentitySerializer.serialize(dict(tensors)),
        ipc_names=sorted(tensors),
    )
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    with pytest.raises(WeightShareError, match="replica-private"):
        _attach(ScratchModel(seed=2), path, private_names=_SCRATCH_PRIVATE)


def test_manifest_hash_includes_classification():
    tensors = ipc_weights._named_shared_tensors(ScratchModel(seed=1))
    assert ipc_weights._manifest_hash(
        tensors, frozenset()
    ) != ipc_weights._manifest_hash(tensors, _SCRATCH_PRIVATE)


def test_verify_attachment_detects_private_rebound(tmp_path):
    path = str(tmp_path / "ScratchModel.weights-ipc")
    _export(ScratchModel(seed=1), path, private_names=_SCRATCH_PRIVATE)
    follower = ScratchModel(seed=2)
    record = _attach(follower, path, private_names=_SCRATCH_PRIVATE)
    follower.scratch.weight.data = torch.zeros_like(follower.scratch.weight)
    with pytest.raises(WeightShareError, match="rebound"):
        verify_attachment(follower, record)


def test_format_version_1_payload_rejected(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    with open(path, "wb") as fh:
        pickle.dump(_min_payload(format_version=1), fh)
    with pytest.raises(WeightShareError, match="format"):
        _attach(TinyModel(), path)


def test_cross_registered_private_tensor_rejected(tmp_path):
    class MixedDup(nn.Module):
        # Note (Jiaxin Deng): the same tensor object registered under a shared
        # parameter name and a private buffer name; the shared rebind would
        # drag the private registration onto leader storage.
        def __init__(self, seed: int = 0):
            super().__init__()
            gen = torch.Generator().manual_seed(seed)
            self.linear = nn.Linear(4, 3)
            self.scratch = nn.Parameter(torch.randn(4, 3, generator=gen))
            self._buffers["scratch_alias"] = self.scratch

    path = str(tmp_path / "MixedDup.weights-ipc")
    leader = MixedDup(seed=1)
    original = leader.scratch.detach().clone()
    _export(leader, path, private_names=frozenset({"scratch_alias"}))
    with pytest.raises(WeightShareError, match="cross-registered"):
        _attach(MixedDup(seed=2), path, private_names=frozenset({"scratch_alias"}))
    # The refusal must land before any private copy writes into the leader.
    assert torch.equal(leader.scratch.detach(), original)


def test_payload_with_unregistered_tensor_rejected(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    tensors = ipc_weights._named_shared_tensors(TinyModel(seed=1))
    blob = dict(tensors)
    blob["ghost.weight"] = torch.randn(2)
    payload = _min_payload(
        manifest_hash=ipc_weights._manifest_hash(tensors, frozenset()),
        ipc_blob=IdentitySerializer.serialize(blob),
        ipc_names=sorted(blob),
    )
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    with pytest.raises(WeightShareError, match="does not register"):
        _attach(TinyModel(seed=2), path)


def test_name_in_both_ipc_and_value_rejected(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    tensors = ipc_weights._named_shared_tensors(TinyModel(seed=1))
    payload = _min_payload(
        manifest_hash=ipc_weights._manifest_hash(tensors, frozenset()),
        ipc_blob=IdentitySerializer.serialize(dict(tensors)),
        ipc_names=sorted(tensors),
        value_blobs={
            "linear.weight": ipc_weights._tensor_to_value_bytes(
                tensors["linear.weight"]
            )
        },
    )
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    with pytest.raises(WeightShareError, match="both as IPC and"):
        _attach(TinyModel(seed=2), path)


def test_schema_rejects_non_string_names(tmp_path):
    path = str(tmp_path / "TinyModel.weights-ipc")
    with open(path, "wb") as fh:
        pickle.dump(_min_payload(ipc_names=[[]]), fh)
    with pytest.raises(WeightShareError, match="unique tensor names"):
        _attach(TinyModel(), path)


def test_shared_scratch_interleaving_corrupts_and_private_isolates(tmp_path):
    """The regression the MOSS policy exists for: two replicas stage different
    requests into the same scratch rows. Shared storage lets the second write
    clobber the first replica's staged data; private storage must not."""
    shared_path = str(tmp_path / "shared" / "ScratchModel.weights-ipc")
    leader = ScratchModel(seed=1)
    follower = ScratchModel(seed=2)
    _export(leader, shared_path)
    _attach(follower, shared_path)
    with torch.no_grad():
        leader.scratch.weight[:2].fill_(1.0)
        follower.scratch.weight[:2].fill_(2.0)
    assert torch.all(leader.scratch.weight[:2] == 2.0)

    private_path = str(tmp_path / "private" / "ScratchModel.weights-ipc")
    leader2 = ScratchModel(seed=1)
    follower2 = ScratchModel(seed=2)
    _export(leader2, private_path, private_names=_SCRATCH_PRIVATE)
    _attach(follower2, private_path, private_names=_SCRATCH_PRIVATE)
    with torch.no_grad():
        leader2.scratch.weight[:2].fill_(1.0)
        follower2.scratch.weight[:2].fill_(2.0)
    assert torch.all(leader2.scratch.weight[:2] == 1.0)
    assert torch.all(follower2.scratch.weight[:2] == 2.0)


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
