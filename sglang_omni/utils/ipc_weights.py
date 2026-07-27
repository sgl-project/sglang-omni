# SPDX-License-Identifier: Apache-2.0
"""Same-GPU weight sharing over CUDA IPC for same-GPU data parallelism.

One replica per GPU is the LEADER: it loads checkpoint weights, serializes
CUDA-IPC handles for every parameter and buffer, and atomically publishes the
handle blob to a file. FOLLOWERS build the same module tree with dummy weights
(no checkpoint I/O), then alias every parameter and buffer onto the leader's
storage by assignment, not copy. KV cache, CUDA graphs, and sampler state stay
private per replica.

Tensors named by the architecture's WeightSharePolicy as replica-private are
the exception: a registered tensor a model writes per request (a decode
staging scratch) must never alias across replicas, so the leader ships it by
value and the follower copies into the storage its own module build created,
keeping the address its CUDA graphs capture. Both sides derive the
classification from the same audited policy and fail closed on any mismatch.

Ordering contract, enforced by the caller and verified here:

- The leader exports at the end of load_model. Builder-level hooks (e.g. Higgs
  truncate_rope_to_bf16) run later and mutate the already-shared storage, so
  they must stay in-place, idempotent, and deterministic: a follower's
  redundant application must write identical bytes.
- Followers attach before KV-pool profiling, warmup, and graph capture.
  verify_attachment re-checks storage identity right before capture, so a
  silently re-initialized tensor fails loudly instead of serving dummy weights.
- The leader must outlive followers, since CUDA IPC mappings die with the
  exporting process; restart replicas together.
- The store dir must be unique per run: an exclusive flock lease refuses a
  second live leader, and a recycled pid or a different checkpoint is caught
  on attach.

Role and store dir come from SGLANG_OMNI_WEIGHT_SHARE (leader:<dir> or
follower:<dir>) so supervisors stay declarative; unset means sharing is off.
Each engine reads and writes <dir>/<ModelClass>.weights-ipc so multi-engine
stages cannot clobber each other.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

ENV_WEIGHT_SHARE = "SGLANG_OMNI_WEIGHT_SHARE"
ENV_WEIGHT_SHARE_TIMEOUT_S = "SGLANG_OMNI_WEIGHT_SHARE_TIMEOUT_S"
ENV_WEIGHT_SHARE_RUN_ID = "SGLANG_OMNI_WEIGHT_SHARE_RUN_ID"
DEFAULT_ATTACH_TIMEOUT_S = 1800.0
# Note (Jiaxin Deng): version 2 added per-tensor share/private classification;
# version-1 handles predate it and must not be attached.
_FORMAT_VERSION = 2
_FILE_SUFFIX = ".weights-ipc"

# Note (Jiaxin Deng): a second export to the same file silently invalidates
# handles attached followers already hold, so track exports and reject repeats.
_EXPORTED_FILES: set[str] = set()


class WeightShareError(RuntimeError):
    """Protocol violation during weight-share export/attach."""


@dataclass(frozen=True)
class WeightShareConfig:
    role: str  # "leader" | "follower"
    dir_path: str
    attach_timeout_s: float
    run_id: str | None = None


def get_weight_share_config(environ=None) -> WeightShareConfig | None:
    """Parse SGLANG_OMNI_WEIGHT_SHARE; None when unset or empty (off)."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_WEIGHT_SHARE) or "").strip()
    if not raw:
        return None
    role, sep, dir_path = raw.partition(":")
    role = role.strip().lower()
    dir_path = dir_path.strip()
    if not sep or role not in ("leader", "follower") or not dir_path:
        raise ValueError(
            f"{ENV_WEIGHT_SHARE} must be 'leader:<dir>' or 'follower:<dir>', "
            f"got {raw!r}"
        )
    run_id = (env.get(ENV_WEIGHT_SHARE_RUN_ID) or "").strip() or None
    timeout_raw = (env.get(ENV_WEIGHT_SHARE_TIMEOUT_S) or "").strip()
    if timeout_raw:
        try:
            timeout_s = float(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                f"{ENV_WEIGHT_SHARE_TIMEOUT_S} must be a number of seconds, "
                f"got {timeout_raw!r}"
            ) from exc
        if timeout_s <= 0:
            raise ValueError(f"{ENV_WEIGHT_SHARE_TIMEOUT_S} must be > 0")
    else:
        timeout_s = DEFAULT_ATTACH_TIMEOUT_S
    return WeightShareConfig(
        role=role, dir_path=dir_path, attach_timeout_s=timeout_s, run_id=run_id
    )


def handle_file_for_model(dir_path: str, model: torch.nn.Module) -> str:
    """Path <dir>/<ModelClass>.weights-ipc: one handle file per engine."""
    return os.path.join(dir_path, type(model).__name__ + _FILE_SUFFIX)


def _gpu_uuid() -> str | None:
    # Note (Jiaxin Deng): binds a publication to the physical GPU it was
    # exported from; None off CUDA so CPU tests stay lenient.
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return str(getattr(props, "uuid", "") or "") or None
    except Exception:
        return None


# Note (Jiaxin Deng): held open for the leader's lifetime so the flock lease
# releases atomically on process exit (no dead-owner unlink races).
_LEASE_FDS: list[int] = []


def _claim_namespace(file_path: str, run_id: str | None) -> None:
    # Note (Jiaxin Deng): a non-blocking flock is the only race-free lease: a
    # second leader on one directory fails to acquire it instead of clobbering
    # the first leader's followers, and the kernel releases it when the owner
    # dies. POSIX only; the launcher's per-run dir covers non-POSIX.
    if not _FS_TRUST_ENFORCED:
        return
    import fcntl

    lock_path = file_path + ".lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise WeightShareError(
            f"another live leader owns the weight-share namespace {file_path}; "
            f"refusing to clobber it ({exc})"
        ) from exc
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n{run_id or ''}\n".encode())
    _LEASE_FDS.append(fd)


class _SglangIpcSerializer:
    """CUDA-IPC (de)serialization via sglang's RLHF weight-update machinery.

    The torch-reductions monkey patch rewrites the reduced tensor's device
    index to the device UUID, which is what makes handles robust to each
    replica's UUID-pinned CUDA_VISIBLE_DEVICES ordering.
    """

    @staticmethod
    def serialize(obj: Any) -> bytes:
        from sglang.srt.utils.common import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        return MultiprocessingSerializer.serialize(obj)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        from sglang.srt.utils.common import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        return MultiprocessingSerializer.deserialize(data)


def _named_shared_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """All named parameters and buffers, deduplicated, name-collision checked.

    Deduplication folds tied parameters (e.g. Higgs modality_head.weight tied
    to the embedding) into one canonical name on both sides, so assigning .data
    through that single Parameter object updates every module holding it.
    """
    tensors: dict[str, torch.Tensor] = {}
    for source in (model.named_parameters(), model.named_buffers()):
        for name, tensor in source:
            if name in tensors:
                raise WeightShareError(
                    f"duplicate tensor name across parameters/buffers: {name!r}"
                )
            tensors[name] = tensor
    return tensors


def _manifest_hash(
    tensors: dict[str, torch.Tensor], private_names: frozenset[str]
) -> str:
    # Note (Jiaxin Deng): the classification is part of the manifest, so two
    # replicas that agree on names/shapes but not on what is private cannot
    # attach to each other.
    digest = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name]
        mode = "private" if name in private_names else "shared"
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}|{mode}\n".encode("utf-8"))
    return digest.hexdigest()


def _tensor_to_value_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu(), buf)
    return buf.getvalue()


def _value_bytes_to_tensor(data: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


@dataclass(frozen=True)
class WeightSharePolicy:
    """Audited per-architecture classification of registered tensors.

    private_tensor_names holds fully-qualified names (as reported by
    named_parameters/named_buffers) the model writes at serving time; each
    replica keeps its own storage for them, so they are never IPC-aliased.
    Every other registered tensor is shared.
    """

    private_tensor_names: frozenset[str] = frozenset()


# Note (Jiaxin Deng): the gate derives from these keys. An entry here requires
# a full post-load mutation audit (any registered tensor written per request
# listed private) AND a passing end-to-end run on the mps_dp launcher at the
# current revision: shared N=2 boot under MPS, attach verification, concurrent
# request correctness, and clean teardown. Audit alone is not enough; those
# entries live in AUDIT_ONLY_WEIGHT_SHARE_POLICIES below.
WEIGHT_SHARE_POLICIES: dict[str, WeightSharePolicy] = {
    "HiggsMultimodalQwen3ForConditionalGeneration": WeightSharePolicy(),
    # Note (Jiaxin Deng): MOSS local stages per-request decode feedback into
    # this registered embedding every step; everything else is load-once.
    "MossTTSLocalSGLangModel": WeightSharePolicy(
        private_tensor_names=frozenset({"_decode_input_embedding.weight"})
    ),
    # Note (Jiaxin Deng): MOSS delay stages decode feedback the same way MOSS
    # local does; its other registered tensors are load or init once.
    "MossTTSDelaySGLangModel": WeightSharePolicy(
        private_tensor_names=frozenset({"_decode_input_embedding.weight"})
    ),
    # Note (Jiaxin Deng): the four ASR models carry no decode staging scratch;
    # every registered tensor is checkpoint-loaded or an init-computed constant.
    "MossTranscribeDiarizeForConditionalGeneration": WeightSharePolicy(),
    "Qwen3ASRForConditionalGeneration": WeightSharePolicy(),
    "WhisperForConditionalGeneration": WeightSharePolicy(),
    "FunAsrNanoForConditionalGeneration": WeightSharePolicy(),
}

# Note (Jiaxin Deng): completed mutation audits whose launcher end-to-end
# validation has not passed yet; the gate rejects them so an audit cannot be
# advertised as support. Promotion to WEIGHT_SHARE_POLICIES requires the same
# e2e evidence as the entries above. The Ming-Omni thinker stays out entirely
# until audited.
AUDIT_ONLY_WEIGHT_SHARE_POLICIES: dict[str, WeightSharePolicy] = {
    # Note (Jiaxin Deng): Ming stages decode feedback like MOSS local; blocked
    # on VRAM (leader alone reaches the card edge on 80 GB).
    "MingTTSSGLangModel": WeightSharePolicy(
        private_tensor_names=frozenset({"_decode_input_embedding.weight"})
    ),
    # Note (Jiaxin Deng): Voxtral keeps its decode staging in an unregistered
    # plain tensor; if that scratch is ever registered, this needs its name.
    "VoxtralSGLangTTSModel": WeightSharePolicy(),
    # Note (Jiaxin Deng): Fish's staging is unregistered and its fast-AR
    # decoder, with per-step KV buffers, joins the module tree only after
    # export; the empty set holds only while export stays at load_model's end.
    "S2ProSGLangTextModel": WeightSharePolicy(),
    # Note (Jiaxin Deng): LLaDA2's denoise-loop state lives in per-replica
    # scheduler and pool state, never in registered tensors; its pipeline also
    # declares no generation SGLang stage, so the launcher cannot drive it.
    "LLaDA2MoeModelLM": WeightSharePolicy(),
    # Note (Jiaxin Deng): Qwen3-TTS stages decode feedback into this
    # dual-registered embedding, listed under its deduped canonical name; the
    # external speech tokenizer never enters the module tree.
    "Qwen3TTSTalker": WeightSharePolicy(
        private_tensor_names=frozenset({"model._decode_feedback_embedding.weight"})
    ),
    # Note (Jiaxin Deng): Qwen3-Omni runs two engines per pipeline; the
    # launcher only drives single-SGLang-engine pipelines, so these cannot
    # reach e2e validation on it. The talker keeps its decode staging in
    # unregistered plain attributes.
    "Qwen3OmniThinkerForCausalLM": WeightSharePolicy(),
    "Qwen3OmniTalker": WeightSharePolicy(),
}

# Note (Jiaxin Deng): the ASR entries assume the pinned sglang's rope caches
# stay bound after load; a bump that rebinds one at serving time (2-D mrope
# positions, spec decode) would silently orphan followers, so re-audit the
# rope stack when bumping sglang.

SUPPORTED_WEIGHT_SHARE_ARCHITECTURES = frozenset(WEIGHT_SHARE_POLICIES)

_FS_TRUST_ENFORCED = os.name == "posix"


def validate_weight_share_architecture(architectures: Any) -> WeightSharePolicy:
    """Fail fast unless the architecture is audited; return its share policy."""
    # Note (Jiaxin Deng): no normalizing away malformed entries; a config that
    # lists anything besides one nonblank architecture string must fail here.
    archs = list(architectures or [])
    if len(archs) != 1 or not isinstance(archs[0], str) or not archs[0].strip():
        raise WeightShareError(
            f"weight sharing requires exactly one model architecture, got {archs!r}"
        )
    arch = archs[0].strip()
    policy = WEIGHT_SHARE_POLICIES.get(arch)
    if policy is None:
        supported = ", ".join(sorted(WEIGHT_SHARE_POLICIES))
        if arch in AUDIT_ONLY_WEIGHT_SHARE_POLICIES:
            raise WeightShareError(
                f"weight sharing for architecture {arch!r} has a completed "
                "mutation audit but no passing launcher end-to-end validation; "
                f"support is in progress. Supported architectures: {supported}"
            )
        raise WeightShareError(
            f"weight sharing is unsupported for architecture {arch!r}; "
            f"supported architectures: {supported}. A model that writes "
            "per-request state in place into a shared parameter would corrupt "
            "co-located replicas"
        )
    return policy


def _validate_secure_dir(dir_path: str) -> None:
    """Reject a store dir another user could write; else they could plant a
    handle file the follower then unpickles. No-op off POSIX."""
    if not _FS_TRUST_ENFORCED:
        return
    st = os.lstat(dir_path)
    if not stat.S_ISDIR(st.st_mode):
        raise WeightShareError(f"weight-share store must be a directory: {dir_path}")
    if st.st_uid != os.geteuid():
        raise WeightShareError(
            f"weight-share store {dir_path} is owned by uid={st.st_uid}, expected "
            f"uid={os.geteuid()}"
        )
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise WeightShareError(
            f"weight-share store {dir_path} must not grant group/world "
            f"permissions, got mode={stat.S_IMODE(st.st_mode):#o}"
        )


def _check_private_stat(st: os.stat_result, file_path: str) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise WeightShareError(
            f"weight-share handle file must be a regular file: {file_path}"
        )
    if st.st_uid != os.geteuid():
        raise WeightShareError(
            f"weight-share handle file {file_path} is owned by uid={st.st_uid}, "
            f"expected uid={os.geteuid()}"
        )
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise WeightShareError(
            f"weight-share handle file {file_path} must not grant group/world "
            f"permissions, got mode={stat.S_IMODE(st.st_mode):#o}"
        )


def _prepare_secure_dir(dir_path: str) -> None:
    os.makedirs(dir_path, mode=0o700, exist_ok=True)
    if _FS_TRUST_ENFORCED:
        os.chmod(dir_path, 0o700)
    _validate_secure_dir(dir_path)


def _proc_stat_fields(pid: int) -> list[str] | None:
    """The whitespace fields of /proc/<pid>/stat after the (comm) field.

    Index 0 is the state char, index 19 is starttime. comm can contain spaces
    and parens, so the caller must split after the last ')'.
    """
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing_paren = stat_line.rfind(")")
    if closing_paren < 0:
        return None
    return stat_line[closing_paren + 1 :].split()


def _is_zombie(pid: int) -> bool:
    fields = _proc_stat_fields(pid)
    # Note (guozhihao): Z still passes kill(pid, 0) until it is reaped.
    return bool(fields) and fields[0] == "Z"


def _proc_start_time(pid: int) -> str | None:
    """Process start time, which differs for a recycled pid (None off Linux)."""
    fields = _proc_stat_fields(pid)
    if not fields or len(fields) <= 19:
        return None
    return fields[19]


def pid_is_alive(pid: int) -> bool:
    if os.name != "posix" or pid <= 0:
        return pid > 0
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _is_zombie(pid)


class LeaderLivenessMonitor:
    """Fail-fast exit the follower when the leader process disappears.

    A follower aliases the leader's CUDA storage, and that mapping is undefined
    once the exporting process exits, so continuing to serve on it would emit
    garbage or fault. The monitor turns that into a clean exit.
    """

    def __init__(
        self,
        leader_pid: int,
        *,
        leader_start_time: str | None = None,
        poll_interval_s: float = 1.0,
        exit_code: int = 70,
    ) -> None:
        self.leader_pid = leader_pid
        self.leader_start_time = leader_start_time
        self.poll_interval_s = poll_interval_s
        self.exit_code = exit_code
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _leader_present(self) -> bool:
        if not pid_is_alive(self.leader_pid):
            return False
        # Note (Jiaxin Deng): a recycled pid (leader died, pid reassigned) has
        # a different start time, so treat a mismatch as the leader being gone.
        if self.leader_start_time is not None:
            return _proc_start_time(self.leader_pid) == self.leader_start_time
        return True

    def start(self) -> None:
        if self._thread is not None or not self.leader_pid:
            return
        self._thread = threading.Thread(
            target=self._run, name="weight-share-leader-liveness", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            if self._leader_present():
                continue
            logger.critical(
                "[weight-share] leader pid=%s is gone; terminating follower",
                self.leader_pid,
            )
            os._exit(self.exit_code)


def _atomic_write(file_path: str, data: bytes) -> None:
    """Publish data to file_path via tmp, fsync, then atomic rename.

    Rename atomicity guarantees a polling follower never observes a partial
    blob. mkstemp creates the temp 0600 with O_EXCL under an unpredictable
    name, so a pre-planted symlink in the store cannot redirect the write.
    """
    dir_path = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(file_path)}.tmp.", dir=dir_path
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            if _FS_TRUST_ENFORCED:
                os.fchmod(fh.fileno(), 0o600)
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    try:
        dir_fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        # Note (Jiaxin Deng): best effort; rename atomicity holds regardless.
        pass


def _resolve_private_names(
    tensors: dict[str, torch.Tensor],
    private_names: frozenset[str],
    model: torch.nn.Module,
) -> frozenset[str]:
    # Note (Jiaxin Deng): a policy name with no matching tensor means the model
    # renamed its scratch; silently sharing it would corrupt replicas, so stop.
    unknown = private_names - set(tensors)
    if unknown:
        raise WeightShareError(
            f"weight-share policy names {sorted(unknown)} that are not "
            f"registered tensors of {type(model).__name__}; the policy and the "
            "model have diverged"
        )
    return private_names


def export_weights(
    model: torch.nn.Module,
    file_path: str,
    *,
    serializer: Any | None = None,
    alias_predicate: Callable[[torch.Tensor], bool] | None = None,
    validate_secure: bool = True,
    model_path: str | None = None,
    model_revision: str | None = None,
    run_id: str | None = None,
    private_names: frozenset[str] = frozenset(),
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Publish CUDA-IPC handles for every model parameter and buffer to a file.

    CUDA tensors are shared zero-copy through the serializer. Non-CUDA tensors
    (rare, e.g. CPU-resident buffers) are embedded by value because
    ForkingPickler's CPU shared-memory reduction does not survive a file handoff
    between unrelated processes, so followers copy those. Tensors in
    private_names ride the by-value path regardless of device: followers copy
    the leader's bytes into their own storage and never alias it.

    Returns a record {name: (data_ptr, shape, dtype)} of the leader-side shared
    tensors for verify_attachment: the leader's own storages must stay put since
    followers alias them, so in-place mutation is fine but rebinding .data after
    export is not.
    """
    abs_path = os.path.abspath(file_path)
    if abs_path in _EXPORTED_FILES:
        raise WeightShareError(
            f"this process already exported weights to {abs_path}; a second "
            "export would invalidate handles held by attached followers"
        )
    if validate_secure:
        _prepare_secure_dir(os.path.dirname(abs_path))
        _claim_namespace(abs_path, run_id)
    # Note (Jiaxin Deng): clear a previous run's stale export before publishing.
    if os.path.exists(abs_path):
        os.unlink(abs_path)
    serializer = _SglangIpcSerializer if serializer is None else serializer
    alias_predicate = (
        (lambda t: t.is_cuda) if alias_predicate is None else alias_predicate
    )

    tensors = _named_shared_tensors(model)
    if not tensors:
        raise WeightShareError("model has no parameters or buffers to export")
    private = _resolve_private_names(tensors, private_names, model)
    ipc_tensors = {
        n: t for n, t in tensors.items() if n not in private and alias_predicate(t)
    }
    value_tensors = {n: t for n, t in tensors.items() if n not in ipc_tensors}

    payload = {
        "format_version": _FORMAT_VERSION,
        "model_class": type(model).__name__,
        "manifest_hash": _manifest_hash(tensors, private),
        "private_names": sorted(private),
        "torch_version": torch.__version__,
        "pid": os.getpid(),
        "leader_start_time": _proc_start_time(os.getpid()),
        "model_path": model_path,
        "model_revision": model_revision,
        "gpu_uuid": _gpu_uuid(),
        "run_id": run_id,
        "ipc_blob": serializer.serialize(ipc_tensors),
        "ipc_names": sorted(ipc_tensors),
        "value_blobs": {n: _tensor_to_value_bytes(t) for n, t in value_tensors.items()},
    }
    _atomic_write(abs_path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    _EXPORTED_FILES.add(abs_path)

    record = {
        name: (t.data_ptr(), tuple(t.shape), t.dtype) for name, t in ipc_tensors.items()
    }
    shared_bytes = sum(t.numel() * t.element_size() for t in ipc_tensors.values())
    logger.info(
        f"[weight-share] leader exported {len(ipc_tensors)} CUDA tensors "
        f"({shared_bytes / (1 << 30):.2f} GiB shared zero-copy) + "
        f"{len(value_tensors)} by-value tensors to {abs_path}"
    )
    return record


def wait_for_export(
    file_path: str,
    timeout_s: float,
    poll_interval_s: float = 0.5,
) -> None:
    """Block until the leader's handle file exists (atomic rename = complete)."""
    deadline = time.monotonic() + timeout_s
    while not os.path.exists(file_path):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"[weight-share] timed out after {timeout_s:.0f}s waiting for "
                f"leader handle file {file_path}; is the leader replica up?"
            )
        time.sleep(poll_interval_s)


def wait_for_any_export(
    dir_path: str,
    timeout_s: float,
    poll_interval_s: float = 0.5,
) -> None:
    """Block until any leader handle file exists under dir_path.

    Used by followers before allocating dummy weights, when the engine's model
    class (hence exact file name) is not known yet. Attach still waits on and
    validates the engine-specific file afterwards.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            entries = os.listdir(dir_path)
        except FileNotFoundError:
            entries = []
        if any(e.endswith(_FILE_SUFFIX) for e in entries):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"[weight-share] timed out after {timeout_s:.0f}s waiting for "
                f"a leader handle file under {dir_path}; is the leader "
                "replica up?"
            )
        time.sleep(poll_interval_s)


# Note (Jiaxin Deng): a closed schema checked before deserialization, so a
# truncated or forged handle raises a named WeightShareError, not a raw KeyError.
_REQUIRED_PAYLOAD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "format_version": int,
    "model_class": str,
    "manifest_hash": str,
    "private_names": list,
    "pid": int,
    "ipc_blob": (bytes, bytearray),
    "ipc_names": list,
    "value_blobs": dict,
}


def _safe_unpickle(fh: Any, file_path: str) -> Any:
    try:
        return pickle.load(fh)
    except WeightShareError:
        raise
    except Exception as exc:
        raise WeightShareError(
            f"weight-share handle {file_path} is not a readable payload: {exc}"
        ) from exc


def _validate_payload_schema(payload: Any, file_path: str) -> None:
    if not isinstance(payload, dict):
        raise WeightShareError(
            f"weight-share handle {file_path} is not a payload dict "
            f"(got {type(payload).__name__})"
        )
    version = payload.get("format_version")
    if version != _FORMAT_VERSION:
        raise WeightShareError(
            f"unsupported weight-share handle format in {file_path} "
            f"(got {version!r}, expected {_FORMAT_VERSION})"
        )
    for field, expected in _REQUIRED_PAYLOAD_FIELDS.items():
        if field not in payload:
            raise WeightShareError(
                f"weight-share handle {file_path} is missing required field {field!r}"
            )
        if not isinstance(payload[field], expected):
            raise WeightShareError(
                f"weight-share handle {file_path} field {field!r} has wrong type "
                f"{type(payload[field]).__name__}"
            )
    for field in ("ipc_names", "private_names"):
        names = payload[field]
        if not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise WeightShareError(
                f"weight-share handle {file_path} field {field!r} must hold "
                "unique tensor names"
            )
    for key, blob in payload["value_blobs"].items():
        if not isinstance(key, str) or not isinstance(blob, (bytes, bytearray)):
            raise WeightShareError(
                f"weight-share handle {file_path} value_blobs must map tensor "
                "names to bytes"
            )


def _load_payload(
    file_path: str, model: torch.nn.Module, *, validate_secure: bool = True
) -> dict[str, Any]:
    if validate_secure and _FS_TRUST_ENFORCED:
        # Note (Jiaxin Deng): O_NOFOLLOW + fstat binds the check to the opened
        # inode before unpickling (an RCE surface).
        _validate_secure_dir(os.path.dirname(os.path.abspath(file_path)))
        try:
            fd = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise WeightShareError(
                f"refusing to open weight-share handle {file_path}: {exc}"
            ) from exc
        try:
            _check_private_stat(os.fstat(fd), file_path)
        except Exception:
            os.close(fd)
            raise
        with os.fdopen(fd, "rb") as fh:
            payload = _safe_unpickle(fh, file_path)
    else:
        with open(file_path, "rb") as fh:
            payload = _safe_unpickle(fh, file_path)
    _validate_payload_schema(payload, file_path)
    if payload["model_class"] != type(model).__name__:
        raise WeightShareError(
            f"handle file {file_path} was exported for model class "
            f"{payload['model_class']!r}, follower model is {type(model).__name__!r}"
        )
    if payload["pid"] <= 0:
        raise WeightShareError(
            f"handle file {file_path} has an invalid leader pid {payload['pid']!r}"
        )
    return payload


def attach_weights(
    model: torch.nn.Module,
    file_path: str,
    *,
    timeout_s: float = DEFAULT_ATTACH_TIMEOUT_S,
    poll_interval_s: float = 0.5,
    serializer: Any | None = None,
    validate_secure: bool = True,
    model_path: str | None = None,
    model_revision: str | None = None,
    run_id: str | None = None,
    private_names: frozenset[str] = frozenset(),
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Alias every model parameter and buffer onto the leader's exported storage.

    Assignment, not copy, so the follower's dummy-initialized storages are
    dropped and freed. Tensors in private_names keep the follower's own storage
    and only receive the leader's bytes. Hard-errors on any name, shape, dtype,
    or classification mismatch in either direction. Returns the attachment
    record for verify_attachment.
    """
    record, _ = _attach_and_check(
        model,
        file_path,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        serializer=serializer,
        validate_secure=validate_secure,
        model_path=model_path,
        model_revision=model_revision,
        run_id=run_id,
        private_names=private_names,
    )
    return record


def _attach_and_check(
    model: torch.nn.Module,
    file_path: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
    serializer: Any | None,
    validate_secure: bool,
    model_path: str | None,
    model_revision: str | None,
    run_id: str | None = None,
    private_names: frozenset[str] = frozenset(),
) -> tuple[dict[str, tuple[int, tuple[int, ...], torch.dtype]], dict[str, Any]]:
    serializer = _SglangIpcSerializer if serializer is None else serializer
    wait_for_export(file_path, timeout_s, poll_interval_s)
    payload = _load_payload(file_path, model, validate_secure=validate_secure)
    _check_model_identity(payload, model_path, model_revision, file_path, run_id=run_id)
    _check_leader_alive(payload, "before attach")
    record = _alias_from_payload(
        model, payload, file_path, serializer, private_names=private_names
    )
    _check_leader_alive(payload, "after attach")
    return record, payload


def _check_model_identity(
    payload: dict[str, Any],
    model_path: str | None,
    model_revision: str | None,
    file_path: str,
    *,
    run_id: str | None = None,
) -> None:
    # Note (Jiaxin Deng): strict only when the leader recorded a value; catches
    # a same-shape different-checkpoint, wrong-GPU, or cross-run attach.
    recorded_path = payload.get("model_path")
    if recorded_path is not None and recorded_path != model_path:
        raise WeightShareError(
            f"handle file {file_path} was exported for model_path "
            f"{recorded_path!r}, follower is {model_path!r}"
        )
    recorded_rev = payload.get("model_revision")
    if recorded_rev is not None and recorded_rev != model_revision:
        raise WeightShareError(
            f"handle file {file_path} was exported for revision "
            f"{recorded_rev!r}, follower is {model_revision!r}"
        )
    # Note (Jiaxin Deng): fail closed when the follower has an identity to
    # match: a running GPU or a launcher run id must find an equal recorded
    # value, so a stale export missing the field cannot slip through.
    own_uuid = _gpu_uuid()
    if own_uuid is not None and payload.get("gpu_uuid") != own_uuid:
        raise WeightShareError(
            f"handle file {file_path} was exported from GPU "
            f"{payload.get('gpu_uuid')!r}, follower is on {own_uuid!r}"
        )
    if run_id is not None and payload.get("run_id") != run_id:
        raise WeightShareError(
            f"handle file {file_path} belongs to run {payload.get('run_id')!r}, "
            f"follower is run {run_id!r}"
        )


def _check_leader_alive(payload: dict[str, Any], when: str) -> None:
    leader_pid = payload.get("pid")
    if not leader_pid:
        return
    if not pid_is_alive(int(leader_pid)):
        raise WeightShareError(
            f"weight-share leader pid={leader_pid} is not alive ({when}); "
            "refusing to serve on a dead leader's CUDA storage"
        )
    # Note (Jiaxin Deng): a recycled pid (new process) has a different start
    # time; require the recorded start time on Linux so this never fails open.
    recorded_start = payload.get("leader_start_time")
    current_start = _proc_start_time(int(leader_pid))
    if _FS_TRUST_ENFORCED and recorded_start is None:
        raise WeightShareError(
            f"weight-share handle for pid={leader_pid} has no leader start time "
            f"({when}); refusing to skip the recycled-pid check"
        )
    if recorded_start is not None and current_start != recorded_start:
        raise WeightShareError(
            f"weight-share leader pid={leader_pid} was recycled ({when}); "
            "the original exporting process is gone"
        )


def _alias_from_payload(
    model: torch.nn.Module,
    payload: dict[str, Any],
    file_path: str,
    serializer: Any,
    private_names: frozenset[str] = frozenset(),
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Alias every model parameter/buffer onto the payload's shared storage."""
    own_tensors = _named_shared_tensors(model)
    private = _resolve_private_names(own_tensors, private_names, model)
    if sorted(private) != payload["private_names"]:
        raise WeightShareError(
            f"weight-share classification mismatch for {file_path}: leader "
            f"marked {payload['private_names']!r} replica-private, this "
            f"follower's policy marks {sorted(private)!r}; the replicas are "
            "not running the same policy"
        )
    own_hash = _manifest_hash(own_tensors, private)
    if own_hash != payload["manifest_hash"]:
        _raise_manifest_mismatch(model, payload, own_tensors, file_path)

    try:
        shared: dict[str, torch.Tensor] = serializer.deserialize(payload["ipc_blob"])
    except WeightShareError:
        raise
    except Exception as exc:
        raise WeightShareError(
            f"failed to open shared CUDA tensors from {file_path}: {exc}"
        ) from exc
    if not isinstance(shared, dict) or not all(
        isinstance(k, str) and isinstance(t, torch.Tensor) for k, t in shared.items()
    ):
        raise WeightShareError(
            f"handle blob in {file_path} did not deserialize to a str->tensor mapping"
        )
    if sorted(shared) != payload["ipc_names"]:
        raise WeightShareError(
            f"handle blob in {file_path} deserialized to a different tensor "
            "set than its manifest declares"
        )
    leaked = sorted(private & set(shared))
    if leaked:
        raise WeightShareError(
            f"handle blob in {file_path} IPC-shares {leaked}, which this "
            "policy keeps replica-private; refusing to alias per-request state"
        )
    try:
        values = {
            n: _value_bytes_to_tensor(b) for n, b in payload["value_blobs"].items()
        }
    except Exception as exc:
        raise WeightShareError(
            f"failed to decode by-value tensors from {file_path}: {exc}"
        ) from exc
    extra = sorted((set(shared) | set(values)) - set(own_tensors))
    if extra:
        raise WeightShareError(
            f"handle blob in {file_path} carries tensors this model does not "
            f"register (first 5): {extra[:5]}"
        )
    doubled = sorted(set(shared) & set(values))
    if doubled:
        raise WeightShareError(
            f"handle blob in {file_path} lists {doubled[:5]} both as IPC and "
            "by-value; the export is malformed"
        )

    params = dict(model.named_parameters())
    # Note (Jiaxin Deng): a buffer can be registered under several module paths,
    # so rebinding must hit every one or a stale holder keeps dummy storage.
    buffer_paths_by_id: dict[int, list[str]] = {}
    for dotted, buf in model.named_buffers(remove_duplicate=False):
        buffer_paths_by_id.setdefault(id(buf), []).append(dotted)
    private_ptrs = {n: own_tensors[n].data_ptr() for n in private}
    aliased_bytes = 0
    for name, own in own_tensors.items():
        if name in shared:
            incoming = shared[name]
        elif name in values:
            incoming = values[name]
        else:
            # Note (Jiaxin Deng): unreachable while the manifest hash matches;
            # kept so a hash bug cannot degrade into a silent skip.
            raise WeightShareError(f"leader export is missing tensor {name!r}")
        if not isinstance(incoming, torch.Tensor):
            raise WeightShareError(
                f"handle blob in {file_path} entry {name!r} is not a tensor"
            )
        if tuple(incoming.shape) != tuple(own.shape) or incoming.dtype != own.dtype:
            raise WeightShareError(
                f"tensor {name!r} mismatch: leader "
                f"{tuple(incoming.shape)}/{incoming.dtype}, follower "
                f"{tuple(own.shape)}/{own.dtype}"
            )
        if incoming.is_cuda and incoming.device != own.device:
            # Note (Jiaxin Deng): a UUID-remap failure would otherwise surface
            # later as a mixed-device kernel error that never names weight sharing.
            raise WeightShareError(
                f"tensor {name!r} arrived on {incoming.device}, expected "
                f"{own.device}: CUDA IPC handle mapped to the wrong device"
            )
        if name in shared:
            if name in params:
                # Note (Jiaxin Deng): tied params are one Parameter object, so a
                # single .data assignment propagates to every module holding it.
                params[name].data = incoming
            else:
                for dotted in buffer_paths_by_id.get(id(own), [name]):
                    _rebind_buffer(model, dotted, incoming)
            aliased_bytes += incoming.numel() * incoming.element_size()

    # Note (Jiaxin Deng): a tensor registered under both a shared and a private
    # name would have moved with the shared rebinds above; copying into it now
    # would write per-request state into the leader's storage, so check first.
    moved_check = _named_shared_tensors(model)
    for name in sorted(private):
        current = moved_check.get(name)
        if (
            own_tensors[name].data_ptr() != private_ptrs[name]
            or current is None
            or current.data_ptr() != private_ptrs[name]
        ):
            raise WeightShareError(
                f"private tensor {name!r} storage moved during shared attach; "
                "it is cross-registered with a shared tensor and cannot stay "
                "replica-private"
            )

    for name, own in own_tensors.items():
        if name not in shared:
            # Note (Jiaxin Deng): replica-private and non-CUDA tensors are
            # copied into the follower's own storage, never aliased, so the
            # address its CUDA graphs capture stays this replica's.
            with torch.no_grad():
                own.copy_(values[name].to(own.device))

    record: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    for name, tensor in _named_shared_tensors(model).items():
        if name in shared:
            expected = shared[name]
            if tensor.data_ptr() != expected.data_ptr():
                raise WeightShareError(
                    f"aliasing failed for {name!r}: module tensor does not "
                    "point at the shared storage after attach"
                )
            record[name] = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        elif name in private:
            # Note (Jiaxin Deng): recorded so verify_attachment also catches a
            # private tensor rebound (e.g. onto shared storage) after attach.
            record[name] = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
    private_count = sum(1 for name in record if name in private)
    logger.info(
        f"[weight-share] follower attached {len(record) - private_count} shared "
        f"tensors ({aliased_bytes / (1 << 30):.2f} GiB aliased, zero-copy) + "
        f"{private_count} replica-private by-value from {file_path}"
    )
    return record


def _rebind_buffer(
    model: torch.nn.Module, dotted_name: str, tensor: torch.Tensor
) -> None:
    module_path, _, leaf = dotted_name.rpartition(".")
    module = model.get_submodule(module_path) if module_path else model
    if leaf not in module._buffers:
        raise WeightShareError(
            f"{dotted_name!r} is not a registered buffer on the follower model"
        )
    module._buffers[leaf] = tensor


def _raise_manifest_mismatch(
    model: torch.nn.Module,
    payload: dict[str, Any],
    own_tensors: dict[str, torch.Tensor],
    file_path: str,
) -> None:
    leader_names = set(payload["ipc_names"]) | set(payload["value_blobs"])
    own_names = set(own_tensors)
    missing = sorted(own_names - leader_names)[:5]
    extra = sorted(leader_names - own_names)[:5]
    raise WeightShareError(
        f"weight manifest mismatch between leader export {file_path} and this "
        f"follower ({type(model).__name__}): the replicas are not running "
        f"identical models/configs. follower-only names (first 5): {missing}; "
        f"leader-only names (first 5): {extra}. If names match, a shape/dtype "
        "differs — compare replica configs."
    )


def verify_attachment(
    model: torch.nn.Module,
    record: dict[str, tuple[int, tuple[int, ...], torch.dtype]],
) -> None:
    """Assert attached tensors were not re-initialized since attach.

    Call this immediately before warmup/CUDA-graph capture: any load-path
    step that re-created a parameter after attach (breaking the alias) turns
    into a hard error here instead of silently serving dummy weights.
    """
    current = _named_shared_tensors(model)
    drifted = []
    for name, (ptr, shape, dtype) in record.items():
        tensor = current.get(name)
        if tensor is None:
            drifted.append(f"{name} (tensor disappeared)")
        elif tensor.data_ptr() != ptr:
            drifted.append(f"{name} (storage rebound after attach)")
        elif tuple(tensor.shape) != shape or tensor.dtype != dtype:
            drifted.append(f"{name} (shape/dtype changed after attach)")
    if drifted:
        raise WeightShareError(
            "weight-share attachment was broken after attach; offending "
            f"tensors (first 10): {drifted[:10]}"
        )


def leader_export(
    model: torch.nn.Module,
    dir_path: str,
    *,
    serializer: Any | None = None,
    validate_secure: bool = True,
    model_path: str | None = None,
    model_revision: str | None = None,
    run_id: str | None = None,
    private_names: frozenset[str] = frozenset(),
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Leader role: export this engine's weights under dir_path."""
    return export_weights(
        model,
        handle_file_for_model(dir_path, model),
        serializer=serializer,
        validate_secure=validate_secure,
        model_path=model_path,
        model_revision=model_revision,
        run_id=run_id,
        private_names=private_names,
    )


def follower_attach(
    model: torch.nn.Module,
    dir_path: str,
    *,
    timeout_s: float = DEFAULT_ATTACH_TIMEOUT_S,
    serializer: Any | None = None,
    validate_secure: bool = True,
    model_path: str | None = None,
    model_revision: str | None = None,
    run_id: str | None = None,
    private_names: frozenset[str] = frozenset(),
) -> tuple[dict[str, tuple[int, tuple[int, ...], torch.dtype]], LeaderLivenessMonitor]:
    """Follower role: attach this engine's weights and watch the leader.

    Returns the attachment record and a started LeaderLivenessMonitor; the
    caller keeps a reference so the follower exits if the leader dies.
    """
    record, payload = _attach_and_check(
        model,
        handle_file_for_model(dir_path, model),
        timeout_s=timeout_s,
        poll_interval_s=0.5,
        serializer=serializer,
        validate_secure=validate_secure,
        model_path=model_path,
        model_revision=model_revision,
        run_id=run_id,
        private_names=private_names,
    )
    monitor = LeaderLivenessMonitor(
        int(payload.get("pid") or 0),
        leader_start_time=payload.get("leader_start_time"),
    )
    monitor.start()
    return record, monitor
