# SPDX-License-Identifier: Apache-2.0
"""HF-parity patch for sglang ``Qwen3VLMoeVisionModel.rot_pos_emb``.

Patches ``rot_pos_emb`` to compute cos/sin in the model dtype, matching HF
transformers' semantics. See sglang-omni issue #434 for context.

Note:(Chenchen Hong) the former ``fast_pos_embed_interpolate`` patch was
dropped on sglang 0.5.12.post1: upstream's native implementation produces
output bit-identical to the patch (verified on Qwen3-Omni vs HF at 4x4 and
32x32 grids, cos identical to 6 digits), so the patch was a no-op.
"""
from __future__ import annotations

import inspect
import logging
import re
import threading
from typing import Sequence

import torch

logger = logging.getLogger(__name__)

_PATCHED_FLAG = "_sglang_omni_hf_parity_patched"

# Serialize first-call apply so two concurrent callers cannot observe a
# half-patched class.
_APPLY_LOCK = threading.Lock()

# Released sglang version whose Qwen3VLMoeVisionModel layout and dependency
# pins match the patch. Dev builds (``0.0.0.dev1+...``) are accepted with a
# warning since they are unversioned snapshots that may or may not match.
_SUPPORTED_SGLANG_VERSIONS: frozenset[str] = frozenset({"0.5.12.post1"})

# Instance attributes the patch reads; preflight checks each is present
# either as a class-level descriptor or as a self.<name> = assignment in
# any parent's __init__.
_REQUIRED_INSTANCE_ATTRS: Sequence[str] = (
    "spatial_merge_size",
    "rotary_pos_emb",
    "rot_pos_ids",
    "dtype",
    "device",
)
# Method the patch replaces.
_REQUIRED_METHODS: Sequence[str] = ("rot_pos_emb",)


def _patched_rot_pos_emb(self, grid_thw):
    """Returns ``(cos, sin)`` at half-dim ``(N, head_dim/2)`` (block cats internally).

    Computes cos/sin in ``self.dtype`` rather than reading fp32 cache.
    """
    pos_ids = []
    for t, h, w in grid_thw:
        base = self.rot_pos_ids(h, w, self.spatial_merge_size)
        pos_ids.append(base if t == 1 else base.repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
    max_grid_size = max(max(h, w) for _, h, w in grid_thw)

    rope = self.rotary_pos_emb
    inv_freq = rope._compute_inv_freq(rope.base).to(
        device=self.device, dtype=self.dtype
    )
    seq = torch.arange(max_grid_size, device=self.device, dtype=self.dtype)
    freq_table = torch.outer(seq, inv_freq)
    embeddings = freq_table[pos_ids].flatten(1)
    return embeddings.cos(), embeddings.sin()


def _check_target_class_surface(cls) -> None:
    """Fail-fast if sglang renamed attrs / methods / helper signatures."""
    missing_methods = [m for m in _REQUIRED_METHODS if not hasattr(cls, m)]
    if missing_methods:
        raise RuntimeError(
            f"sglang {cls.__module__}.{cls.__name__} missing methods: "
            f"{missing_methods}"
        )

    init_sources: list[str] = []
    for klass in cls.__mro__:
        if klass is object:
            break
        try:
            init_sources.append(inspect.getsource(klass.__init__))
        except (TypeError, OSError):
            # built-in or source unavailable — skip
            continue
    combined_init_src = "\n".join(init_sources)

    missing_attrs: list[str] = []
    for attr in _REQUIRED_INSTANCE_ATTRS:
        if hasattr(cls, attr):
            continue
        if f"self.{attr}" in combined_init_src:
            continue
        missing_attrs.append(attr)

    if missing_attrs:
        raise RuntimeError(
            f"sglang {cls.__module__}.{cls.__name__} missing attrs: " f"{missing_attrs}"
        )

    # The patches call ``self.rot_pos_ids(h, w, spatial_merge_size)`` and
    # ``rope._compute_inv_freq(rope.base)``; an upstream rename of any of
    # these positional parameters would be a silent semantic break.
    _check_signature(
        cls, "rot_pos_ids", expected_names=("h", "w", "spatial_merge_size")
    )

    try:
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    except ImportError as exc:
        raise RuntimeError("sglang RotaryEmbedding import failed") from exc

    _check_signature(RotaryEmbedding, "_compute_inv_freq", expected_names=("base",))


def _check_signature(cls, name: str, *, expected_names: tuple[str, ...]) -> None:
    """Verify ``cls.name`` accepts ``expected_names`` as leading positional args.

    ``self`` is dropped before comparison. Built-in / unintrospectable callables
    are skipped (we cannot reason about their signatures and refusing to patch
    on that basis would be over-strict).
    """
    fn = getattr(cls, name, None)
    if not callable(fn):
        raise RuntimeError(
            f"sglang {cls.__module__}.{cls.__name__}.{name} missing or not callable"
        )
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if positional and positional[0].name == "self":
        positional = positional[1:]
    actual_names = tuple(p.name for p in positional[: len(expected_names)])
    if actual_names != expected_names:
        raise RuntimeError(
            f"sglang {cls.__module__}.{cls.__name__}.{name} signature changed: "
            f"expected leading params {expected_names}, got {actual_names} (full sig: {sig})"
        )


# Matches PEP 440 ``.devN`` markers in the public version segment only.
# Strip the local-version suffix (``+...``) before matching so legitimate
# release+local strings like ``0.5.9+foo.dev1`` aren't misclassified.
_DEV_VERSION_RE = re.compile(r"\.dev\d*$")


def _is_dev_version(version: str) -> bool:
    public = version.split("+", 1)[0]
    return bool(_DEV_VERSION_RE.search(public))


def _check_sglang_version() -> None:
    """Reject sglang versions outside the validated set; warn for dev builds."""
    try:
        import sglang
    except ImportError as exc:
        raise RuntimeError(
            "sglang import failed; cannot apply HF-parity patches."
        ) from exc

    version = getattr(sglang, "__version__", None)
    if version is None or _is_dev_version(version):
        logger.warning(
            "sglang version %r is a dev build; assuming layout is "
            "compatible with %s. Pin to a supported release if you see "
            "parity drift.",
            version,
            sorted(_SUPPORTED_SGLANG_VERSIONS),
        )
        return

    if version not in _SUPPORTED_SGLANG_VERSIONS:
        raise RuntimeError(
            f"sglang=={version} is not in the patch's supported version "
            f"set {sorted(_SUPPORTED_SGLANG_VERSIONS)}. Either pin sglang "
            f"to a supported version, or update / remove this patch module."
        )


def apply_qwen3_vl_hf_parity_patches() -> None:
    """Apply the ``rot_pos_emb`` HF-parity patch to ``Qwen3VLMoeVisionModel``.

    Idempotent (class-level flag) and threadsafe (module-level lock).
    Raises ``RuntimeError`` on unsupported sglang version or class-surface drift.
    """
    _check_sglang_version()

    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

    # Fast path without acquiring the lock.
    if getattr(Qwen3VLMoeVisionModel, _PATCHED_FLAG, False):
        return

    with _APPLY_LOCK:
        # Re-check under the lock — another thread may have applied while
        # we were waiting.
        if getattr(Qwen3VLMoeVisionModel, _PATCHED_FLAG, False):
            return

        _check_target_class_surface(Qwen3VLMoeVisionModel)

        Qwen3VLMoeVisionModel.rot_pos_emb = _patched_rot_pos_emb
        setattr(Qwen3VLMoeVisionModel, _PATCHED_FLAG, True)

    logger.info(
        f"Applied rot_pos_emb HF-parity patch to "
        f"{Qwen3VLMoeVisionModel.__module__}.{Qwen3VLMoeVisionModel.__name__} "
        f"(see sglang-omni issue #434 for context).",
    )
