# SPDX-License-Identifier: Apache-2.0
"""Checkpoint ownership utilities for Ming-Omni-TTS."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

OWNER_AR_MODEL = "ar_model"
OWNER_TTS_HEADS = "tts_heads"
OWNER_AUDIO_VAE = "audio_vae"
OWNER_INTENTIONAL_SKIP = "intentional_skip"
OWNER_UNKNOWN = "unknown"

MING_TTS_AUDIO_PREFIX = "audio."
MING_TTS_LM_HEAD_PREFIX = "model.lm_head."
MING_TTS_LM_HEAD_SKIP_REASON = "hidden-only TTS path"
MING_TTS_ROTARY_BUFFER_SKIP_REASON = "runtime rotary buffer"

_OWNERS = (
    OWNER_AR_MODEL,
    OWNER_TTS_HEADS,
    OWNER_AUDIO_VAE,
    OWNER_INTENTIONAL_SKIP,
    OWNER_UNKNOWN,
)
_AR_PREFIXES = ("model.model.", "word_embeddings.", "layers.", "norm.")
_TTS_HEAD_PREFIXES = (
    "linear_proj_audio.",
    "flowloss.",
    "stop_head.",
    "spk_head.",
)
_INDEX_FILENAME = "model.safetensors.index.json"
_SINGLE_FILENAME = "model.safetensors"
_SAMPLE_LIMIT = 8


@dataclass(frozen=True)
class MingTTSWeightManifest:
    """Index-only view of a Ming-Omni-TTS composite checkpoint."""

    model_path: str
    source: str
    total_tensors: int
    total_size_bytes: int
    prefix_counts: dict[str, int]
    keys_by_owner: dict[str, list[str]]
    shards_by_owner: dict[str, list[str]]
    weight_map: dict[str, str]

    def owner_count(self, owner: str) -> int:
        return len(self.keys_by_owner.get(owner, ()))

    def owner_counts(self) -> dict[str, int]:
        return {owner: self.owner_count(owner) for owner in _OWNERS}

    def unknown_keys(self) -> list[str]:
        return list(self.keys_by_owner.get(OWNER_UNKNOWN, ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "source": self.source,
            "total_tensors": int(self.total_tensors),
            "total_size_bytes": int(self.total_size_bytes),
            "prefix_counts": dict(self.prefix_counts),
            "owner_counts": self.owner_counts(),
            "shards_by_owner": {
                owner: list(shards) for owner, shards in self.shards_by_owner.items()
            },
            "samples_by_owner": {
                owner: list(keys[:_SAMPLE_LIMIT])
                for owner, keys in self.keys_by_owner.items()
            },
        }


@dataclass
class MingTTSWeightReport:
    """Strict-load report for checkpoint ownership and coverage diagnostics."""

    manifest: MingTTSWeightManifest | None = None
    loaded: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, list[str]] = field(default_factory=dict)
    deferred: dict[str, list[str]] = field(default_factory=dict)
    leftovers: list[str] = field(default_factory=list)
    missing: dict[str, list[str]] = field(default_factory=dict)
    loaded_keys: dict[str, list[str]] = field(default_factory=dict)
    loaded_params: dict[str, list[str]] = field(default_factory=dict)
    loaded_shards: dict[str, list[str]] = field(default_factory=dict)
    required_shards: dict[str, list[str]] = field(default_factory=dict)

    def add_loaded(
        self,
        owner: str,
        checkpoint_key: str,
        *,
        target_param: str | None = None,
    ) -> None:
        self.loaded[owner] = self.loaded.get(owner, 0) + 1
        self.loaded_keys.setdefault(owner, []).append(checkpoint_key)
        if target_param is not None:
            self.loaded_params.setdefault(owner, []).append(target_param)

    def add_loaded_shard(self, target_param: str, shard: str | int) -> None:
        bucket = self.loaded_shards.setdefault(target_param, [])
        shard_id = str(shard)
        if shard_id not in bucket:
            bucket.append(shard_id)

    def add_required_shards(
        self,
        target_param: str,
        shards: Iterable[str | int],
    ) -> None:
        bucket = self.required_shards.setdefault(target_param, [])
        for shard in shards:
            shard_id = str(shard)
            if shard_id not in bucket:
                bucket.append(shard_id)

    def to_dict(self) -> dict[str, Any]:
        def bucket_summary(buckets: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
            return {
                key: {
                    "count": len(values),
                    "sample": list(values[:_SAMPLE_LIMIT]),
                }
                for key, values in buckets.items()
                if values
            }

        def packed_shard_summary(
            loaded_shards: dict[str, list[str]],
            required_shards: dict[str, list[str]],
        ) -> dict[str, dict[str, Any]]:
            summary = {}
            for target, required in required_shards.items():
                loaded = loaded_shards.get(target, [])
                missing = sorted(set(required) - set(loaded))
                if missing:
                    summary[target] = {
                        "loaded_count": len(loaded),
                        "required_count": len(required),
                        "loaded_sample": list(loaded[:_SAMPLE_LIMIT]),
                        "missing_sample": list(missing[:_SAMPLE_LIMIT]),
                        "missing_count": len(missing),
                    }
            return summary

        return {
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "loaded": dict(self.loaded),
            "loaded_keys": bucket_summary(self.loaded_keys),
            "loaded_params": bucket_summary(self.loaded_params),
            "skipped": bucket_summary(self.skipped),
            "deferred": bucket_summary(self.deferred),
            "leftover_count": len(self.leftovers),
            "leftovers": list(self.leftovers[:_SAMPLE_LIMIT]),
            "missing": bucket_summary(self.missing),
            "packed_shards": packed_shard_summary(
                self.loaded_shards,
                self.required_shards,
            ),
        }

    def summary(self) -> str:
        def sample(keys: list[str]) -> list[str]:
            return list(keys[:_SAMPLE_LIMIT])

        def format_size(size_bytes: int) -> str:
            if size_bytes <= 0:
                return f"{size_bytes}B"
            size = float(size_bytes)
            for unit in ("B", "KiB", "MiB", "GiB"):
                if size < 1024.0 or unit == "GiB":
                    return f"{size:.2f}{unit}"
                size /= 1024.0
            return f"{size_bytes}B"

        lines = ["Ming-TTS weight coverage:"]
        if self.manifest is not None:
            total_size = format_size(self.manifest.total_size_bytes)
            shard_count = len(set(self.manifest.weight_map.values()))
            lines.append(
                "  total_tensors="
                f"{self.manifest.total_tensors} total_size={total_size} "
                f"shards={shard_count}"
            )
            for owner in _OWNERS:
                total = self.manifest.owner_count(owner)
                if total:
                    loaded = self.loaded.get(owner, 0)
                    deferred = len(self.deferred.get(owner, ()))
                    detail = f"loaded={loaded}"
                    if deferred:
                        detail += f" deferred={deferred}"
                    lines.append(f"  {owner}: {detail} total={total}")
        else:
            for owner, count in sorted(self.loaded.items()):
                lines.append(f"  {owner}: loaded={count}")

        for reason, keys in sorted(self.skipped.items()):
            if not keys:
                continue
            lines.append(
                f"  skipped: count={len(keys)} reason={reason} "
                f"sample={sample(keys)}"
            )
        for owner, keys in sorted(self.deferred.items()):
            if not keys:
                continue
            lines.append(
                f"  deferred: owner={owner} count={len(keys)} sample={sample(keys)}"
            )
        if self.missing:
            for owner, keys in sorted(self.missing.items()):
                lines.append(
                    f"  missing: owner={owner} count={len(keys)} "
                    f"sample={sample(keys)}"
                )
        lines.append(f"  leftovers={len(self.leftovers)}")
        if self.leftovers:
            lines.append(f"  leftover_sample={sample(self.leftovers)}")
        if self.required_shards:
            incomplete = _incomplete_packed_shards(
                self.loaded_shards,
                self.required_shards,
            )
            lines.append(
                f"  packed_targets={len(self.required_shards)} "
                f"incomplete={len(incomplete)}"
            )
            if incomplete:
                target, missing = next(iter(incomplete.items()))
                lines.append(f"  packed_missing_sample={target}: {sample(missing)}")
        return "\n".join(lines)


def classify_ming_tts_weight(name: str) -> str:
    """Return the owner bucket for a raw Ming-Omni-TTS checkpoint key."""

    if name.startswith(MING_TTS_LM_HEAD_PREFIX):
        return OWNER_INTENTIONAL_SKIP
    if name.startswith(_AR_PREFIXES):
        return OWNER_AR_MODEL
    if name.startswith(_TTS_HEAD_PREFIXES):
        return OWNER_TTS_HEADS
    if name.startswith(MING_TTS_AUDIO_PREFIX):
        return OWNER_AUDIO_VAE
    return OWNER_UNKNOWN


def scan_ming_tts_weights(
    model_path: str | Path,
    *,
    local_files_only: bool = False,
) -> MingTTSWeightManifest:
    """Scan checkpoint metadata without materializing checkpoint tensors."""

    from sglang_omni.models.weight_loader import resolve_model_path

    resolved_path = resolve_model_path(
        str(model_path),
        local_files_only=local_files_only,
    )

    index_path = resolved_path / _INDEX_FILENAME
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = dict(index.get("weight_map") or {})
        total_size = int((index.get("metadata") or {}).get("total_size") or 0)
        source = str(index_path)
    else:
        single_path = resolved_path / _SINGLE_FILENAME
        if not single_path.exists():
            raise FileNotFoundError(
                "Ming-Omni-TTS expects model.safetensors.index.json or "
                f"model.safetensors under {resolved_path}"
            )
        from safetensors import safe_open

        with safe_open(str(single_path), framework="pt", device="cpu") as handle:
            weight_map = {key: _SINGLE_FILENAME for key in handle.keys()}
        total_size = int(single_path.stat().st_size)
        source = str(single_path)

    if not weight_map:
        raise FileNotFoundError(f"No checkpoint tensors found under {resolved_path}")

    prefix_counts: dict[str, int] = {}
    keys_by_owner: dict[str, list[str]] = {owner: [] for owner in _OWNERS}
    shard_sets: dict[str, set[str]] = {owner: set() for owner in _OWNERS}
    for key, shard in weight_map.items():
        owner = classify_ming_tts_weight(key)
        keys_by_owner[owner].append(key)
        shard_sets[owner].add(shard)
        if key.startswith("model.model."):
            prefix = "model.model"
        elif key.startswith(MING_TTS_LM_HEAD_PREFIX):
            prefix = "model.lm_head"
        elif key.startswith(MING_TTS_AUDIO_PREFIX):
            prefix = "audio"
        else:
            prefix = key.split(".", 1)[0]
            for tts_prefix in _TTS_HEAD_PREFIXES:
                if key.startswith(tts_prefix):
                    prefix = tts_prefix[:-1]
                    break
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    return MingTTSWeightManifest(
        model_path=str(resolved_path),
        source=source,
        total_tensors=len(weight_map),
        total_size_bytes=total_size,
        prefix_counts=dict(sorted(prefix_counts.items())),
        keys_by_owner=keys_by_owner,
        shards_by_owner={
            owner: sorted(shards) for owner, shards in shard_sets.items() if shards
        },
        weight_map=weight_map,
    )


def load_ming_tts_audio_vae_weights(
    model_path: str | Path,
    audio_vae: Any,
    *,
    local_files_only: bool = False,
) -> MingTTSWeightReport:
    """Load the composite checkpoint's audio.* tensors into AudioVAE."""

    manifest = scan_ming_tts_weights(
        model_path,
        local_files_only=local_files_only,
    )
    report = MingTTSWeightReport(
        manifest=manifest,
        loaded={OWNER_AUDIO_VAE: 0},
        skipped={
            MING_TTS_LM_HEAD_SKIP_REASON: manifest.keys_by_owner[
                OWNER_INTENTIONAL_SKIP
            ],
        },
        deferred={
            OWNER_AR_MODEL: manifest.keys_by_owner[OWNER_AR_MODEL],
            OWNER_TTS_HEADS: manifest.keys_by_owner[OWNER_TTS_HEADS],
        },
        leftovers=manifest.unknown_keys(),
    )
    assert_ming_tts_weight_coverage(report)

    if manifest.owner_count(OWNER_AUDIO_VAE) <= 0:
        raise FileNotFoundError(
            f"No {MING_TTS_AUDIO_PREFIX} tensors found under {manifest.model_path}"
        )

    from safetensors import safe_open

    keys_by_shard: dict[str, list[str]] = {}
    for key in manifest.keys_by_owner[OWNER_AUDIO_VAE]:
        keys_by_shard.setdefault(manifest.weight_map[key], []).append(key)

    state_dict = {}
    checkpoint_keys_by_target: dict[str, str] = {}
    resolved_path = Path(manifest.model_path)
    for shard, shard_keys in sorted(keys_by_shard.items()):
        shard_path = resolved_path / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                output_key = key
                if key.startswith(MING_TTS_AUDIO_PREFIX):
                    output_key = key[len(MING_TTS_AUDIO_PREFIX) :]
                state_dict[output_key] = handle.get_tensor(key)
                checkpoint_keys_by_target[output_key] = key

    if not state_dict:
        raise FileNotFoundError(
            f"No {MING_TTS_AUDIO_PREFIX} tensors found under {manifest.model_path}"
        )

    incompatible = audio_vae.load_state_dict(state_dict, strict=False)
    missing = [str(key) for key in getattr(incompatible, "missing_keys", ())]
    unexpected = [str(key) for key in getattr(incompatible, "unexpected_keys", ())]
    for target_key in state_dict:
        report.add_loaded(
            OWNER_AUDIO_VAE,
            checkpoint_keys_by_target[target_key],
            target_param=target_key,
        )
    if missing:
        report.missing.setdefault(OWNER_AUDIO_VAE, []).extend(missing)
    if unexpected:
        report.leftovers.extend(unexpected)
    assert_ming_tts_weight_coverage(report)

    audio_vae.eval()
    audio_vae._ming_tts_weight_report = report
    return report


def assert_ming_tts_weight_coverage(report: MingTTSWeightReport) -> None:
    def format_key_count(keys: list[str]) -> str:
        sample = ", ".join(keys[:_SAMPLE_LIMIT])
        if len(keys) > _SAMPLE_LIMIT:
            sample += f", ... ({len(keys)} total)"
        return sample

    errors = []
    if report.leftovers:
        errors.append("leftover weights: " f"{format_key_count(report.leftovers)}")
    for owner, keys in sorted(report.missing.items()):
        if keys:
            errors.append(f"missing {owner} weights: {format_key_count(keys)}")
    for target, missing in _incomplete_packed_shards(
        report.loaded_shards,
        report.required_shards,
    ).items():
        errors.append(
            f"incomplete packed weight {target}: missing {format_key_count(missing)}"
        )
    if errors:
        raise RuntimeError("Ming-Omni-TTS weight coverage failed: " + "; ".join(errors))


def _incomplete_packed_shards(
    loaded_shards: dict[str, list[str]],
    required_shards: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        target: sorted(set(required) - set(loaded_shards.get(target, ())))
        for target, required in sorted(required_shards.items())
        if set(required) - set(loaded_shards.get(target, ()))
    }


__all__ = [
    "MING_TTS_AUDIO_PREFIX",
    "MING_TTS_LM_HEAD_PREFIX",
    "MING_TTS_LM_HEAD_SKIP_REASON",
    "MING_TTS_ROTARY_BUFFER_SKIP_REASON",
    "MingTTSWeightManifest",
    "MingTTSWeightReport",
    "OWNER_AR_MODEL",
    "OWNER_AUDIO_VAE",
    "OWNER_INTENTIONAL_SKIP",
    "OWNER_TTS_HEADS",
    "OWNER_UNKNOWN",
    "assert_ming_tts_weight_coverage",
    "classify_ming_tts_weight",
    "load_ming_tts_audio_vae_weights",
    "scan_ming_tts_weights",
]
