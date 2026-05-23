# SPDX-License-Identifier: Apache-2.0
"""SeedTTS speaker-similarity asset bootstrapper.

Single source of truth for the two HuggingFace files that
:class:`benchmarks.metrics.speaker_similarity.WavLMSpeakerSimilarity` needs:

- ``wavlm_large_finetune.pth`` — fine-tuned WavLM SV head
  (``popsoda2002/seedtts-wavlm-sim``)
- ``wavlm_large.pt`` — WavLM base weights
  (``s3prl/converted_ckpts``)

The s3prl Python package itself is consumed as a regular PyPI dependency
(``s3prl>=0.4.18`` in ``pyproject.toml``).  Earlier versions of this PR
also git-cloned the s3prl repository to read ``s3prl.upstream.wavlm.hubconf``
off-tree; the pip-installed package ships the identical module, so the
clone has been removed (per @zhaochenyang20's PR #469 review).

The bootstrapper is consumed by exactly one runtime call site
(:func:`benchmarks.tasks.tts.run_seedtts_similarity`) so the CI workflows
and the user-facing ``--similarity-only`` entry points share one code path
for asset preparation.

Usage from CI / scripts:

    python -m benchmarks.metrics.speaker_similarity_assets --warm-cache

Usage from Python:

    assets = ensure_speaker_similarity_assets()
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device="cuda:0",
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# HuggingFace sources — kept here so anyone changing the asset provenance
# touches one constant rather than chasing strings through CI yaml.
_FINETUNE_REPO_ID = "popsoda2002/seedtts-wavlm-sim"
_FINETUNE_FILENAME = "wavlm_large_finetune.pth"
_WAVLM_BASE_REPO_ID = "s3prl/converted_ckpts"
_WAVLM_BASE_FILENAME = "wavlm_large.pt"

# Minimum acceptable file size in bytes, used as a fast sanity check before
# trusting cached files (catches truncated downloads / empty placeholders
# without paying the cost of recomputing a checksum on every HIT). The real
# file sizes are recorded into the marker at download time and re-verified
# on cache HIT, so an HF-side asset version change with a different size
# also invalidates the cache automatically.
_MIN_ASSET_SIZE_BYTES = 100 * 1024 * 1024  # 100 MiB

# JSON cache marker: written atomically only after every asset passes
# validation. Carries enough provenance metadata (schema version, repo_id,
# filename, size) that an upstream asset replacement or a local on-disk
# corruption is detected on the next call. Schema bumps invalidate older
# markers (consumers of the cache rebuild from scratch).
_MARKER_FILENAME = ".complete"
_MARKER_SCHEMA_VERSION = 2

_CACHE_DIR_ENV = "SEEDTTS_SIM_CACHE_DIR"


@dataclass(frozen=True)
class SpeakerSimilarityAssets:
    """Resolved on-disk paths for the SeedTTS speaker-similarity scorer."""

    finetune_checkpoint: Path
    wavlm_base: Path


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    """Pick the cache directory in priority order: arg → env → user cache."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    env_value = os.environ.get(_CACHE_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("~/.cache/sglang-omni/speaker_sim").expanduser().resolve()


def _hf_download(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Download ``filename`` from ``repo_id`` into ``dest_dir``.

    Uses the ``huggingface_hub`` Python API (not the ``huggingface-cli``
    subprocess) so progress, errors, and ``HF_ENDPOINT`` mirror handling are
    all in-process.  ``hf_hub_download`` already validates the downloaded
    file against the HF-side metadata before returning, so a transport
    truncation surfaces here rather than as a silent partial cache.
    """
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )
    return Path(local_path)


def _read_marker(marker: Path) -> dict | None:
    """Parse the cache marker JSON, returning ``None`` on any error.

    Treats any unreadable / malformed / wrong-schema marker as invalid so
    the caller falls through to re-validate and re-download.
    """
    try:
        data = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != _MARKER_SCHEMA_VERSION:
        return None
    return data


def _write_marker(marker: Path, files_info: dict[str, dict]) -> None:
    """Atomically write the cache marker with per-file provenance."""
    payload = {
        "schema_version": _MARKER_SCHEMA_VERSION,
        "files": files_info,
    }
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(marker)


def _validate_asset(
    path: Path,
    *,
    expected_size: int | None = None,
    expected_repo_id: str | None = None,
    recorded_repo_id: str | None = None,
) -> tuple[bool, str]:
    """Validate a single asset file, returning ``(ok, reason)``.

    A file is considered valid only if it exists, exceeds the minimum
    plausible size, matches the size that was recorded into the marker
    when it was downloaded (catches truncation or upstream-side asset
    replacement), and was originally fetched from the repo that the
    bootstrapper is currently configured to use.
    """
    if not path.is_file():
        return False, f"not on disk: {path}"
    actual_size = path.stat().st_size
    if actual_size < _MIN_ASSET_SIZE_BYTES:
        return False, (
            f"size {actual_size} below minimum {_MIN_ASSET_SIZE_BYTES} "
            f"({path}) — likely truncated"
        )
    if expected_size is not None and actual_size != expected_size:
        return False, (
            f"size {actual_size} does not match marker-recorded {expected_size} "
            f"({path}) — upstream asset may have changed or local file is corrupt"
        )
    if (
        expected_repo_id is not None
        and recorded_repo_id is not None
        and recorded_repo_id != expected_repo_id
    ):
        return False, (
            f"recorded repo_id {recorded_repo_id!r} does not match expected "
            f"{expected_repo_id!r} for {path.name}"
        )
    return True, "ok"


def _migrate_legacy_layout(cache_dir: Path) -> None:
    """One-time migration of pre-flat layouts into the current flat cache.

    PR #469's earlier CI shell block put ``wavlm_large.pt`` under
    ``cache_dir/s3prl/`` so it could sit next to a git-cloned ``s3prl``
    source tree. The bootstrapper now expects a flat layout — both HF
    files sit directly in ``cache_dir``. On first run after this change
    against an existing CI cache, move the legacy file into place so the
    ~1.2 GB asset is *not* re-downloaded over a flaky proxy. No-op when
    no legacy layout is present.
    """
    legacy = cache_dir / "s3prl" / _WAVLM_BASE_FILENAME
    flat = cache_dir / _WAVLM_BASE_FILENAME
    if not legacy.is_file() or flat.exists():
        return
    if legacy.stat().st_size < _MIN_ASSET_SIZE_BYTES:
        return
    logger.info("[sim-assets] migrating legacy %s -> %s", legacy, flat)
    try:
        legacy.rename(flat)
    except OSError:
        # Cross-filesystem rename — fall back to copy + unlink.
        import shutil

        shutil.move(str(legacy), str(flat))


def _download_and_validate(
    repo_id: str,
    filename: str,
    cache_dir: Path,
) -> Path:
    """Fetch ``filename`` from ``repo_id`` and require it to pass validation.

    If the freshly downloaded file is below the minimum size it is removed
    and the error is raised, so a broken download cannot land in the cache
    and the next call will retry from scratch.
    """
    path = _hf_download(repo_id, filename, cache_dir)
    ok, reason = _validate_asset(path)
    if not ok:
        try:
            path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"[sim-assets] freshly downloaded {repo_id}/{filename} failed "
            f"validation: {reason}"
        )
    return path


def ensure_speaker_similarity_assets(
    cache_dir: Path | None = None,
    finetune_checkpoint_override: Path | None = None,
) -> SpeakerSimilarityAssets:
    """Make sure the WavLM SV scorer's two asset files are on disk.

    Resolution rules:

    - ``cache_dir`` is resolved in priority order:
      explicit arg → ``SEEDTTS_SIM_CACHE_DIR`` env → ``~/.cache/sglang-omni/speaker_sim``.
    - If ``finetune_checkpoint_override`` is provided (typically the user
      passing ``--similarity-checkpoint``), it is used as-is for the fine-tune
      head and is **not** re-downloaded.  The WavLM base file is still
      ensured under ``cache_dir``.
    - Cache validity is *not* "the marker file exists".  The marker is a
      JSON record of which (repo_id, filename, size) tuples were validated
      at the time it was written; on every call we re-confirm that each
      cached file still passes the recorded size and minimum-size checks
      and that the repo_id we are configured to fetch from has not changed.
      Any mismatch invalidates the cache, removes the bad file, and
      re-downloads against the currently configured repo.

    Idempotent: a second call after a successful first call is a no-op
    aside from a "cache HIT" log line and a few file ``stat`` calls.
    """
    cache_dir = _resolve_cache_dir(cache_dir)
    _migrate_legacy_layout(cache_dir)
    marker = cache_dir / _MARKER_FILENAME

    # Which (filename, repo_id) tuples should this cache contain after the
    # call?  When the user overrides the fine-tune head, only the base file
    # is owned by the cache.
    expected_files: dict[str, str] = {_WAVLM_BASE_FILENAME: _WAVLM_BASE_REPO_ID}
    if finetune_checkpoint_override is None:
        expected_files[_FINETUNE_FILENAME] = _FINETUNE_REPO_ID

    if finetune_checkpoint_override is not None:
        finetune_checkpoint = Path(finetune_checkpoint_override).expanduser().resolve()
        if not finetune_checkpoint.is_file():
            raise FileNotFoundError(
                f"--similarity-checkpoint override not found: {finetune_checkpoint}"
            )
    else:
        finetune_checkpoint = cache_dir / _FINETUNE_FILENAME

    wavlm_base = cache_dir / _WAVLM_BASE_FILENAME

    # ---------- cache HIT path: validate every recorded file ----------
    cache_complete = False
    marker_data = _read_marker(marker) if marker.is_file() else None
    if marker_data is not None:
        recorded = marker_data.get("files", {})
        # The marker must describe exactly the files we plan to return — no
        # more, no less.  A mismatch means the override mode changed since
        # the marker was written, so we invalidate and rebuild.
        if isinstance(recorded, dict) and set(recorded) == set(expected_files):
            validations = []
            for filename, expected_repo_id in expected_files.items():
                entry = recorded.get(filename, {})
                ok, reason = _validate_asset(
                    cache_dir / filename,
                    expected_size=(
                        entry.get("size") if isinstance(entry, dict) else None
                    ),
                    expected_repo_id=expected_repo_id,
                    recorded_repo_id=(
                        entry.get("repo_id") if isinstance(entry, dict) else None
                    ),
                )
                validations.append((filename, ok, reason))
            cache_complete = all(ok for _, ok, _ in validations)
            if not cache_complete:
                for filename, ok, reason in validations:
                    if not ok:
                        logger.warning(
                            "[sim-assets] cache invalidation: %s — %s",
                            filename,
                            reason,
                        )

    if cache_complete:
        logger.info(f"[sim-assets] cache HIT at {cache_dir}")
        return SpeakerSimilarityAssets(
            finetune_checkpoint=finetune_checkpoint,
            wavlm_base=wavlm_base,
        )

    # ---------- cache MISS path: remove bad files + redownload ----------
    logger.info(f"[sim-assets] cache MISS at {cache_dir} — fetching")
    if marker.exists():
        marker.unlink()

    # If any file is on disk but currently fails validation against the
    # *expected repo* (without a marker-recorded size yet), remove it so
    # _download_and_validate redownloads from scratch.  This catches
    # truncated leftovers from earlier interrupted runs.
    for filename in expected_files:
        target = cache_dir / filename
        if target.exists():
            ok, reason = _validate_asset(target)
            if not ok:
                logger.warning(
                    "[sim-assets] removing stale file %s — %s", target, reason
                )
                target.unlink()

    files_info: dict[str, dict] = {}
    for filename, repo_id in expected_files.items():
        target = cache_dir / filename
        if not target.is_file():
            target = _download_and_validate(repo_id, filename, cache_dir)
        files_info[filename] = {
            "repo_id": repo_id,
            "size": target.stat().st_size,
        }

    _write_marker(marker, files_info)
    logger.info(f"[sim-assets] cached to {cache_dir}")

    return SpeakerSimilarityAssets(
        finetune_checkpoint=finetune_checkpoint,
        wavlm_base=wavlm_base,
    )


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Pre-download SeedTTS speaker-similarity assets into the cache "
            f"directory (override via {_CACHE_DIR_ENV})."
        ),
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Resolve and download all asset files into the cache directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved cache directory and intended downloads, "
        "without downloading anything.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Override cache directory (else uses {_CACHE_DIR_ENV} or "
        "~/.cache/sglang-omni/speaker_sim).",
    )
    args = parser.parse_args()

    cache_dir = _resolve_cache_dir(args.cache_dir)
    if args.dry_run:
        logger.info(f"[sim-assets] cache dir would be: {cache_dir}")
        logger.info(
            f"[sim-assets] would fetch {_FINETUNE_REPO_ID}/{_FINETUNE_FILENAME}"
        )
        logger.info(
            f"[sim-assets] would fetch {_WAVLM_BASE_REPO_ID}/{_WAVLM_BASE_FILENAME}"
        )
        return

    if not args.warm_cache:
        parser.error("pass --warm-cache to actually download, or --dry-run")

    assets = ensure_speaker_similarity_assets(cache_dir=args.cache_dir)
    logger.info(f"[sim-assets] finetune_checkpoint = {assets.finetune_checkpoint}")
    logger.info(f"[sim-assets] wavlm_base          = {assets.wavlm_base}")


if __name__ == "__main__":
    _main()
