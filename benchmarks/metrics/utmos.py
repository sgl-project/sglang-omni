# SPDX-License-Identifier: Apache-2.0
"""UTMOS MOS predictor for TTS quality evaluation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
import torchaudio
from filelock import FileLock
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_HF_REPO_ID = "balacoon/utmos"
_HF_FILENAME = "utmos.jit"
_CACHE_DIR_ENV = "UTMOS_CACHE_DIR"

_MARKER_FILENAME = ".utmos_cache.json"
_LOCK_FILENAME = ".utmos.lock"
_MARKER_SCHEMA_VERSION = 1
_MIN_ASSET_SIZE_BYTES = 10 * 1024 * 1024


def _resolve_utmos_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve cache dir: explicit arg → env var → default."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    env_val = os.environ.get(_CACHE_DIR_ENV)
    if env_val:
        return Path(env_val).expanduser().resolve()
    return Path("~/.cache/sglang-omni/utmos").expanduser().resolve()


def _read_marker(marker: Path) -> dict | None:
    """Parse the cache marker JSON, returning ``None`` on any error."""
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
    model_path: Path,
    marker_data: dict,
    expected_filename: str,
    expected_repo_id: str,
) -> bool:
    """Strict schema validation for a cached asset entry.

    All fields must be present and correct; any missing or mismatched field
    returns False and triggers a fresh download.
    """
    files = marker_data.get("files")
    if not isinstance(files, dict):
        return False
    entry = files.get(expected_filename)
    if not isinstance(entry, dict):
        return False
    recorded_repo_id = entry.get("repo_id")
    recorded_size = entry.get("size")
    if recorded_repo_id != expected_repo_id:
        return False
    if not isinstance(recorded_size, int) or recorded_size <= 0:
        return False
    if not model_path.is_file():
        return False
    return model_path.stat().st_size == recorded_size


def ensure_utmos_assets(cache_dir: str | Path | None = None) -> Path:
    """Download the UTMOS model weights to the cache directory if not present.

    Resolution order: explicit arg → ``UTMOS_CACHE_DIR`` env →
    ``~/.cache/sglang-omni/utmos``.

    Cache integrity is verified via a JSON marker file (``.utmos_cache.json``)
    that records the repo_id and file size recorded at download time.  A
    missing or malformed marker, a file below the 10 MiB minimum, a size
    mismatch, or a repo_id mismatch all trigger a fresh download.

    Returns the path to the validated model file.
    """
    resolved = _resolve_utmos_cache_dir(cache_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    logger.info(f"[utmos-assets] cache dir: {resolved}")

    model_path = resolved / _HF_FILENAME
    marker = resolved / _MARKER_FILENAME

    marker_data = _read_marker(marker) if marker.is_file() else None
    if marker_data is not None and _validate_asset(
        model_path, marker_data, _HF_FILENAME, _HF_REPO_ID
    ):
        logger.info(f"[utmos-assets] cache HIT at {resolved}")
        return model_path

    with FileLock(str(resolved / _LOCK_FILENAME)):
        marker_data = _read_marker(marker) if marker.is_file() else None
        if marker_data is not None and _validate_asset(
            model_path, marker_data, _HF_FILENAME, _HF_REPO_ID
        ):
            logger.info(f"[utmos-assets] cache HIT at {resolved} (post-lock)")
            return model_path

        logger.info(f"[utmos-assets] cache MISS at {resolved} — fetching")
        if marker.exists():
            marker.unlink()
        if model_path.exists():
            ok = (
                model_path.is_file()
                and model_path.stat().st_size >= _MIN_ASSET_SIZE_BYTES
            )
            if not ok:
                logger.warning(f"[utmos-assets] removing stale file {model_path}")
                model_path.unlink()

        local = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_HF_FILENAME,
            local_dir=str(resolved),
        )
        model_path = Path(local)

        actual_size = model_path.stat().st_size
        if actual_size < _MIN_ASSET_SIZE_BYTES:
            try:
                model_path.unlink()
            except OSError:
                pass
            raise RuntimeError(
                f"[utmos-assets] freshly downloaded {_HF_REPO_ID}/{_HF_FILENAME} "
                f"is only {actual_size} bytes — likely truncated"
            )

        _write_marker(
            marker,
            {_HF_FILENAME: {"repo_id": _HF_REPO_ID, "size": actual_size}},
        )
        logger.info(f"[utmos-assets] model weights ready: {model_path}")
        return model_path


class UTMOSScorer:
    """UTMOS MOS predictor backed by balacoon/utmos.

    Scores are in [1, 5].

    Args:
        device: CUDA device string (e.g. "cuda:0") or "cpu".
        cache_dir: Optional directory for the model cache.
            Defaults to ``UTMOS_CACHE_DIR`` env var or
            ``~/.cache/sglang-omni/utmos``.
    """

    def __init__(self, device: str, cache_dir: str | Path | None = None):
        model_path = ensure_utmos_assets(cache_dir)
        logger.info(f"Loading UTMOS from {model_path} ...")
        self.predictor = torch.jit.load(str(model_path), map_location=device)
        self.predictor.eval()
        self.device = device
        logger.info(f"UTMOS loaded on {device}")

    def score_batch(self, wav_paths: list[str]) -> list[float]:
        """Score a batch of wav files.

        Each file is loaded, resampled to 16 kHz if needed, converted to
        int16, and scored independently. Returns a list of float MOS scores
        in [1, 5].
        """
        scores: list[float] = []
        for path in wav_paths:
            wave, sr = torchaudio.load(path)
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
            wave_int16 = (wave * 32767).clamp(-32768, 32767).to(torch.int16)
            wave_int16 = wave_int16.to(self.device)
            with torch.no_grad():
                score = self.predictor(wave_int16)
            scores.append(float(score.mean().item()))
        return scores


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Pre-download UTMOS model weights into the cache directory "
            f"(override via {_CACHE_DIR_ENV})."
        )
    )
    parser.add_argument("--warm-cache", action="store_true")
    args = parser.parse_args()
    if args.warm_cache:
        ensure_utmos_assets()
