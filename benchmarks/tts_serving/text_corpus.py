# SPDX-License-Identifier: Apache-2.0
"""Target-text corpora for TTS serving workloads."""

from __future__ import annotations

from pathlib import Path

SEEDTTS_TEXT_CORPUS_REPO = "zhaochenyang20/seed-tts-eval"
SEEDTTS_TEXT_CORPUS_REVISION = "8f5e1aa2a35d42f42e940074c1983358b9491f89"
SEEDTTS_TEXT_CORPUS_FILE = "en/meta.lst"


def load_seedtts_target_texts() -> tuple[str, ...]:
    """Load unique English target texts from pinned SeedTTS metadata."""
    from huggingface_hub import hf_hub_download

    metadata_path = hf_hub_download(
        SEEDTTS_TEXT_CORPUS_REPO,
        SEEDTTS_TEXT_CORPUS_FILE,
        repo_type="dataset",
        revision=SEEDTTS_TEXT_CORPUS_REVISION,
    )
    texts: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(metadata_path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) != 4 or not parts[3]:
            raise ValueError(
                f"Invalid SeedTTS metadata at line {line_number}: {line!r}"
            )
        target_text = parts[3]
        if target_text not in seen:
            seen.add(target_text)
            texts.append(target_text)
    return tuple(texts)
