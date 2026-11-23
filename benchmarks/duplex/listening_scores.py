# SPDX-License-Identifier: Apache-2.0
"""Score the replies cut by listening_replies: omitted words, UTMOS and voice
consistency, each reply and pooled per dataset category.

Omission compares a completed reply's Parakeet text with the transcript the
server gave in response.done; a cancelled reply's transcript may run past the
audio it played, so it is not compared. UTMOS scores replies of at least
SCORED_DURATION_S. Voice consistency is the cosine similarity of the speaker
embedding of a reply's middle SCORED_DURATION_S with the mean embedding of the
same window of every other such reply in the repeat.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from jiwer import process_words
from pydantic import JsonValue
from whisper.normalizers import EnglishTextNormalizer

from benchmarks.duplex.listening_replies import REPLIES_FILE
from benchmarks.duplex.reference_core import read_json
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v15_audio import write_json
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from benchmarks.metrics.utmos import UTMOSScorer, ensure_utmos_assets

SCORES_FILE = "scores.json"
SUMMARY_FILE = "summary.json"
VOICE_WINDOW_DIR = "voice-window"
LOW_PERCENTILE = 10
# note (luojiaxuan): middle crops of replies of 6 s or more score within 0.07 MOS
# of the full reply from 2 s on, and 0.38 below it at 1 s. A speaker embedding
# depends on length at every duration measured, so voice similarity compares
# windows of one length; on MiniCPM-o replies, original and pitch-shifted copies
# separate by d = 2.7 with 2 s windows, 4.0 with 3 s and 5.0 with 4 s, while the
# replies long enough to score drop from 44 of 50 at 3 s to 38 at 4 s.
SCORED_DURATION_S = 3.0
# note (luojiaxuan): v1.5 pools its categories into "all", as its timing table
# does; v1.0 subsets are never pooled.
POOLED_DATASETS = ("v1.5",)
# note (luojiaxuan): the normalizer of the English WER in the ASR and TTS benchmarks.
NORMALIZE = EnglishTextNormalizer()


def word_errors(transcript: str, asr_text: str) -> dict[str, int] | None:
    """Alignment counts of the reply's audio text against its transcript, or
    None when the transcript has no words."""
    reference = NORMALIZE(transcript)
    if not reference.split():
        return None
    else:
        pass
    measures = process_words(reference, NORMALIZE(asr_text))
    return {
        "reference_words": measures.hits + measures.substitutions + measures.deletions,
        "substitutions": measures.substitutions,
        "deletions": measures.deletions,
        "insertions": measures.insertions,
    }


def leave_one_out_similarity(embeddings: torch.Tensor) -> list[float | None]:
    """Cosine similarity times 100 of each embedding with the mean of the other
    unit-length embeddings; None for all when fewer than two replies exist."""
    count = embeddings.shape[0]
    if count < 2:
        return [None] * count
    else:
        pass
    unit = F.normalize(embeddings.float(), dim=-1)
    others = (unit.sum(dim=0, keepdim=True) - unit) / (count - 1)
    return (F.cosine_similarity(unit, others, dim=-1) * 100.0).tolist()


def write_voice_window(listening_dir: Path, wav: str) -> Path:
    """The middle SCORED_DURATION_S of a reply, under voice-window/."""
    audio, rate = soundfile.read(listening_dir / wav, dtype="int16")
    length = round(SCORED_DURATION_S * rate)
    start = (len(audio) - length) // 2
    window = listening_dir / VOICE_WINDOW_DIR / wav
    window.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(window, audio[start : start + length], rate, subtype="PCM_16")
    return window


def score_replies(listening_dir: Path, device: str) -> list[dict[str, JsonValue]]:
    replies = read_json(listening_dir / REPLIES_FILE)["replies"]
    for reply in replies:
        reply["utmos"] = reply["voice_similarity"] = None
        if reply["status"] == "completed" and reply["transcript"] is not None:
            # note (luojiaxuan): a completed reply without audio said nothing.
            heard = "" if reply["asr_text"] is None else reply["asr_text"]
            reply["word_errors"] = word_errors(reply["transcript"], heard)
        else:
            reply["word_errors"] = None
    scored = [reply for reply in replies if reply["duration_s"] >= SCORED_DURATION_S]
    if not scored:
        return replies
    else:
        pass
    utmos = UTMOSScorer(device)
    assets = ensure_speaker_similarity_assets()
    speaker = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device=device,
    )
    embeddings = torch.cat(
        [
            speaker.embed([str(write_voice_window(listening_dir, reply["wav"]))]).cpu()
            for reply in scored
        ]
    )
    for reply, similarity in zip(
        scored, leave_one_out_similarity(embeddings), strict=True
    ):
        reply["utmos"] = utmos.score_batch([str(listening_dir / reply["wav"])])[0]
        reply["voice_similarity"] = similarity
    return replies


def low_percentile(values: list[float]) -> float | None:
    return float(np.percentile(values, LOW_PERCENTILE)) if values else None


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_group(replies: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    counted = [reply["word_errors"] for reply in replies if reply["word_errors"]]
    words = sum(errors["reference_words"] for errors in counted)
    utmos_scores = [reply["utmos"] for reply in replies if reply["utmos"] is not None]
    similarities = [
        reply["voice_similarity"]
        for reply in replies
        if reply["voice_similarity"] is not None
    ]
    # note (luojiaxuan): "missing" counts replies that never got a response.done.
    statuses = Counter(
        "missing" if reply["status"] is None else reply["status"] for reply in replies
    )
    return {
        "replies": len(replies),
        "replies_with_audio": sum(reply["wav"] is not None for reply in replies),
        "short_replies": sum(
            reply["wav"] is not None and reply["duration_s"] < SCORED_DURATION_S
            for reply in replies
        ),
        "statuses": dict(sorted(statuses.items())),
        "compared_replies": len(counted),
        "reference_words": words,
        "omission_rate": (
            sum(errors["deletions"] for errors in counted) / words if words else None
        ),
        "word_error_rate": (
            sum(
                errors["substitutions"] + errors["deletions"] + errors["insertions"]
                for errors in counted
            )
            / words
            if words
            else None
        ),
        "utmos_mean": mean(utmos_scores),
        f"utmos_p{LOW_PERCENTILE}": low_percentile(utmos_scores),
        "voice_similarity_mean": mean(similarities),
        f"voice_similarity_p{LOW_PERCENTILE}": low_percentile(similarities),
    }


def summarize(scored: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    datasets: dict[str, dict[str, list[dict[str, JsonValue]]]] = {}
    for reply in scored:
        groups = datasets.setdefault(reply["dataset"], {})
        groups.setdefault(reply["category"], []).append(reply)
        if reply["dataset"] in POOLED_DATASETS:
            groups.setdefault("all", []).append(reply)
        else:
            pass
    return {
        dataset: {
            category: summarize_group(replies)
            for category, replies in sorted(groups.items())
        }
        for dataset, groups in sorted(datasets.items())
    }


def run_score(listening_dir: Path, device: str) -> None:
    """Write scores.json with every reply's metrics and summary.json with the
    per-category pools and the files the models were loaded from."""
    scored = score_replies(listening_dir, device)
    assets = ensure_speaker_similarity_assets()
    write_json(listening_dir / SCORES_FILE, {"replies": scored})
    write_json(
        listening_dir / SUMMARY_FILE,
        {
            "asr": read_json(listening_dir / REPLIES_FILE)["asr"],
            "models": {
                "utmos_sha256": file_sha256(ensure_utmos_assets()),
                "speaker_finetune_sha256": file_sha256(assets.finetune_checkpoint),
                "speaker_wavlm_sha256": file_sha256(assets.wavlm_base),
            },
            "text_normalizer": "whisper.normalizers.EnglishTextNormalizer",
            "scored_duration_s": SCORED_DURATION_S,
            "datasets": summarize(scored),
        },
    )
