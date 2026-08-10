# SPDX-License-Identifier: Apache-2.0
"""Summarize direct Arabic ASR metrics for an Audar FLEURS run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from jiwer import process_characters, process_words
from sacrebleu.metrics import BLEU, CHRF

from benchmarks.tasks.asr import normalize_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--wer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quality_metrics(wer: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    generation_samples = generation["samples"]
    if generation["successful_samples"] != len(generation_samples) or any(
        not sample["is_success"] for sample in generation_samples
    ):
        raise ValueError("generation contains failed samples")
    if generation["truncated_samples"] or any(
        sample["reached_max_new_tokens"] for sample in generation_samples
    ):
        raise ValueError("generation contains truncated samples")

    generation_by_id = {sample["sample_id"]: sample for sample in generation_samples}
    if len(generation_by_id) != len(generation_samples):
        raise ValueError("generation contains duplicate sample IDs")

    asr_samples = wer["per_sample"]
    if any(not sample["is_success"] for sample in asr_samples):
        raise ValueError("ASR contains failed samples")
    asr_by_id = {sample["id"]: sample for sample in asr_samples}
    if len(asr_by_id) != len(asr_samples):
        raise ValueError("ASR contains duplicate sample IDs")
    if asr_by_id.keys() != generation_by_id.keys():
        raise ValueError("ASR and generation sample IDs differ")
    if len(asr_by_id) < 2:
        raise ValueError("Arabic quality evaluation requires at least two samples")

    references: list[str] = []
    hypotheses: list[str] = []
    for sample_id in sorted(asr_by_id):
        expected_reference = normalize_text(
            generation_by_id[sample_id]["target_text"], "ar"
        )
        sample = asr_by_id[sample_id]
        if sample["ref_norm"] != expected_reference:
            raise ValueError(f"ASR reference does not match target text: {sample_id}")
        references.append(expected_reference)
        hypotheses.append(sample["hyp_norm"])

    word_measures = process_words(references, hypotheses)
    recorded_wer = float(wer["summary"]["wer_corpus"])
    if not math.isclose(word_measures.wer, recorded_wer, rel_tol=1e-12):
        raise ValueError("recorded corpus WER does not match direct recomputation")

    cer_errors = 0
    cer_reference_characters = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        measures = process_characters(
            reference.replace(" ", ""), hypothesis.replace(" ", "")
        )
        cer_errors += measures.substitutions + measures.deletions + measures.insertions
        cer_reference_characters += (
            measures.substitutions + measures.deletions + measures.hits
        )

    bleu = BLEU(tokenize="none")
    return {
        "sample_count": len(asr_by_id),
        "asr_model": wer["config"]["asr_model"],
        "arabic_wer": word_measures.wer,
        "arabic_cer": cer_errors / cer_reference_characters,
        "arabic_bleu": bleu.corpus_score(hypotheses, [references]).score,
        "arabic_bleu_signature": str(bleu.get_signature()),
        "arabic_chrf_pp": CHRF(word_order=2)
        .corpus_score(hypotheses, [references])
        .score,
        "normalization": (
            "NFKC; Arabic diacritics, tatweel, and punctuation removed; "
            "Alef variants, alif maqsura, and digits normalized"
        ),
    }


def main() -> None:
    args = _parse_args()
    generation = _load(args.generation)
    metrics = _quality_metrics(_load(args.wer), generation)
    summary = {
        "schema_version": 1,
        "model_path": generation["model_path"],
        "audar_revision": generation["audar_revision"],
        "codec_revision": generation["codec_revision"],
        "dataset": generation["dataset"],
        "metrics": metrics,
        "scope": (
            "Informal Arabic intelligibility smoke benchmark; not a naturalness, "
            "prosody, speaker-similarity, or vendor comparison"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
