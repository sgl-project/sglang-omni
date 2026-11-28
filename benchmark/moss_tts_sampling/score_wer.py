"""Score generated WAVs with Whisper-large-v3 (FP32) and corpus WER.

Uses the Whisper EnglishTextNormalizer on both sides, greedy decoding forced to
language=en, task=transcribe. Audio is resampled to 16 kHz in-process.

Corpus error counts come from jiwer's alignment (exact substitutions,
deletions, insertions and hits) rather than from multiplying a rounded
per-sample WER by a word count. Missing or unreadable audio and decode failures
are recorded so a run with gaps cannot be reported as a clean score.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import jiwer
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper_normalizer.english import EnglishTextNormalizer


def load_audio(path: pathlib.Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    return audio


def word_errors(ref: str, hyp: str) -> dict:
    out = jiwer.process_words(ref, hyp)
    return {
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "hits": int(out.hits),
        "wer": float(out.wer),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = json.loads(pathlib.Path(args.manifest).read_text())
    wav_dir = pathlib.Path(args.wav_dir)

    processor = WhisperProcessor.from_pretrained(
        args.model_path, language="en", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    normalizer = EnglishTextNormalizer()
    per_sample = []
    failures = []
    pending = []

    items = []
    for row in rows:
        path = wav_dir / f"{row['index']:05d}_{row['sample_id']}.wav"
        if not path.exists():
            failures.append(
                {
                    "index": row["index"],
                    "sample_id": row["sample_id"],
                    "reason": "missing wav",
                }
            )
            continue
        try:
            items.append((row, load_audio(path)))
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            failures.append(
                {
                    "index": row["index"],
                    "sample_id": row["sample_id"],
                    "reason": "unreadable wav",
                    "error": repr(exc),
                }
            )

    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            chunk = items[start : start + args.batch_size]
            try:
                features = processor.feature_extractor(
                    [audio for _, audio in chunk],
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.to(args.device, dtype=torch.float32)
                generated = model.generate(
                    features,
                    language="en",
                    task="transcribe",
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=444,
                )
                texts = processor.batch_decode(generated, skip_special_tokens=True)
            except Exception as exc:  # noqa: BLE001 - keep batch visible, continue
                for row, _ in chunk:
                    failures.append(
                        {
                            "index": row["index"],
                            "sample_id": row["sample_id"],
                            "reason": "transcribe failed",
                            "error": repr(exc),
                        }
                    )
                continue
            for (row, _), text in zip(chunk, texts):
                pending.append((row, text))

    substitutions = deletions = insertions = hits = 0
    for row, hyp in pending:
        ref = normalizer(row["target_text"])
        hyp_n = normalizer(hyp)
        counts = word_errors(ref, hyp_n)
        substitutions += counts["substitutions"]
        deletions += counts["deletions"]
        insertions += counts["insertions"]
        hits += counts["hits"]
        per_sample.append(
            {
                "index": row["index"],
                "sample_id": row["sample_id"],
                "ref": ref,
                "hyp": hyp_n,
                "wer": round(counts["wer"], 4),
                "substitutions": counts["substitutions"],
                "deletions": counts["deletions"],
                "insertions": counts["insertions"],
                "hits": counts["hits"],
                "ref_words": counts["hits"]
                + counts["substitutions"]
                + counts["deletions"],
            }
        )

    error_words = substitutions + deletions + insertions
    ref_words = hits + substitutions + deletions
    corpus_wer = float(error_words / ref_words) if ref_words else None
    severe = sorted(
        (entry for entry in per_sample if entry["wer"] > 0.5),
        key=lambda entry: entry["wer"],
        reverse=True,
    )[:20]
    result = {
        "model_path": args.model_path,
        "device": args.device,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "decode": {
            "language": "en",
            "task": "transcribe",
            "num_beams": 1,
            "max_new_tokens": 444,
        },
        "num_samples": len(rows),
        "num_scored": len(per_sample),
        "success": len(failures) == 0 and len(per_sample) == len(rows),
        "failures": failures,
        "corpus_wer": corpus_wer,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "hits": hits,
        "error_words": error_words,
        "ref_words": ref_words,
        "mean_sample_wer": (
            round(float(np.mean([e["wer"] for e in per_sample])), 4)
            if per_sample
            else None
        ),
        "empty_hypotheses": sum(1 for entry in per_sample if not entry["hyp"]),
        "severe": severe,
        "per_sample": per_sample,
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("per_sample", "severe")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
