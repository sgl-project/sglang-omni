# SPDX-License-Identifier: Apache-2.0
"""Generate a fixed FLEURS Arabic set with Audar-TTS."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

AUDAR_REVISION = "51f5635f32de3ab45ff28a4b958464532225b247"
CODEC_REVISION = "30c1fdd19e68aee65d542cf043750d4c0165893e"
DATASET_REPO = "zhaochenyang20/sglang-omni-arabic-tts-smoke"
DATASET_REVISION = "65835c3a1047037f9e0cd4947652722c0a58c304"
DATASET_SIZE = 50
REFERENCE_FILE = "samples/demo_male_1_ar.wav"
REFERENCE_TEXT = (
    "لا يمكنني الانتظار لأخبرك — [excited] لقد أنجزنا المشروع أخيراً بعد كلّ "
    "هذا التعب، [laughs] وصدّقني، إنه أجمل شعورٍ على الإطلاق!"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="audarai/Audar-TTS-V1-Turbo")
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _load_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_REPO,
        split="test",
        revision=DATASET_REVISION,
    )
    if len(dataset) != DATASET_SIZE:
        raise ValueError(f"expected {DATASET_SIZE} dataset rows, found {len(dataset)}")
    if args.samples > len(dataset):
        raise ValueError(
            f"requested {args.samples} samples from a {len(dataset)}-row dataset"
        )
    return [dict(row) for row in dataset.select(range(args.samples))]


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call: Any) -> tuple[Any, float]:
    _sync()
    started = time.perf_counter()
    result = call()
    _sync()
    return result, time.perf_counter() - started


def _payload(
    *,
    request_id: str,
    target_text: str,
    reference_path: Path,
    seed: int,
    max_new_tokens: int,
) -> Any:
    from sglang_omni.proto import OmniRequest, StagePayload

    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={
                "text": target_text,
                "references": [
                    {
                        "audio_path": str(reference_path),
                        "text": REFERENCE_TEXT,
                    }
                ],
            },
            params={"max_new_tokens": max_new_tokens},
            metadata={"tts_params": {"seed": seed}},
        ),
        data={},
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_wav(path: Path, waveform: Any, sample_rate: int) -> None:
    import numpy as np

    pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def main() -> None:
    args = _parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    commit = _git_commit()

    import numpy as np

    from sglang_omni.models.audar_tts import stages
    from sglang_omni.models.audar_tts.payload_types import AudarTTSState

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    targets = _load_targets(args)

    reference_path = args.reference_path
    if reference_path is None:
        reference_path = Path(
            hf_hub_download(
                "audarai/Audar-TTS-V1-Turbo",
                REFERENCE_FILE,
                revision=AUDAR_REVISION,
            )
        )

    preprocessing = stages.create_preprocessing_executor()
    reference_encoder = stages.create_reference_encoder_executor(
        gpu_id=args.gpu_id,
        codec_revision=CODEC_REVISION,
    )
    tts_engine = stages.create_tts_engine_executor(
        args.model_path,
        gpu_id=args.gpu_id,
        model_revision=AUDAR_REVISION,
    )
    vocoder = stages.create_vocoder_executor(
        gpu_id=args.gpu_id,
        codec_revision=CODEC_REVISION,
    )

    samples: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for target in targets:
        started = time.perf_counter()
        sample = {**target, "is_success": False, "error": None}
        generated_entry = {
            "sample_id": target["sample_id"],
            "target_text": target["target_text"],
            "wav_path": "",
            "is_success": False,
            "latency_s": 0.0,
            "audio_duration_s": 0.0,
        }
        try:
            payload = _payload(
                request_id=target["sample_id"],
                target_text=target["target_text"],
                reference_path=reference_path,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            payload, preprocessing_s = _timed(lambda: preprocessing._fn(payload))
            payload, reference_s = _timed(lambda: reference_encoder._fn(payload))
            payload, engine_s = _timed(lambda: tts_engine._fn(payload))
            state = AudarTTSState.from_dict(payload.data)
            codes = np.asarray(state.audio_codes, dtype=np.int32)
            payload, vocoder_s = _timed(lambda: asyncio.run(vocoder._fn(payload)))
            waveform = np.frombuffer(payload.data["audio_waveform"], dtype=np.float32)
            sample_rate = int(payload.data["sample_rate"])
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise RuntimeError("invalid waveform")

            wav_path = audio_dir / f"{target['sample_id']}.wav"
            _write_wav(wav_path, waveform, sample_rate)
            duration_s = waveform.size / sample_rate
            total_s = preprocessing_s + reference_s + engine_s + vocoder_s
            sample.update(
                {
                    "is_success": True,
                    "audio_code_count": int(codes.size),
                    "audio_code_sha256": _sha256(codes.tobytes()),
                    "waveform_sample_count": int(waveform.size),
                    "waveform_sha256": _sha256(waveform.tobytes()),
                    "wav_sha256": _sha256(wav_path.read_bytes()),
                    "sample_rate": sample_rate,
                    "audio_duration_s": duration_s,
                    "completion_tokens": state.completion_tokens,
                    "reached_max_new_tokens": (
                        state.completion_tokens >= args.max_new_tokens
                    ),
                    "preprocessing_s": preprocessing_s,
                    "reference_s": reference_s,
                    "engine_s": engine_s,
                    "vocoder_s": vocoder_s,
                    "total_s": total_s,
                }
            )
            generated_entry.update(
                {
                    "wav_path": str(wav_path.resolve()),
                    "is_success": True,
                    "latency_s": total_s,
                    "audio_duration_s": duration_s,
                }
            )
        except Exception as exc:
            sample["error"] = str(exc)
            generated_entry["error"] = str(exc)
            generated_entry["latency_s"] = time.perf_counter() - started
        samples.append(sample)
        generated.append(generated_entry)

    result = {
        "schema_version": 1,
        "commit": commit,
        "model_path": args.model_path,
        "audar_revision": AUDAR_REVISION,
        "codec_revision": CODEC_REVISION,
        "reference_file": REFERENCE_FILE,
        "dataset": {
            "repo": DATASET_REPO,
            "split": "test",
            "revision": DATASET_REVISION,
            "available_samples": DATASET_SIZE,
            "selected_samples": len(targets),
            "source": {
                "repo": "google/fleurs",
                "config": "ar_eg",
                "split": "test",
                "revision": "ab93cf03f9d0cd083c853fad065a6377067408aa",
            },
        },
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "successful_samples": sum(sample["is_success"] for sample in samples),
        "truncated_samples": sum(
            sample.get("reached_max_new_tokens", False) for sample in samples
        ),
        "samples": samples,
    }
    (args.output_dir / "generation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "generated.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["successful_samples"] != args.samples or result["truncated_samples"]:
        raise RuntimeError(
            "quality generation had failed or max-token-truncated samples"
        )


if __name__ == "__main__":
    main()
